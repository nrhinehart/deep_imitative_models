
import attrdict
import collections
import dill
import logging
import numpy as np
import os
import pdb
import random
import time

import carla.transform as transform

import precog.utils.np_util as npu

import dim.env.plot.carla_plot as cplot
import dim.env.preprocess.carla_preprocess as preproc
import dim.env.util.agent_util as agent_util
import dim.plan.autopilot_controller as autopilot_controller
from dim.plan.waypointer import Waypointer

log = logging.getLogger(os.path.basename(__file__))

def run_episode(client,
                streaming_loader,
                model,
                phi,
                future,
                midlow_controller,
                car_pid_controllers,
                plottable_manager,
                waypointerconf,
                episode_params,
                metrics,
                fig,
                axes,
                cfg):
    dimconf = cfg.dim
    mainconf = cfg.main
    dataconf = cfg.data
    plotconf = cfg.plotting
    trackersconf = cfg.trackers
    expconf = cfg.experiment
    waypointerconf = cfg.waypointer
    del cfg
    
    # Build dirs.
    directory, plot_dir, dim_feeds_dir = set_up_directories(episode_params, cfg=plotconf, mainconf=mainconf, dataconf=dataconf)

    # Dump some objects.
    save_metadata(episode_params=episode_params, directory=directory, metrics=metrics)

    # Print the metrics.
    if episode_params.episode > 0: log.info("Current overall metrics: {}".format(metrics))

    # Randomize the environment.
    episode_params.settings.randomize_seeds()
    
    if midlow_controller is not None: midlow_controller.reset()
    if model is not None:
        assert(phi is model.phi)

    # Now we load these settings into the server. The server replies
    # with a scene description containing the available start spots for
    # the player. Here we can provide a CarlaSettings object or a
    # CarlaSettings.ini file as string.
    log.info("Loading the settings into the server...")
    scene = client.load_settings(episode_params.settings)
    
    # Hang out?
    time.sleep(0.01)

    # Notify the server that we want to start the episode at the
    # player_start index. This function blocks until the server is ready
    # to start the episode.
    client.start_episode(choose_player_start(scene))

    have_specific_episodes = len(expconf.specific_episodes) > 0
    if not have_specific_episodes:
        # Possibly skip the episode.
        if episode_params.episode < expconf.skip_to:
            log.warning("SKIPPING EPISODE {}\n".format(episode_params.episode) * 10)
            return
    else:
        log.info("Filtering episodes to '{}'".format(expconf.specific_episodes))
        if episode_params.episode not in expconf.specific_episodes:
            log.info("{} not in specific episodes".format(episode_params.episode))
            return
    
    # Reset the plotting.
    if plotconf.plot: _ = [ax.cla() for ax in axes.ravel()]
    plot_state = cplot.PlotState.from_plottables(plotconf, plottable_manager.base_plottables) if plotconf.plot else None

    # Hang out for a bit, see https://github.com/carla-simulator/carla/issues/263
    time.sleep(4)

    # Collect the recent measurements and controls.
    measurement_buffer = collections.deque(maxlen=dataconf.measurement_buffer_length)

    # Instatiate waypointer.
    waypointer = Waypointer(waypointerconf=waypointerconf, planner_or_map_name=expconf.scene)

    # For now, just track whether the robot is stationary.
    stop_tracker = agent_util.StationaryTracker(
        size=trackersconf.stationary_window_size, thresh=trackersconf.stationary_threshold, track_others=False)
    # Use this to detect if agent is stationary longer than it should be (e.g. 100frames->10 seconds).
    # Should be beyond the max light time and before the final timeout time (stop tracker).
    stuck_tracker = agent_util.NonredStuckTracker(
        waypointer=waypointer, size=trackersconf.stuck_window_size, thresh=trackersconf.stationary_threshold)

    # If in the past N frames, the car's pose has changed by more than M degrees, then it's executing a turn.
    turn_tracker = agent_util.TurnTracker(size=trackersconf.turn_tracker_frames, thresh=trackersconf.turn_tracker_thresh)

    # Reset the controllers.
    car_pid_controllers.reset()

    # Initially not collided.
    sum_collision_impulse = 0

    # ---------------------------------------
    # TODO incorporate pasts of other agents!
    # ---------------------------------------    
    log.warning("Defaulting to a hardcoded A past of 1")    
    A_past = 5

    # Ensure all previous episodes were properly concluded.
    for em in metrics.all_episode_metrics: assert(em.concluded)
    log.info("Current number of successes and failures: {}, {}".format(
        metrics.n_successes, metrics.n_failures))

    # Create new episode metric
    metrics.begin_new_episode(episode_params)

    waypointerconf.have_planned = False

    # Step through the simulation.
    for frame in range(0, episode_params.frames_per_episode):
        log.debug("On frame {:06d}".format(frame))
        if frame == episode_params.frames_per_episode - 1:
            summary = "Ran out of episode frames! Treating as failure"
            log.warning(summary)
            metrics.conclude_episode(success=False, summary=summary)
            with open(episode_params.root_dir + "/metrics.dill".format(episode_params.episode), 'wb') as f:
                dill.dump(metrics, f)
            break
        have_control = frame > expconf.min_takeover_frames
        
        # -----------------------
        # Read the data produced by the server this frame. Record measurements.
        # -----------------------
        measurement, sensor_data = client.read_data()
        measurement_buffer.append(measurement)
        sensor_data = attrdict.AttrDict(sensor_data)
        # Prevent sensor data mutation.
        agent_util.lock_observations(sensor_data)
        transform_now  = transform.Transform(measurement.player_measurements.transform).inverse()

        if mainconf.pilot != 'auto':
            waypoints, waypoints_local = waypointer.get_waypoints_from_transform_and_measurement(transform_now, measurement)

            if waypoints is None or waypoints_local is None:
                summary = "Waypointer error! Episode failed"
                log.error(summary)
                metrics.conclude_episode(success=False, summary=summary)
                break
        else:
            waypoints, waypoints_local = None, None

        # Some measurement-only computations
        if have_control:
            # Index the measurement to check if we've stopped.
            stop_tracker.index_new_measurement(measurement)
            stuck_tracker.index_new_measurement(measurement)
            stationary, stuck = stuck_tracker.is_stuck_far_from_red_lights(ignore_reds=waypointerconf.stuck_ignore_reds)
            if stuck: log.warning("Determined that the vehicle is stuck not near a red light!")
        else:
            stationary, stuck = False, False
                
        turn_tracker.index_new_measurement(measurement)            

        # -----------------------
        # Instantiate the object that represents all observations at the current time.
        # -----------------------
        current_obs = preproc.PlayerObservations(
            measurement_buffer, t_index=-1, radius=200, A=A_past, waypointer=waypointer, frame=frame)
        # Store some tracking data in the observation.
        current_obs.is_turning = turn_tracker.is_turning()
        current_obs.is_stuck = stuck
        current_obs.is_stationary = stationary
        transform_now = current_obs.inv_tform_t  # world2local
        if frame == 0:
            start_obs = current_obs
        log.debug("Episode={}, Current frame={}".format(episode_params.episode, current_obs.frame))

        # Populate extra metrics.
        extra_metrics = {}
        traffic_light_state, traffic_light_data = waypointer.get_upcoming_traffic_light(measurement, sensor_data)

        log.debug("Upcoming traffic light is: '{}'.".format(traffic_light_state))
        current_obs.traffic_light_state = traffic_light_state
        current_obs.traffic_light_data = traffic_light_data
        extra_metrics['red_light_violations'] = waypointer.red_light_violations
        extra_metrics['intersection_count'] = waypointer.intersection_count

        if trackersconf.reset_trackers_on_red and traffic_light_state == 'RED':
            log.debug("Resetting trackers because we observed a RED light")
            stop_tracker.reset()
            stuck_tracker.reset()
            
        # Prune near and far waypoints, possibly perturb.
        if mainconf.pilot != 'auto':
            # pdb.set_trace()
            waypoints_control, waypoint_metadata = waypointer.prepare_waypoints_for_control(waypoints_local, current_obs, dimconf=dimconf)
            npu.lock_nd(waypoints_control)
        else:
            waypoints_control = None
            waypoint_metadata = {}
            
        # Reset plotter.
        plottable_manager.reset()
            
        if mainconf.pilot != 'auto':
            # Check if we have enough waypoints to continue, and if we've stopped.
            episode_state, summary = agent_util.decide_episode_state(
                waypointer, waypoints, stop_tracker, start_obs, current_obs, goal_distance_threshold=expconf.goal_distance_threshold)
            if episode_state == agent_util.EpisodeState.SUCCESS:
                metrics.conclude_episode(success=True, summary=summary)
                break
            elif episode_state == agent_util.EpisodeState.FAILURE:
                metrics.conclude_episode(success=False, summary=summary)
                break
            elif episode_state == agent_util.EpisodeState.INPROGRESS:
                pass
            else:
                raise ValueError('unknown episode state')
                
        # ------------------
        # Update the metrics
        # ------------------
        metrics.update(measurement, extra_metrics)
        if have_control:
            metrics.update_passenger_comfort(current_obs)

        # Periodically print 
        if episode_params.episode > 0 and frame % 1000 == 0:
            log.info("Intermediate print... current overall metrics: {}".format(metrics))
            
        # Check for collision.
        if check_for_collision(measurement, sum_collision_impulse, episode_params, metrics, frame, player_transform=current_obs.inv_tform_t,
                               allow_vehicles_to_hit_us_from_behind=expconf.allow_vehicles_to_hit_us_from_behind):
            # Create control anyway in case server is looking for it?
            control = autopilot_controller.noisy_autopilot(
                measurement, replan_index=dimconf.replan_period, replan_period=dimconf.replan_period, cfg=dataconf)
            # Send the control.
            client.send_control(control)
            # Quit the episode.
            break
        
        # Update current total collision impulse.
        sum_collision_impulse = measurement.player_measurements.collision_other

        # Get the feed_dict for this frame to build features / plottables.
        # TODO should be T_past, not 3.
        if frame > 3:
            fd = streaming_loader.populate_phi_feeds(
                phi=phi,
                sensor_data=sensor_data,
                measurement_buffer=measurement_buffer,
                with_bev=True,
                with_lights=True,
                observations=current_obs,
                frame=frame)
            if have_control and dataconf.save_data:
                fd_previous = streaming_loader.populate_expert_feeds(current_obs, future, frame)
                if frame % dataconf.save_period_frames == 0:
                    fn = "{}/feed_{:08d}.json".format(dim_feeds_dir, frame)
                    log.debug("Saving feed to '{}'".format(fn))
                    preproc.dict_to_json(fd_previous, fn)
                streaming_loader.prune_old(frame)
                    
        # If we have a non-autopilot controller, and enough frames                 
        if have_control and mainconf.pilot not in ('auto', 'user', 'zero'):
            prune_nearby = current_obs.is_turning and waypointerconf.drop_near_on_turn
            if stuck or prune_nearby:
                log.warning("Vehicle is stuck or turning, preparing to adjust goal likelihood")                
                waypoints_control = waypointer.get_unsticking_waypoints(waypoints_control, midlow_controller, current_obs)
            if mainconf.save_dim_feeds:
                fn = "{}/feed_{:08d}.json".format(dim_feeds_dir, frame)
                preproc.dict_to_json(fd, fn)
            if mainconf.override_pilot:
                log.debug("Building autopilot control")
                # If not in controlled mode, or not in controlled mode yet, use autopilot.     
                control = autopilot_controller.noisy_autopilot(
                    measurement, replan_index=dimconf.replan_period, replan_period=dimconf.replan_period, cfg=dataconf)
                approach_plotting_data = None
                have_control = False
            else:
                # Generate the control for the pilot.
                control, approach_plotting_data = generate_control(
                    mainconf.pilot,
                    midlow_controller=midlow_controller,
                    car_pid_controllers=car_pid_controllers,
                    model=model,
                    fd=fd,
                    measurement=measurement,
                    waypoints_local=waypoints_control,
                    transform_now=transform_now,
                    waypointer=waypointer,
                    current_obs=current_obs,
                    sensor_data=sensor_data,
                    waypoint_metadata=waypoint_metadata)
        else:
            log.debug("Building autopilot control")
            # If not in controlled mode, or not in controlled mode yet, use autopilot.     
            control = autopilot_controller.noisy_autopilot(
                measurement, replan_index=dimconf.replan_period, replan_period=dimconf.replan_period, cfg=dataconf)
            approach_plotting_data = None

        log.debug("Control: {}".format(control))
        client.send_control(control)
        waypointerconf.have_planned = have_control

        if plotconf.plot and frame > 3:
            plot_data = cplot.GenericPerFramePlottingData(
                pilot=mainconf.pilot,
                hires=plotconf.hires_plot,
                measurement=measurement,
                feed_dict=fd,
                waypoints_local=waypoints_local,
                waypoints_control=waypoints_control,
                waypoint_metadata=waypoint_metadata,
                transform_now=transform_now,
                current_obs=current_obs,
                control=control)

            if have_control:
                cplot.update_pilots_plot(mainconf.pilot, plottable_manager, plot_data, approach_plotting_data)
            else:
                cplot.update_pilots_plot('auto', plottable_manager, plot_data, approach_plotting_data)
            
            # plottable_manager.update_from_observation(current_obs, plot_data=plot_data, control=control)
            log.debug("Plotting data")
            cplot.online_plot(model=model,
                              measurements=measurement,
                              sensor_data=sensor_data,
                              overhead_lidar=fd[phi.overhead_features][0,...,0],
                              overhead_semantic=None,
                              plottables=plottable_manager.plottables,
                              fd=fd,
                              axes=axes,
                              fig=fig,
                              plot_state=plot_state)
            if plotconf.save_plots:
                log.debug("Saving plot")
                # pdb.set_trace()
                # fig.savefig('{}/plot_{:08d}.jpg'.format(plot_dir, frame), dpi=180)

                # Get the extent of the specific axes we'll save.
                if plotconf.remove_second_row:
                    joint_extent = cplot.full_extent([axes[1,0], axes[0,1]])
                    extent0 = joint_extent.transformed(fig.dpi_scale_trans.inverted())
                    # fig.savefig('{}/plot_{:08d}.png'.format(plot_dir, frame), bbox_inches=extent0, dpi=cfg.dpi*2)
                    fig.savefig('{}/plot_{:08d}.{}'.format(plot_dir, frame, plotconf.imgfmt), bbox_inches=extent0, dpi=plotconf.dpi*2)
                else:
                    # fig.savefig('{}/plot_{:08d}.png'.format(plot_dir, frame), dpi=plotconf.dpi)
                    fig.savefig('{}/plot_{:08d}.{}'.format(plot_dir, frame, plotconf.imgfmt), dpi=plotconf.dpi)
        else:
            pass

        if frame % episode_params.snapshot_frequency == 0:
            log.info("Saving visual snapshot of environment. Frame={}".format(frame))
            # Serialize the sensor data to disk.
            for name, datum in sensor_data.items():
                if name == 'CameraRGB':
                    filename = os.path.join(directory, '{}_{:06d}'.format(name, frame))
                    datum.save_to_disk(filename)

def set_up_directories(episode_params, cfg, mainconf, dataconf):
    directory = '{}/episode_{:06d}/'.format(episode_params.root_dir, episode_params.episode)
    os.makedirs(directory)
    if mainconf.save_dim_feeds or dataconf.save_data:
        dim_feeds_dir = directory + "/dim_feeds/"
        os.makedirs(dim_feeds_dir)
    else:
        dim_feeds_dir = None

    if cfg.plot and cfg.save_plots:
        plot_dir = directory + '/plots/'
        os.mkdir(plot_dir)
    else:
        plot_dir = None
    return directory, plot_dir, dim_feeds_dir

def save_metadata(episode_params, directory, metrics):
    with open(directory + "/metrics_before_episode_{:08d}.dill".format(episode_params.episode), 'wb') as f:
        dill.dump(metrics, f)
    with open(episode_params.root_dir + "/metrics.dill", 'wb') as f: 
        dill.dump(metrics, f)

def choose_player_start(scene):
    """Generates the starting index of the player's vehicle.

    :param scene: 
    :param cfg: 
    :returns: 
    :rtype: 

    """
    number_of_player_starts = len(scene.player_start_spots)
    log.debug("N player starts: {}".format(number_of_player_starts))
    player_start = random.randint(0, max(0, number_of_player_starts - 1))
    log.debug("Player start index: {}".format(player_start))
    return player_start

def check_for_collision(
        measurement,
        sum_collision_impulse,
        episode_params,
        metrics,
        frame,
        allow_vehicles_to_hit_us_from_behind=False,
        player_transform=None,
        last_collision_impulses={'vehicle': 0.0, 'person': 0.0}):
    log.debug("Collision impulse {:.3f}".format(measurement.player_measurements.collision_other))
    # Only reset if hit vehicles or people.
    impulse_delta_v = measurement.player_measurements.collision_vehicles - last_collision_impulses['vehicle']
    impulse_delta_p = measurement.player_measurements.collision_pedestrians - last_collision_impulses['person']
    
    hit_vehicle = impulse_delta_v > 0
    hit_person = impulse_delta_p > 0

    last_collision_impulses['vehicle'] = measurement.player_measurements.collision_vehicles
    last_collision_impulses['person'] = measurement.player_measurements.collision_pedestrians

    vehicle_behind = is_closest_vehicle_behind_us(measurement, player_transform)
    
    if hit_vehicle and vehicle_behind:
        log.warning("A vehicle hit the player from behind!")
        if allow_vehicles_to_hit_us_from_behind:
            log.warning("Not counting it as a collision! It's the other vehicle's fault!")
            hit_vehicle = False
    
    if hit_vehicle or hit_person:
        summary = "Player is in collision! Resetting the episode"
        log.info(summary)
        metrics.conclude_episode(success=False, summary=summary)
        metrics_filename = episode_params.root_dir + "/metrics.dill".format(episode_params.episode)
        with open(metrics_filename, 'wb') as f:
            dill.dump(metrics, f)
        return True
    return False

def is_closest_vehicle_behind_us(measurement, player_transform):
    vehicle_positions = []
    for agent in measurement.non_player_agents:
        if agent.HasField('vehicle'):
            pos = preproc.vector3_to_np(agent.vehicle.transform.location)
            vehicle_positions.append(pos)
    other_positions_world = np.stack(vehicle_positions, axis=0)
    other_positions_local = player_transform.transform_points(other_positions_world)
    xy_distances = np.linalg.norm(other_positions_local[:,:2], axis=-1)
    nearest_vehicle_index = np.argmin(xy_distances)
    nearest_vehicle_position_local = other_positions_local[nearest_vehicle_index]
    x = nearest_vehicle_position_local[0]
    if x < -0.01:
        return True
    else:
        return False
 
pid_waypoint_index = 0

def generate_control(pilot,
                     midlow_controller,
                     car_pid_controllers,
                     model,
                     fd,
                     measurement,
                     waypoints_local,
                     transform_now,
                     waypointer,
                     current_obs,
                     sensor_data,
                     waypoint_metadata,
                     force_replan=False):
    """Decide controls for the pilot

    :param pilot: str pilot name
    :param midlow_controller: possible joint mid-low controller
    :param model: inference.Model
    :param fd: feed_dict
    :param measurement: carla Measurement
    :param waypoints_local: ndarray of waypoints in local coords
    :param transform_now: transform of robot at time t
    :param plottable_manager: PlottableManager inst
    :param waypointer: Waypointer inst
    :returns: carla Control
    """
    approach_plotting_data = None
    
    if pilot == 'dim':
        # Generate the plan and the control with DIM.
        dim_plan, control = midlow_controller.generate_mid_and_low_level_controls(
            measurement=measurement,
            transform_now=transform_now,
            waypoints_local=waypoints_local,
            current_obs=current_obs,
            fd=fd,
            waypoint_metadata=waypoint_metadata,
            force_replan=force_replan)
        approach_plotting_data = cplot.DIMPlottingData(plan=dim_plan, midlow_controller=midlow_controller)
    else:
        raise ValueError("Unknown pilot '{}'".format(pilot))
    return control, approach_plotting_data
