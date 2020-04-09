#!/usr/bin/env python3

import atexit
import dill
import getpass
import git
import hydra
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pdb
import signal
import sys
import tensorflow as tf
import time
import yaml

from carla.client import make_carla_client
from carla.tcp import TCPConnectionError

import precog.interface
import precog.utils.tfutil as tfutil
import precog.utils.rand_util as randu
import precog.utils.tensor_util as tensoru

import dim.env.run_carla_episode as run_carla_episode
import dim.env.plot.carla_plot as cplot
import dim.env.preprocess.carla_preprocess as preproc
import dim.env.util.carla_settings as carla_settings
import dim.env.util.agent_util as agent_util
import dim.env.util.agent_server as agent_server

import dim.plan.goal_distributions as goal_distributions
import dim.plan.pid_controller as pid_controller
import dim.plan.runtime_metrics as runtime_metrics
import dim.plan.dim_plan as dim_plan
import dim.plan.dim_controller as dim_controller

# Each user will need to configure their own gsheets credentials if desired.
if getpass.getuser() == 'nrhinehart': import dim.env.util.carla_gsheets as gu

log = logging.getLogger(os.path.basename(__file__))

# Nicer numpy printing
np.set_printoptions(precision=3, suppress=True, threshold=10)

@hydra.main(config_path='config/dim_config.yaml')
def main(cfg):
    print(' '.join(sys.argv))
    # Update the configs with the parsed cfg, in case the configs are used elsewhere.
    # util.postprocess_configs_with_cfg(cfg, [apilotcfg, apcfg, acfg, ecfg, cpcfg, cilcfg, dimcfg])
    postprocess_cfg(cfg)
    log.info('listening to server {}:{}'.format(cfg.server.host, cfg.server.port))
    run_and_complete_client(cfg)

def run_and_complete_client(cfg):
    log.info("Starting client")
    os.setpgrp()
    max_restarts = 5
    restarts = -1

    # Start at skip_to
    cfg.experiment.n_episodes_run = 0
    log.info("Total to run={}. Skip={}".format(cfg.experiment.n_episodes, cfg.experiment.skip_to))

    # While we haven't finished...
    while restarts < max_restarts:
        restarts += 1
        
        # Start the server.
        log.info("Starting server")
        pro, server_log_fn = agent_server.start_server(name=cfg.experiment.scene,
                                                       carla_dir=cfg.server.carla_dir,
                                                       server_log_dir=cfg.server.server_log_dir,
                                                       port=cfg.server.port)

        # Make sure the server dies when the program shuts down.
        log.info("Registering exit handler")
        atexit.register(os.killpg, pro.pid, signal.SIGKILL)
                        
        # Wait a bit for the server to spin up.
        time.sleep(2)

        try:
            ##############################
            # Try running the client.
            ##############################
            # We should skip to the right episode. e.g. ran 1 (ie. ep01), failed on 2, go to 2.
            log.info("Total run so far={}, Skip-to={}".format(cfg.experiment.n_episodes_run, cfg.experiment.skip_to))
            run_carla_client(cfg, server_log_fn)
            # If we run successfully, we're done.
            log.info("Completed run_carla_client")
            break
        # If we fail midway, we'll just report it and go back to the start of the loop.
        except TCPConnectionError as error:
            log.info("Caught error when trying to run a bunch of episodes: {}".format(error))
            # Kill the server and we'll try again...
            os.killpg(pro.pid, signal.SIGKILL)
            # Skip to the index of the failed run.
            cfg.experiment.skip_to = cfg.experiment.n_episodes_run
            continue
        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
            log.warning("Cancelled by user.")
            # If user wants, we're done. 
            break

    log.info("Exiting process.")
    sys.exit(0)

def run_carla_client(cfg, server_log_fn):
    pidconf = cfg.pid    
    dimconf = cfg.dim
    mainconf = cfg.main
    dataconf = cfg.data
    plotconf = cfg.plotting
    serverconf = cfg.server
    expconf = cfg.experiment
    waypointerconf = cfg.waypointer
    altcfg = cfg
    tag = _fmt_cfg(cfg)
    del cfg
    
    # Set the seed.
    log.debug("seed: {}".format(expconf.seed))
    log.info("Command-line arguments: {}".format(' '.join(sys.argv)))
    randu.seed(int(expconf.seed))

    # How many frames total to run for each episode.
    frames_per_episode = expconf.frames_per_episode

    # List of the start frame indices of each datum.
    starts = list(range(dataconf.record_wait_frames, frames_per_episode - dataconf.TT, dataconf.save_period_frames))

    # Total \approx n_frames / (save_period_seconds*Hz). Eg desired=1000 -> n_frames = 1000 * (5 * 2)
    log.info("Will collect {} examples per episode".format(len(starts)))
    
    if plotconf.plot:
        plt.close('all')
        if plotconf.hires_plot:
            fig, axes = cplot.create_agent_figure(figsize=(6,6), image_size=None, n_wide=2, remove_second_row=plotconf.remove_second_row)
        else:
            fig, axes = cplot.create_agent_figure(figsize=(3,3), image_size=None, n_wide=2, remove_second_row=plotconf.remove_second_row)
    else:
        fig, axes = None, None

    # Instantiate the object used to track all metrics.
    metrics = runtime_metrics.MultiepisodeMetrics(altcfg)
    model = None

    # Create and bundle the PID controllers.
    car_pid_controllers = pid_controller.CarPIDControllers.frompidconf(pidconf)
    lidar_params = preproc.LidarParams()
        
    if mainconf.pilot == 'dim':
        sess = tfutil.create_session(allow_growth=True, per_process_gpu_memory_fraction=.2)
        log.info("Loading DIM.")
        if dimconf.old_version:
            raise NotImplementedError("Old model loading not implemented yet")
        else:
            # Create model+inference network.
            _, _, tensor_collections = tfutil.load_annotated_model(dimconf.model_path, sess, dimconf.checkpoint_path)
            model = precog.interface.ESPInference(tensor_collections)        
            model_config = yaml.safe_load(open("{}/.hydra/config.yaml".format(dimconf.model_path),'r'))
            del tensor_collections
        
        phi = model.phi
        # TODO this is wrong, but we shouldn't need to use it if we're using the model?
        S_future_world_frame = None
        if mainconf.debug_post_load: pdb.set_trace()
        
        # Extract the sizes of the past and future trajectories.
        B, A, T_past, D = tensoru.shape(model.test_input.phi.S_past_world_frame)
        _B, K, _A, T, _D = tensoru.shape(model.sampled_output.rollout.S_world_frame)
        assert(_B == B and _A == A and _D == D)
        
        # Create goal likelihood.
        goal_likelihood = goal_distributions.create(model=model, dimconf=dimconf)
        
        # The object that generates plans.
        log.info("Not building multiagent planner")
        dim_planner = dim_plan.DIMPlanner(model, goal_likelihood=goal_likelihood, dimconf=dimconf, sess=sess)
        # Package the planner and the PID controllers into a single controller.
        midlow_controller = dim_controller.DIMJointMiddleLowController(
            dim_planner=dim_planner,
            model=model,
            replan_period=dimconf.replan_period,
            car_pid_controllers=car_pid_controllers,
            dimconf=dimconf)
    elif mainconf.pilot == 'auto':
        s = dataconf.shapes
        T = s.T
        T_past = s.T_past
        # Reset the graph here so that these names are always valid even if this function called multiple times by same program.
        tf.compat.v1.reset_default_graph()
        S_past_world_frame = tf.zeros((s.B, s.A, s.T_past, s.D), dtype=tf.float64, name="S_past_world_frame") 
        S_future_world_frame = tf.zeros((s.B, s.A, s.T, s.D), dtype=tf.float64, name="S_future_world_frame")
        yaws = tf.zeros((s.B, s.A), dtype=tf.float64, name="yaws")
        overhead_features = tf.zeros((s.B, s.H, s.W, s.C), dtype=tf.float64, name="overhead_features")
        agent_presence = tf.zeros((s.B, s.A), dtype=tf.float64, name="agent_presence")
        light_strings = tf.zeros((s.B,), dtype=tf.string, name="light_strings")
        phi = precog.interface.ESPPhi(
            S_past_world_frame=S_past_world_frame,
            yaws=yaws,
            overhead_features=overhead_features,
            agent_presence=agent_presence,
            light_strings=light_strings,
            feature_pixels_per_meter=lidar_params.pixels_per_meter,
            yaws_in_degrees=True)
        midlow_controller = None
    else: raise ValueError(mainconf.pilot)

    with make_carla_client(serverconf.host, serverconf.port, timeout=serverconf.client_timeout) as client:
        log.info('CarlaClient connected')
        root_dir = os.path.realpath(os.getcwd())
        # possible cleanup 
        # atexit.register(query_remove_logdir, root_dir)
        server_tag = '{:05d}'.format(int.from_bytes(os.urandom(2), 'big'))
        os.symlink(server_log_fn, root_dir + '/server_log_link_{}.log'.format(server_tag))

        # Save the git info.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        repo = git.Repo(dir_path, search_parent_directories=True)
        sha = repo.head.object.hexsha
        with open(root_dir + '/hexsha.txt', 'w') as f:
            f.write(sha + '\n' + 'dirty: {}\n'.format(repo.is_dirty()))

        log.info("Building settings programmatically")
        settings_carla = carla_settings.create_settings(
            n_vehicles=expconf.n_vehicles,
            n_pedestrians=expconf.n_pedestrians,
            quality_level=serverconf.quality_level,
            image_size=dataconf.image_size,
            use_lidar=dataconf.use_lidar,
            root_dir=root_dir,
            lidar_params=lidar_params)

        # TODO hack
        if 'val_obstacle' not in settings_carla.lidar_params:
            settings_carla.lidar_params['val_obstacle'] = 1.

        # Build the object that will process data and create feed_dicts for R2P2 forecasting.
        streaming_loader = preproc.StreamingCARLALoader(settings=settings_carla, T_past=T_past, T=T,
                                                        # TODO assumes model config
                                                        with_sdt=model_config['dataset']['params']['sdt_bev'])
        
        # Manage the things we plot.
        plottable_manager = cplot.PlottableManager(plotconf)

        # TODO using gsheets is hardcoded to my username. You can either configure it for yourself or ignore that functionality.
        record_google_sheets = mainconf.pilot != 'auto' and getpass.getuser() == 'nrhinehart' and expconf.record_google_sheets and gu.gsheets_util.have_pygsheets
        if record_google_sheets:
            gsheet_results = gu.MultiepisodeResults(tag=tag + 't_' + server_tag, multiep_metrics=metrics, root_dir=root_dir)

        for episode in range(0, expconf.n_episodes):
            log.info('Starting episode {}/{}'.format(episode, expconf.n_episodes))
            
            # Bundle the parameters of the episode.
            episode_params = agent_util.EpisodeParams(
                episode=episode,
                frames_per_episode=frames_per_episode,
                root_dir=root_dir,
                settings=settings_carla)
            
            # Start running the episode.
            run_carla_episode.run_episode(
                client=client,
                streaming_loader=streaming_loader,
                model=model,
                phi=phi,
                future=S_future_world_frame,
                midlow_controller=midlow_controller,
                car_pid_controllers=car_pid_controllers,
                plottable_manager=plottable_manager,
                waypointerconf=waypointerconf,
                episode_params=episode_params,
                metrics=metrics,
                fig=fig,
                axes=axes,
                cfg=altcfg)
            log.info("Episode {} complete".format(episode))
            
            if record_google_sheets: gsheet_results.update()
            # Store the fact that we've run another episode.
            expconf.n_episodes_run += 1

    # Save metrics once we're done with the episodes.
    with open(root_dir + "/final_metrics.dill".format(episode), 'wb') as f:
        dill.dump(metrics, f)
    with open(root_dir + "/metrics.dill".format(episode), 'wb') as f:
        dill.dump(metrics, f)
    log.info("Done with running CARLA client")
    # We're done. Leave the logdir.
    # atexit.unregister(query_remove_logdir)

def postprocess_cfg(cfg):
    cfgexp = cfg.experiment
    if cfgexp.n_vehicles > 0:
        if cfgexp.scene == 'Town02': assert(cfgexp.n_vehicles == 20)
        elif cfgexp.scene == 'Town01': assert(cfgexp.n_vehicles == 50)
        else: pass
            
    if len(cfgexp.specific_episodes) > 0:
        # Ensure we'll actually get to run
        if cfgexp.n_episodes < max(cfgexp.specific_episodes) - 1:
            raise ValueError("Refusing to run for less episodes than needed to run the largest specific episode index")

    assert(cfg.dim.min_planning_steps <= cfg.dim.max_planning_steps)
    if cfg.dim.goal_likelihood == 'RegionIndicator':
        cfg.waypointer.region_control = True

    # Set a unique tag for everything that happens next. Useful if an episode crashes --
    #   things will resume from that episode but still tagged with this tag.
    cfgexp.rstr = '{:05d}'.format(int.from_bytes(os.urandom(2), 'big'))

    mpl_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cfg.plotting.colors = mpl_cycle
    cfg.plotting.expert_color = mpl_cycle[0]
    cfg.plotting.generator_color = mpl_cycle[3]
    cfg.plotting.past_color = mpl_cycle[1]
    cfg.plotting.cw_colors = cm.coolwarm(np.linspace(1., 0., 120)).tolist()

def query_remove_logdir(logdir):
    yesf = lambda x: x.lower().startswith('y')
    nof = lambda x: x.lower().startswith('n') or x is None or len(x) == 0
    while True:
        inp = input("Remove '{}'? [y/N]".format(logdir))
        if yesf(inp) and os.path.exists(logdir):
            print("doing shutil.rmtree(logdir)")
        elif nof(inp):
            break
        else:
            pass

def _fmt_cfg(cfg):
    dimconf = cfg.dim    
    expconf = cfg.experiment
    waypointerconf = cfg.waypointer
    return '{}_gl-{}_dng-{}_seed-{}_{}'.format(
        expconf.scene,
        dimconf.goal_likelihood,
        waypointerconf.drop_near_on_green,
        expconf.seed,
        expconf.rstr)

    
if __name__ == '__main__':
    # warnings.filterwarnings("default", category=RuntimeWarning)
    main()
