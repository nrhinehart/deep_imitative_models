
import logging
import numpy as np
import os
import pdb

from carla.carla_server_pb2 import Control as VehicleControl
import precog.utils.class_util as classu

log = logging.getLogger(os.path.basename(__file__))

class DIMJointMiddleLowController:
    @classu.member_initialize
    def __init__(self, dim_planner, model, replan_period, car_pid_controllers, dimconf):
        log.info("Instantiating {}".format(self.__class__.__name__))
        self.plan_index = replan_period
        self.total_plan_count = 0

    def reset(self):
        self.dim_plan = None
        self.plan_index = self.replan_period
        self.n_consecutive_risky = 0
        
    def generate_mid_and_low_level_controls(self,
                                            measurement,
                                            transform_now,
                                            waypoints_local,
                                            fd,
                                            current_obs=None,
                                            waypoint_metadata=None,
                                            force_replan=False,
                                            debug=True):
        log.info("Running prior sampling + inference ... likelihoods should be high")
        
        if debug:
            Z = self.dim_planner.sess.run(self.dim_planner.Z)
            fd[self.dim_planner.Z] = Z
            logq_x_np0 = self.dim_planner.sess.run([self.dim_planner.posterior_components.log_prior], fd)[0]
            logq_x_np1 = self.dim_planner.sess.run([self.dim_planner.posterior_components.log_prior], fd)[0]
            assert(np.equal(logq_x_np0, logq_x_np1).all())
            log.info("logq_x_np max {:.1f}".format(logq_x_np0.max()))

        # Re-plan.
        index_replan = self.plan_index == self.replan_period
        if index_replan or force_replan:
            if force_replan and not index_replan:
                log.info("Forcing replan off-index!")
            self.dim_plan = self.dim_planner.plan(waypoint_states=waypoints_local, feed_dict=fd)
            self.total_plan_count += 1
            self.plan_index = 0
            self.transform_plan_start = transform_now
            self.yaw_plan_start = measurement.player_measurements.transform.rotation.yaw
            # TODO HACK
            self.dim_plan.waypoint_metadata = waypoint_metadata
        # Use existing plan.
        else:
            log.info("Using existing plan")
        self.plan_index += 1                        

        yaw_now = measurement.player_measurements.transform.rotation.yaw
        transform_before_to_now = transform_now * self.transform_plan_start.inverse()

        control = self.dim_pid_controller(
            dim_plan=self.dim_plan,
            plan_index=self.plan_index,
            steer_controller=self.car_pid_controllers.steer_controller,
            throttle_controller=self.car_pid_controllers.throttle_controller,
            brake_controller=self.car_pid_controllers.brake_controller,
            transform_before_to_now=transform_before_to_now,
            yaw_now=yaw_now,
            yaw_plan_start=self.yaw_plan_start,
            current_obs=current_obs)
        return self.dim_plan, control

    def dim_pid_controller(self,
                           dim_plan,
                           plan_index,
                           steer_controller,
                           throttle_controller,
                           brake_controller,
                           transform_before_to_now,
                           yaw_now,
                           yaw_plan_start,
                           current_obs):
        # control = carla_protocol.Control()
        control = VehicleControl()

        measurement = current_obs.measurements[-1]
        p = measurement.player_measurements
        current_forward_speed = p.forward_speed

        # Use offset of -1 to account for already incrementing plan_index
        plan_position_tform_start = dim_plan.get_plan_position(
            t_future_index=self.dimconf.position_setpoint_index + plan_index - 1)

        # Transform the waypoint in the old coordinates to the current.
        plan_position_tform_start_3d = np.concatenate((plan_position_tform_start, [0]))
        position_setpoint = (transform_before_to_now.transform_points(
            plan_position_tform_start_3d[None]))[0]

        # Get the x coord of the position setpoint in the local car coords of current time.
        # How many meters at a specific point in the future we should be from where we are now.
        forward_setpoint = position_setpoint[0]
        
        # The plan positions are time-profiled. Hardcoded here to assume 5Hz.
        plan_Hz = 10
        plan_period = 1./plan_Hz
        T = 20
        time_points = np.arange(plan_period, plan_period*T, plan_period)
        # How many seconds the target spatiotemporal forward point is in the future.
        time_offset = time_points[self.dimconf.position_setpoint_index]
        
        # The target forward speed along the orientation.
        target_forward_speed = forward_setpoint / time_offset

        # [TODO] Hacking this in for plotting purposes.
        dim_plan.current_target_forward_speed = target_forward_speed
        dim_plan.current_forward_speed_error = target_forward_speed - current_forward_speed
        dim_plan.current_target_forward_displacement = forward_setpoint
        dim_plan.plan_step = self.plan_index
        dim_plan.total_plan_count = self.total_plan_count

        # Use the headings derived by aiming towards points in the plan.
        # Use offset of -1 to account for already incrementing plan_index
        angle_position_tform_start = dim_plan.get_plan_position(
            t_future_index=min(self.dimconf.angle_setpoint_index + plan_index - 1, self.model.metadata.T - 1))

        # Transform the waypoint in the old coordinates to the current.
        angle_position_tform_start_3d = np.concatenate(
            (angle_position_tform_start,[0]))
        # Get the current position to aim to in current car's coords.
        angle_setpoint_tform = (transform_before_to_now.transform_points(
            angle_position_tform_start_3d[None]))[0]

        # Force x coordinates in front.
        plan_angle_plancoords = np.arctan2(angle_setpoint_tform[1], abs(angle_setpoint_tform[0]))
        current_angle_plancoords = 0.

        # Steer controller is controlling to the target angle in the plan.
        steer_unsnapped = steer_controller.update(target_angle=plan_angle_plancoords, current_angle=current_angle_plancoords)

        # Update the PID controllers.
        throttle = throttle_controller.update(setpoint=target_forward_speed, process_variable=current_forward_speed)

        need_to_brake = (target_forward_speed - current_forward_speed) < 0.
        if need_to_brake:
            brake = -throttle
            throttle = 0.0
        else:
            throttle = throttle
            brake = 0.0
            
        # Clip the steering to the valid steering range
        steer = max(-1., min(1., steer_unsnapped))
        brake = max(min(brake, 1.0), 0.0)
        throttle = min(max(throttle, 0.0), 1.0)

        # [TODO] hardcoded off.
        hand_brake = False
        reverse = False

        # Set the attributes of the control message.
        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        control.hand_brake = hand_brake
        control.reverse = reverse
        return control
