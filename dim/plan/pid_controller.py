
import collections
import logging
import os
import numpy as np

import precog.interface
import precog.utils.tfutil as tfutil
import precog.utils.class_util as classu
import precog.utils.np_util as npu
import precog.utils.rand_util as randu
import precog.utils.tensor_util as tensoru
import carla.carla_server_pb2 as carla_protocol

log = logging.getLogger(os.path.basename(__file__))

class CarPIDControllers:
    @classu.member_initialize
    def __init__(self, throttle_controller, steer_controller, brake_controller=None, horizon_seconds=1.):
        pass
    
    def reset(self):
        self.throttle_controller.reset()
        self.steer_controller.reset()
        if self.brake_controller is not None:
            self.brake_controller.reset()
            
    @staticmethod
    def frompidconf(pidconf):
        # Create and bundle the PID controllers.
        throttle_controller = PIDThrottleController(
            kp=pidconf.kp_throttle,
            ki=pidconf.ki_throttle,
            kd=pidconf.kd_throttle,
            form=pidconf.form_throttle,
            control_bias=pidconf.throttle_bias)
        steer_controller = PIDSteerController(
            kp=pidconf.kp_steer,
            ki=pidconf.ki_steer,
            kd=pidconf.kd_steer,
            form=pidconf.form_steer,
            control_bias=pidconf.steer_bias)
        brake_controller = PIDBrakeController(kp=pidconf.kp_brake,
                                              ki=pidconf.ki_brake,
                                              kd=pidconf.kd_brake,
                                              form=pidconf.form_brake,
                                              control_bias=pidconf.brake_bias)
        return CarPIDControllers(
            throttle_controller, steer_controller, brake_controller=brake_controller)

    def update(self, measurement, current_angle, target_angle, position_setpoint):
        forward_speed = measurement.player_measurements.forward_speed
        
        # Steer controller is controlling to the target angle in the plan.
        steer_unsnapped = self.steer_controller.update(
            target_angle=target_angle, current_angle=current_angle)

        heading_2d = [1., 0.]
        # Compute distance of the setpoint along the heading.
        horizon_meters = compute_setpoint_signed_distance(setpoint=position_setpoint, heading_2d=heading_2d)
        # Target seconds to match
        target_forward_speed = horizon_meters / self.horizon_seconds
        
        # Update the PID controllers.
        # Throttle controller is controlling to a position in local coordinates.
        throttle_unsnapped = self.throttle_controller.update(
            setpoint=target_forward_speed, process_variable=forward_speed)
        log.debug("\n\tSigned distance {:.3f}".format(horizon_meters) +
                  "\n\tForward speed {:.3f}".format(forward_speed) +
                  "\n\tTarget forward speed {:.3f}".format(target_forward_speed) +
                  "\n\tThrottle unsnapped: {:.3f}".format(throttle_unsnapped))
        control = carla_protocol.Control()
        control.steer = clip_steer(steer_unsnapped)
        control.throttle = clip_throttle(throttle_unsnapped)
        # if self.brake_controller is not None:
        #     brake = self.brake_controller.update(
        #     control.brake = max(min(brake, 1.0), 0.0)
        control.brake = 0.0
        control.hand_brake = False
        control.reverse = False
        if throttle_unsnapped < -5:
            log.warning("Hacking very negative throttle to braking")
            control.brake = 1.0
        return control

def compute_setpoint_signed_distance(setpoint, heading_2d):
    heading_2d_norm = heading_2d / np.linalg.norm(heading_2d)
    return setpoint[0] * heading_2d_norm[0] + setpoint[1] * heading_2d_norm[1]

def clip_throttle(t):
    return min(max(t, 0.), 1.)

def clip_steer(s):
    return min(max(s, -1.), 1.)
    

class PIDController:
    @classu.member_initialize
    def __init__(self, kp, ki=0.0, kd=0.0, control_bias=0.0, process_variable=0.0, history_length=20,
                 form='standard'):
        """

        :param kp: 
        :param control_bias: 
        :param process_variable: the bias of the quantity we're controlling.
        :param history_length: how many error terms to use.
        :returns: 
        :rtype: 

        """
        log.info("Instantiating {}(kp={}, ki={}, kd={}, bias={}, center={})".format(
            self.__class__.__name__,
            kp,
            ki,
            kd,
            control_bias,
            process_variable))
        assert(form in ('standard', 'full'))
        self.reset()

    def reset(self):
        self.errors = collections.deque(maxlen=self.history_length)

    def update(self, setpoint, process_variable=None):
        """Update the controller with the new setpoint.

        :param setpoint: quantity we want to control to
        :returns: 
        :rtype: 

        """
        log.info("New setpoint: {:.3f}".format(setpoint))
        if process_variable is None:
            process_variable = self.process_variable
        error = setpoint - process_variable
        self.errors.append(error)

        # Compute nonproportional terms for controller in standard form.            
        if self.form == 'standard':
            Ti = len(self.errors)
            Td = 1
            ki = self.kp / Ti
            kd = self.kp * Td
        # Compute nonproportional terms for controller via provided parameters.
        else:
            ki = self.ki 
            kd = self.kd

        proportional = self.kp * error
        # Approximate integral with sum.
        integral = ki * sum(self.errors)
        # Compute derivate via backward difference.
        derivative = kd * np.diff(np.asarray(self.errors)[-2:]).sum()
        # Compute full control.
        pid = proportional + integral + derivative
        pid_with_bias = pid + self.control_bias
        log.debug("PID control summary:\n" +
                  "\tpid with bias={:.3f}\n".format(pid_with_bias) +
                  "\tpid={:.3f}\n".format(pid) +
                  "\tproportional={:.3f}\n".format(proportional) +
                  "\tintegral={:.3f}\n".format(integral) +
                  "\tderivative={:.3f}".format(derivative))
        return pid_with_bias

class PIDThrottleController(PIDController): pass

class PIDSteerController(PIDController):
    def update(self, target_angle, current_angle):
        """Control based on the angle to the setpoint."""
        # The target angle corresponds to the velocity vector at t.
        return super().update(setpoint=target_angle, process_variable=current_angle)

class PIDBrakeController(PIDController):
    pass

def pilot_with_PID_steer_controller(waypoints_local_2d, coeff_steer):
    angle = np.arctan2(waypoints_local_2d[0,1], waypoints_local_2d[0,0])
    steer = min(max(coeff_steer * angle, -1.), 1.)
    throttle = 0.5
    brake = 0.  # not considered yet
    control = carla_protocol.Control()
    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = False
    control.reverse = False
    return control


def plan_p_steer(setpoint, coeff_steer):
    """

    :param setpoint: 
    :param coeff_steer: return the steering angle
    :returns: 
    :rtype: 

    """
    assert(setpoint.ndim == 1)
    angle = coeff_steer * np.arctan2(setpoint[1], setpoint[0])
    return min(max(coeff_steer * angle, -1.), 1.)

