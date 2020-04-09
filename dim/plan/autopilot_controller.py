
import logging
import numpy as np
import os
import random

import carla.carla_server_pb2 as carla_protocol

log = logging.getLogger(os.path.basename(__file__))

def noisy_autopilot(measurement, replan_index, replan_period, cfg):
                        
    # Together with the measurements, the server has sent the
    # control that the in-game autopilot would do this frame. We
    # can enable autopilot by sending back this control to the
    # server. We can modify it if wanted, here for instance we
    # will add some noise to the steer.
    autocontrol = measurement.player_measurements.autopilot_control
    control = carla_protocol.Control()
    brake = autocontrol.brake
    hand_brake = autocontrol.hand_brake
    steer = autocontrol.steer
    throttle = autocontrol.throttle
    reverse = autocontrol.reverse

    # Autopilot perturbation.
    if cfg.add_steering_noise:
        # Sometimes add noise.
        if random.random() > 0.5:
            steer_noise = 0.1
            snoise = random.uniform(-steer_noise, steer_noise)
            steer += snoise
            log.debug("snoise {}. steer".format(snoise, steer))
        # Clip before CARLA internally clips them.
        steer = max(min(steer, 1.), -1.)
    if cfg.add_throttle_noise:
        if random.random() > 0.5:
            if brake > 0.2:
                throttle_noise = 0.0
            else:
                throttle_noise = 0.1
                brake = 0.0
            tnoise = random.uniform(-throttle_noise, throttle_noise)
            throttle += tnoise
            log.debug("tnoise {}. throttle: {}".format(tnoise, throttle))

    throttle = max(min(throttle, 1.0), 0.0)

    control.throttle = throttle
    control.brake = brake
    control.steer = steer
    control.hand_brake = hand_brake
    control.reverse = reverse
    return control
