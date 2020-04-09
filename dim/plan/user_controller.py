
import dill
import numpy as np
import logging
import pygame.locals
import pygame

import os

# from pynput import keyboard
# import keyboard


import carla.carla_server_pb2 as carla_protocol

log = logging.getLogger(os.path.basename(__file__))

class UserController:
    def __init__(self):
        pygame.init()
        size = (800, 400)
        window_surface = pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        window_surface.fill((0,0,0))
        self.all_controls = []
        self.control_trajectory = []
        self.reset()
        
    def load_controls(self, filename):
        self.all_controls = dill.load(open(filename, 'rb'))

    def reset(self, add_new=True):
        self.steer = 0
        self.throttle = 0
        self.brake = 0
        self.reverse = False
        self.t = 0
        if len(self.control_trajectory) > 0 and add_new:
            self.all_controls.append(self.control_trajectory)
        self.control_trajectory = []

    def replay_control(self):
        if self.t == 0:
            self.chosen_control_trajectory = self.all_controls[np.random.choice(len(self.all_controls))]
            log.info("Chose new control trajectory")
        control = carla_protocol.Control()
        control.hand_brake = False
        control.reverse = False
        control.steer = self.chosen_control_trajectory[self.t][0]
        control.throttle = self.chosen_control_trajectory[self.t][1]
        self.t += 1
        return control
    
    def user_control(self):
        restart = False

        control = carla_protocol.Control()
        control.hand_brake = False

        takeover = 37
        if self.t < takeover:
            self.steer = 0
            self.throttle = 1.
            self.brake = 0.0
            self.reverse = False

            for event in pygame.event.get():
                pass
        if self.t == takeover:
            self.steer = 0.
            self.throttle = 0.
            self.brake = 0.0
        if self.t >= takeover:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.steer = -1.0
                    if event.key == pygame.K_d:
                        self.steer = 1.0
                    if event.key == pygame.K_w:
                        self.throttle = 1.0
                    if event.key == pygame.K_s:
                        self.brake = 1.0
                    if event.key == pygame.K_r:
                        restart = True
                    if event.key == pygame.K_e:
                        self.reverse = not self.reverse
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        self.steer = 0
                    if event.key == pygame.K_d:
                        self.steer = 0
                    if event.key == pygame.K_w:
                        self.throttle = 0
                    if event.key == pygame.K_s:
                        self.brake = 0
                    if event.key == pygame.K_r:
                        restart = True

        log.info("t: {}".format(self.t))

        control.steer = self.steer
        control.throttle = self.throttle
        control.brake = self.brake
        control.reverse = self.reverse
        self.control_trajectory.append((self.steer, self.throttle))
        self.t += 1
        return control, restart

def init():
    # pygame.init()
    # size = (800, 400)
    # pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)
    pass
