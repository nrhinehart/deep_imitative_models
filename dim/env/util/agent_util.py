
import collections
import copy
import logging
import functools
import numpy as np
import pdb

import precog.utils.class_util as classu
import precog.utils.np_util as npu

import dim.env.preprocess.carla_preprocess as preproc

log = logging.getLogger(__file__)

class EpisodeState:
    INPROGRESS = 0
    SUCCESS = 1
    FAILURE = 2

class SensorParams:
    @classu.member_initialize
    def __init__(self, settings, center_save_blacklist):
        """Hold info about which sensors we have and will save"""
        # Retrieve all of the active sensor names.
        self.sensor_names = [_.SensorName for _ in settings._sensors]

        # In general, don't need any sensors for timesteps that we won't be predicting at.
        self.surround_save_blacklist = copy.deepcopy(self.sensor_names)
        
        self.lidar_sensor = settings._sensors[self.sensor_names.index('Lidar32')]

class EpisodeParams:
    @classu.member_initialize
    def __init__(self, episode, frames_per_episode, root_dir, settings, snapshot_frequency=200):
        """Hold info about episode related things"""

class StationaryTracker:
    def __init__(self, size, thresh=0.001, track_others=True):
        size = int(round(size))
        self.size = size
        self.thresh = thresh
        self.track_others = track_others
        self.reset()

    def reset(self):
        self.position_history = collections.defaultdict(functools.partial(collections.deque, maxlen=self.size))
        self.orientation_history = collections.defaultdict(functools.partial(collections.deque, maxlen=self.size))
        self.n_anyone_stationary = 0
        self.last_measurement = None

    def is_anyone_stationary(self):
        max_dist = -np.inf
        for window in self.position_history.values():
            ws, max_dist = self.is_window_stationary(window)
            if ws:
                self.n_anyone_stationary += 1
                return True, max_dist
        self.n_anyone_stationary = 0
        return False, max_dist

    def is_window_stationary(self, window):
        if len(window) < self.size: return False, -np.inf
        start = window[-self.size]
        middle = window[-self.size//2]
        end = window[-1]
        # print("SME: {} {} {}".format(start, middle, end))
        sm, dist0 = self.is_stationary(start, middle)
        me, dist1 = self.is_stationary(middle, end)
        max_dist = max(dist0, dist1)
        return sm and me, max_dist

    def is_stationary(self, position_old, position_new):
        dist = np.linalg.norm(position_old - position_new)
        return dist < self.thresh, dist

    def index_new_measurement(self, measurement):
        self.last_measurement = measurement
        player_pos = preproc.vector3_to_np(measurement.player_measurements.transform.location)
        player_ori = preproc.vector3_to_np(measurement.player_measurements.transform.orientation)
        self.position_history['player'].append(player_pos)
        self.orientation_history['player'].append(player_ori)
        if self.track_others:
            for agent in measurement.non_player_agents:
                if agent.HasField('vehicle'):
                    pos = preproc.vector3_to_np(agent.vehicle.transform.location)
                    self.position_history[agent.id].append(pos)

class NonredStuckTracker(StationaryTracker):
    @classu.member_initialize
    def __init__(self, waypointer, size, thresh=0.001):
        super().__init__(size, thresh, track_others=False)

    def is_stuck_far_from_red_lights(self, thresh_meters=25., ignore_reds=False):
        """

        :param thresh_meters: 
        :param ignore_reds: TODO HACK
        :returns: 
        :rtype: 

        """
        is_stationary, max_dist = self.is_anyone_stationary()
        light_state, _ = self.waypointer.get_upcoming_traffic_light(self.last_measurement, None, thresh_meters=thresh_meters)
        is_red = light_state == 'RED'
        stuck = (is_stationary and ignore_reds) or (is_stationary and not is_red)
        log.debug("NonredStuckTracker Stationary" +
                 "={} (thresh={:.3f}, max_dist={:.3f}, size={}). Far light red={}. Ignoring reds={}. Light_state={}. Stuck={}.".format(
                     is_stationary, self.thresh, max_dist, self.size, is_red, ignore_reds, light_state, stuck))
        return is_stationary, stuck

class TurnTracker(StationaryTracker):
    @classu.member_initialize
    def __init__(self, size, thresh):
        """

        :param size: 
        :param thresh: minimum angular difference in degrees
        :returns: 
        :rtype: 

        """
        
        super().__init__(size, thresh, track_others=False)

    def is_turning(self):
        try:
            positions = np.asarray(self.position_history['player'])
        except KeyError:
            return False
        if positions.shape[0] < self.size:
            return False
        headings = np.asarray(self.orientation_history['player'])
        dists = np.linalg.norm(positions - positions[0], axis=-1)
        
        # # Headings from each position to the next position along a sequence
        # dists = np.linalg.norm(headings, axis=-1)
        # # Normalized
        # headings = (headings.T / np.linalg.norm(headings, axis=-1)).T
        heading0 = headings[0]
        dps = np.einsum('j,ij->i', heading0, headings)
        mags = np.linalg.norm(headings, axis=-1)
        # Get the angles between the first heading and subsequent headings.
        cosa = dps / (mags[0] * mags)
        angles = np.arccos(cosa)
        angles[np.isnan(angles)] = 0.0
        angles_deg = 180/np.pi * angles

        # This helps to filter heading errors when car is nearly stationary.
        is_far = dists > 5.
        log.debug("turn angles: {}".format(angles_deg))
        log.debug("turn dists: {}".format(dists))
        
        # is_increasing = (np.argsort(angles_deg) == np.arange(angles_deg.size)).all()
        # is_decreasing = (np.argsort(angles_deg)[::-1] == np.arange(angles_deg.size)).all()
        # If the angles increase and the final heading is sufficiently rotated from the original, it's a left turn.
        turning_left = np.logical_and(angles_deg >= self.thresh, is_far).any()
        # If the angles decrease and the final heading is sufficiently rotated from the original, it's a right turn.
        turning_right = np.logical_and(angles_deg <= -self.thresh, is_far).any()
        log.debug("Turning left={}. Turning right={}".format(turning_left, turning_right))
        turning = turning_right or turning_left
        return turning

def discretize_throttle(throttle):
    # assert(0. <= throttle <= 1.0)
    if throttle < 0.25:
        # return 0.
        return 0.
    elif throttle < 0.75:
        return 0.5
    else:
        return 1.0

def decide_episode_state(waypointer, waypoints, stop_tracker, start_obs, current_obs, goal_distance_threshold):
    # In progress, unset goal.
    if waypointer.goal_position is None:
        summary = "No goal set yet."
        log.info(summary)
        return EpisodeState.INPROGRESS, summary
    
    current_location = current_obs.player_positions_world[-1]
    distance_to_goal, distance_to_goal_snapped = waypointer.get_distance_to_goal(current_location)
    smaller_dist = min(distance_to_goal, distance_to_goal_snapped)
    log.info("Current real, snapped distances to goal: {:.2f}, {:.2f}. Location: {}. Goal location: {}".format(
        distance_to_goal, distance_to_goal_snapped, current_location, waypointer.goal_position))

    # Success.
    if smaller_dist <= goal_distance_threshold:
        summary = "Within goal threshold! Declaring success."
        log.info(summary)
        return EpisodeState.SUCCESS, summary
        
    is_stationary, max_dist = stop_tracker.is_anyone_stationary()
    log.info("Stationary? {}".format(is_stationary))

    # Fail.
    if is_stationary:
        summary = "Agent was stationary for too long ({} frames)! Episode failed.".format(stop_tracker.size)
        log.info(summary)
        return EpisodeState.FAILURE, summary

    # In progress, set goal.
    return EpisodeState.INPROGRESS, "In progress"

def lock_observations(sensor_data):
    # Prevent data mutation.
    npu.lock_nd(sensor_data.Lidar32.point_cloud._array)
    npu.lock_nd(sensor_data.CameraRGB.data)
    npu.lock_nd(sensor_data.CameraRGBBEV.data)
    npu.lock_nd(sensor_data.CameraDepth.data)
    npu.lock_nd(sensor_data.CameraSemantic.data)

def build_sensor_blacklist(args):
    # Decide which sensors to save for the 'centers', which each corresponds to an example.
    center_save_blacklist = ['CameraRGBBEV']
    if not args.save_lidar: center_save_blacklist.append('Lidar32')
    if not args.save_semantic: center_save_blacklist.append('CameraSemantic')
    else:
        if not args.save_depth:
            log.info("HACKING THE DEPTH SAVING ON, TO CREATE SEMANTIC FEATURES")
            args.save_depth = True
    if not args.save_frontal_rgb: center_save_blacklist.append('CameraRGB')
    if not args.save_depth: center_save_blacklist.append('CameraDepth')
    return center_save_blacklist
