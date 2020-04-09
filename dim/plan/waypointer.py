#!/usr/bin/env python3

"""Generates waypoints to follow."""

import logging
import numpy as np
import os
import pdb
import scipy.spatial

import precog.utils.class_util as classu
import precog.utils.np_util as npu
import dim.env.util.geom_util as geom_util
import dim.plan.traffic_light_decider as traffic_light_decider

from carla.planner.planner import Planner
from carla.planner.graph import sldist

log = logging.getLogger(os.path.basename(__file__))

class WaypointerError(Exception): pass

class Waypointer(traffic_light_decider.TrafficLightDecider):

    # Orientations ("oris") w.r.t. the birds-eye-view of carla plots
    NORTH = (0., -1.)
    SOUTH = (0., 1.)
    WEST = (-1., 0.)
    EAST = (1., 0.)

    # LANE_WIDTH_PIXELS = 25  # measured
    LANE_WIDTH_PIXELS = 24.515  # measured
    # CORNER_SHIFT_PIXELS = 8  # measured
    CORNER_SHIFT_PIXELS = 6  # measured

    @classu.member_initialize
    def __init__(self, waypointerconf, planner_or_map_name):

        self._planner = Planner(planner_or_map_name) if isinstance(planner_or_map_name, str) else planner_or_map_name
        self._params = waypointerconf

        # internal carla objects for planning
        self._city_track = self._planner._city_track

        # pdb.set_trace()
        # print(self.LANE_WIDTH_PIXELS * self._city_track.get_pixel_density())
        
        self._map = self._city_track._map
        self._converter = self._map._converter
        self._graph = self._map._graph
        self._grid = self._map._grid
        self._grid_structure = self._grid._structure    # shape [49, 41], of zeros and ones (0==road, 1==not_road)

        # goal and waypoint variables
        self._road_nodes = self.get_road_nodes()
        self.reset_route()

        # adjustment information for extra curvy map corners
        res = self._graph._resolution
        self._corner_nodes = [(0, 0), (res[0]-1, 0), (0, res[1]-1), (res[0]-1, res[1]-1)]
        corner_nodes_adjacent = [(1, 1), (res[0]-2, 1), (1, res[1]-2), (res[0]-2, res[1]-2)]
        self._corner_shift_ori = np.array(corner_nodes_adjacent) - np.array(self._corner_nodes)

        # traffic light cached data
        self.traffic_light_inodes_and_oris = None
        self.last_traffic_light_state = None
        self.intersection_count = 0
        self.red_light_violations = 0

    def reset_route(self):
        self._current_route = None  # route as a list of world coordinates, one per route_nodes (shifted onto lane etc)
        self.route_nodes = None  # waypoints_plot.py uses this
        self.route_oris = None   # waypoints_plot.py uses this
        self.goal_node = None
        self.goal_position = None

    def reset_plan_to_goal(self):
        # Just reset the thing that will force us to replan.
        self._current_route = None
        self.reset_route()
        
    def convert(self, position, output_type):
        """
        :param position: position of any type PIXEL, WORLD, NODE
        :param output_type: type to convert to, use string either 'pixel', 'world', 'node'
        :return: position of desired type
        """
        if position is None or self.get_position_type(position) == output_type:
            return position
        else:
            converter = getattr(self._converter, 'convert_to_' + output_type)
            return converter(position)
        
    def convert_to_ori(self, ori):
        """
        :param ori: a tuple of two floats, or the string "NORTH", "SOUTH", "WEST", "EAST"
        :return: a tuple of two floats that defines a 2d direction vector
        """
        return getattr(self, ori) if isinstance(ori, str) else ori

    def get_position_type(self, position):
        """
        :param type: either 'pixel', 'world', 'node'
        """
        # PIXEL = 0  # hack: copying form carla converter
        # WORLD = 1
        # NODE = 2
        types = ('pixel', 'world', 'node')
        type_index = self._converter._check_input_type(position)
        return types[type_index]

    def get_road_nodes(self):
        """
        :return: returns a NODE list where roads are (a node is a tuple of two ints)
        """
        out = zip(*np.where(self._grid_structure == 0))
        out = [(int(x[0]), int(x[1])) for x in out]  # numpy.int64 --> int
        return out

    def get_intersection_nodes(self):
        """
        :return: returns a NODE list where road intersections are (a node is a tuple of two ints)
        """
        return list(self._graph._nodes)

    def get_route_nodes_at_intersections(self, route_nodes):
        return set(route_nodes).intersection(self.get_intersection_nodes())

    def get_nonintersection_route_nodes(self, route_nodes):
        inters = self.get_intersection_nodes()
        return [_ for _ in route_nodes if _ not in inters]

    def get_waypoints_from_measurement(self,
                                       measurement,
                                       goal_position=None,
                                       goal_ori=None, **kwargs):
        """
        Generates a set of waypoints for the robot to follow towards a goal_position. If goal_position is not set, the
        road node furthest from the car will be chosen.
        :param measurement: what the carla client reads from client.read_data()
        :param goal_position: goal robot position, either type PIXEL, WORLD, NODE (see carla.planner.converter)
        :param goal_ori: goal robot orientation, either 2d-float-tuple or "NORTH", "SOUTH", "WEST", "EAST"
        :param include_current_position: include a waypoint at the car's current position
        :return: a set of WORLD waypoints (see carla.planner.converter for WORLD type)
        """
        player_world = self._get_world(measurement.player_measurements)
        player_ori = self._get_ori(measurement.player_measurements)
        return self.get_waypoints(current_position=player_world,
                                  current_ori=player_ori,
                                  goal_position=goal_position,
                                  goal_ori=goal_ori,
                                  **kwargs)

    def _get_world(self, measurement):
        """
        :param measurement: what the carla client reads from client.read_data()
        :return: the player's WORLD position (see carla.planner.converter for WORLD type)
        """
        x = measurement.transform.location.x
        y = measurement.transform.location.y
        z = measurement.transform.location.z
        world = [x, y, z]
        return world

    def _get_ori(self, measurement):
        """
        :param measurement: what the carla client reads from client.read_data()
        :return: the player's ORI (orienation), a 2d-float-tuple
        """
        orix = measurement.transform.orientation.x
        oriy = measurement.transform.orientation.y
        ori = [orix, oriy]
        return ori

    def get_upcoming_traffic_light(self, measurement, sensor_data, thresh_meters=15.):
        """

        Algorithm:
        1) snap car location to closest road node
        2) snap car orientation to closest NORTH/SOUTH/EAST/WEST discretization
        3) snap all traffic lights to the closest intersection-road nodes
        4) look ahead from car's node position until it intersects with a snapped-traffic light. There are three traffic lights there, so use the car's + traffic lights orientation to determine which traffic light (of 3) is the relevent one at that intersection
        5) if look ahead has not traffic lights ahead (e.g. corners of map), then return `None`
        6) if in the middle of intersection (determined by pixel value of `CARLA_0.8.4/PythonClient/carla/planner/TownXXLanes.png`) then return `None` traffic light data (but with a string that specifics because it's `"INTERSECTION"`.

        :param measurement: what the carla client reads from client.read_data()
        :param sensor_data: IGNORED
        :param thresh_meters: 
        :return: traffic_light_state (str): either "GREEN", "YELLOW", "RED", "INTERSECTION", or "NONE".
                 traffic_light_data: carla traffic_light object (None if no traffic light ahead); and
                    .id: (int)
                    .traffic_light
                        .transform
                            .location
                                .x: (float)
                                .y: (float)
                                .z: (float)
                            .orientation
                                .x: (float) in [-1,1]
                                .y: (float) in [-1,1]
                            .rotation
                                .yaw: (float) in degrees
                        state: (int) 0 mean GREEN, 1 means YELLOW, 2 means RED

        """
        player_world = self._get_world(measurement.player_measurements)
        in_intersection = self.in_intersetion(player_world)
        if in_intersection:
            if self.last_traffic_light_state == 'RED':
                # If the traffic light was RED before, and we entered the intersection, then we entered during a red light!
                self.red_light_violations += 1
            if self.last_traffic_light_state != 'INTERSECTION':
                # We switched from non-intersection to INTERSECTION. Every time this happens, count it as a new intersection.
                self.intersection_count += 1
            self.last_traffic_light_state = "INTERSECTION"
            return self.last_traffic_light_state, None

        round_ori = lambda ori: tuple(map(round, ori))  # because we'll be doing equality comparisons
        rotate180 = lambda ori: (-ori[0], -ori[1])
        node_on_map = lambda node: 0 <= node[0] < self._graph._resolution[0] and 0 <= node[1] < self._graph._resolution[1]
        next_node = lambda node, ori: (node[0] + ori[0], node[1] + ori[1])

        # where are the traffic lights (to the closest intersection node)
        traffic_lights = [_ for _ in measurement.non_player_agents if _.HasField('traffic_light')]
        if self.traffic_light_inodes_and_oris is None:  # then compute it (we only need to compute it once)
            tl_nodes = [self.convert(self._get_world(tl.traffic_light), 'node') for tl in traffic_lights]
            correct_traffic_light_ori = lambda x: (-x[1], x[0])  # to point in direction the light faces
            tl_oris_round = [correct_traffic_light_ori(round_ori(self._get_ori(tl.traffic_light))) for tl in traffic_lights]

            # not all traffic lights are at roads or intersection nodes, let's snap to closest intersection node
            intersection_nodes = self.get_intersection_nodes()  # "inode" for short
            closest_intersection_node = lambda node: min(intersection_nodes, key=lambda inode: sldist(inode, node))
            tl_inodes = list(map(closest_intersection_node, tl_nodes))
            self.traffic_light_inodes_and_oris = list(map(lambda x, y: x + y, tl_inodes, tl_oris_round))  # 36, 4

        try:
            player_road_node = self.get_closest_road_node(player_world)
        except IndexError:
            self.last_traffic_light_state = 'NONE'
            return self.last_traffic_light_state, None
        
        player_ori_round = round_ori(self._get_ori(measurement.player_measurements))
        target_traffic_light_ori_round = rotate180(player_ori_round)

        vector3_to_np = lambda v: np.array([v.x, v.y, v.z])

        # now look for closest traffic light in this direction
        node_ahead = player_road_node

        # tform_now = transform.Transform(measurement.player_measurements.transform).inverse()
        
        while node_on_map(node_ahead):
            key = node_ahead + target_traffic_light_ori_round  # concat for tuple of 4 elements
            if key in self.traffic_light_inodes_and_oris:
                # found this traffic light ahead!
                i = self.traffic_light_inodes_and_oris.index(key)
                traffic_light_data = traffic_lights[i]

                data = vector3_to_np(traffic_light_data.traffic_light.transform.location)
                # tform_now.transform_points(data)
                    
                distance_to_player = np.linalg.norm(player_world - data)
                log.debug("Distance of light to player: {:.1f}".format(distance_to_player))
                if distance_to_player > thresh_meters:
                    # Light is too far away!
                    self.last_traffic_light_state = "NONE"
                    log.debug("Light is too far, does not apply")
                    return self.last_traffic_light_state, None
                else:
                    traffic_light_state_int = traffic_light_data.traffic_light.state  # (int) either: 0, 1, 2
                    traffic_light_state_str = ["GREEN", "YELLOW", "RED"][traffic_light_state_int]
                    self.last_traffic_light_state = traffic_light_state_str
                    log.debug("Light is close enough, applies. State={}. Str='{}'".format(
                        traffic_light_data.traffic_light.state, self.last_traffic_light_state))
                    return self.last_traffic_light_state, traffic_light_data
            node_ahead = next_node(node_ahead, player_ori_round)
        else:
            self.last_traffic_light_state = "NONE"
            return self.last_traffic_light_state, None

    def in_intersetion(self, world):
        pixel = self.convert(world, 'pixel')
        pixel_color = self._map.map_image_lanes[pixel[1], pixel[0]]
        is_white = lambda color: all(color == [255, 255, 255, 255])
        in_intersection = is_white(pixel_color)  # see CARLA_0.8.4/PythonClient/carla/planner/Town01Lanes.png
        return in_intersection

    def get_closest_road_node(self, position):
        """
        :param position: position, either type PIXEL, WORLD, NODE (see carla.planner.converter)
        """
        node = self.convert(position, 'node')
        try:
            closest_road_node = self._map.search_on_grid(node)
        except IndexError:
            log.error("Caught index error when looking for node {}".format(node))
            raise WaypointerError("Couldn't find road node")
        return closest_road_node

    def get_waypoints_from_transform(self, transform, goal_position=None, goal_ori=None):
        """

        :param transform: carla.transform.Transform object
        :param goal_position: if None, will choose a new goal
        :param goal_ori: if None, will choose a new goal
        :returns: 
        :rtype: 

        """
        world_xyz = transform.matrix[:3, -1].copy().tolist()
        yaw = transform.yaw  * np.pi / 180.
        # TODO verify.
        world_ori = [round(np.cos(yaw), 0), round(np.sin(yaw), 0)]
        return self.get_waypoints(current_position=world_xyz, current_ori=world_ori, goal_position=goal_position, goal_ori=goal_ori)

    def get_waypoints(self,
                      current_position,
                      current_ori,
                      goal_position=None,
                      goal_ori=None):
        """
        Generates a set of waypoints for the robot to follow towards a goal_position. If goal position is not set, the
        road node furthest from the car will be chosen. If the goal orientation is not set, the orientation of
        shortest distance will be chosen.
        :param current_position: current robot position, either type PIXEL, WORLD, NODE (see carla.planner.converter)
        :param current_ori: current robot orientation, either 2d-float-tuple or "NORTH", "SOUTH", "WEST", "EAST"
        :param goal_position: goal robot position, either type PIXEL, WORLD, NODE (see carla.planner.converter)
        :param goal_ori: goal robot orientation, either 2d-float-tuple or "NORTH", "SOUTH", "WEST", "EAST"
        :return: a set of WORLD waypoints (see carla.planner.converter for WORLD type)
        """
        current_world = self.convert(current_position, 'world')
        current_node = self.convert(current_position, 'node')

        # Project onto closest road node.
        current_node = self.get_closest_road_node(current_node)
        current_ori = self.convert_to_ori(current_ori)
        goal_node = self.convert(goal_position, 'node')
        goal_ori = self.convert_to_ori(goal_ori)

        print(goal_node, goal_ori)
        
        # No code below should mistakenly use these, since their types are arbitrary.
        del current_position  
        del goal_position

        route = self._get_route(current_node, current_ori, goal_node, goal_ori)
            
        # Distance to each route position.
        dists_to_route = np.linalg.norm(np.array(route) - np.array(current_world), axis=1)
        assert(len(dists_to_route) == len(route))

        # Ahead or behind
        closest_route_index = max(np.argmin(dists_to_route) - 1 - 2 * self._params.extra_route_node, 0)
            
        log.debug("Closest route index: {}".format(closest_route_index))
        remaining_route = route[closest_route_index:]
        
        # No code below should mistakenly use this.
        del route  

        current_node_as_world = self.convert(current_node, 'world')
        end_route = remaining_route[-1]
        log.debug("Current snapped pos: {}, Goal snapped pos: {}".format(np.asarray(current_node_as_world), np.asarray(end_route)))
        dist = np.linalg.norm(np.asarray(current_node_as_world) - np.asarray(end_route))
        udist = np.linalg.norm(np.asarray(current_world) - np.asarray(end_route))

        self.goal_node = end_route
        self.goal_position = self.convert(self.goal_node, 'world')
        
        log.debug("Distances to goal. Snapped: {:.3f}. Unsnapped: {:.3f}".format(dist, udist))
        if dist < self._params.control_meters_per_waypoint:
            log.info("Likely near goal. Distance: {:.3f}".format(dist))
        log.debug("In get_waypoints. Input position {}. Goal target {}".format(current_world, end_route))

        if self._params.interpolate:
            log.debug("Interpolating waypoints")
            x, y, z = [], [], []
            for wp0, wp1 in zip(remaining_route[:-1], remaining_route[1:]):
                x.extend(np.linspace(wp0[0], wp1[0], self._params.interp_points_per_pair, endpoint=False))
                y.extend(np.linspace(wp0[1], wp1[1], self._params.interp_points_per_pair, endpoint=False))
                z.extend(np.linspace(wp0[2], wp1[2], self._params.interp_points_per_pair, endpoint=False))
            waypoints = np.stack((x, y, z), axis=-1)
        else:
            log.debug("not interpolating waypoints")
            # waypoints = remaining_route[1:]  # 0 could be behind the car
            # We should allow for points behind the car?
            waypoints = remaining_route

        return waypoints

    def _get_route(self,
                   current_node,
                   current_ori,
                   goal_node,
                   goal_ori,
                   shift_intersection_nodes=False):
        """
        Plans a route if goal_node is not None, otherwise will use previous route, otherwise will plan a new route based
        on a goal far away.
        :param current_node:(mandatory: cannot be None)
        :param current_ori:(mandatory: cannot be None)
        :param goal_node: (optional: can be None)
        :param goal_ori: (optional: can be None)
        :return: waypoints_per_node
        """

        # has user updated their destination?
        if goal_node is not None:
            if self.route_nodes is not None and goal_node == self.route_nodes[-1]:
                log.debug("Received same goal as the current route's, using the current route.")
                return self._current_route
            else:
                log.debug("Using inputted goal to plan a new route.")
                return self._plan_route(current_node, current_ori, goal_node, goal_ori)
        # otherwise stick to current route
        elif self._current_route is not None:
            log.debug("Continuing to use cached route.")
            return self._current_route
        # otherwise auto-generate a route
        else:
            log.debug("Planning a new route to the farthest road node.")
            dist_from_current = lambda node, curr=np.array(current_node): (np.linalg.norm(np.array(node) - curr))
            goal_node = max(self._road_nodes, key=dist_from_current)
            return self._plan_route(current_node, current_ori, goal_node, goal_ori, shift_intersection_nodes=shift_intersection_nodes)

    def get_distance_to_goal(self, position):
        assert(self.goal_position is not None)
        position_world = self.convert(position, 'world')
        position_snapped_world = self.convert(self.get_closest_road_node(position_world), 'world')
        distance_world = np.linalg.norm(self.goal_position - position_world)
        distance_world_snapped = np.linalg.norm(self.goal_position - position_snapped_world)
        return distance_world, distance_world_snapped

    def _plan_route(self,
                    current_node,
                    current_ori,
                    goal_node,
                    goal_ori,
                    shift_intersection_nodes=False):
        """
        :param current_node: (mandatory: cannot be None)
        :param current_ori: (mandatory: cannot be None)
        :param goal_node: (mandatory: cannot be None)
        :param goal_ori: (optional: can be None)
        :return:
        """

        log.info("Planning route: current node: {}, current_ori: {}, Goal node: {}".format(current_node, current_ori, goal_node))
        if goal_node not in self._road_nodes:
            log.error('ERROR[waypointer]: goal_node {} not on road!'.format(goal_node))

        # goal orientation was an optional input
        if goal_ori is None:  # then use orientation of shortest distance
            # NOTE: It seems carla planning does not care about goal orientation even though the API asks for it.
            # So I will leave this (currently redundant) code here in case that changes in the future.
            oris = [self.NORTH, self.SOUTH, self.EAST, self.WEST]
            num_nodes = []
            for ori in oris:
                nodes, _ = self._get_route_nodes_and_oris(current_node, current_ori, goal_node, ori)
                num_nodes.append(len(nodes))
            goal_ori = oris[np.argmin(num_nodes)]

        # first compute the route using carla's framework of discrete nodes
        self.route_nodes, self.route_oris = \
            self._get_route_nodes_and_oris(current_node, current_ori, goal_node, goal_ori)

        # a route is a is one waypoint per route node, shifted to be in the middle of the right lane
        lane_width_meters = self.LANE_WIDTH_PIXELS * self._city_track.get_pixel_density()
        rot90right = lambda x: [-x[1], x[0]]
        route = []  # a route is one world coord per route node (shifted into correct lane etc)
        log.debug("Waypointer fudge factor: '{}'".format(self.waypointerconf.lane_fudge_factor))
        inters = self.get_intersection_nodes()
        
        for idx, (node, ori) in enumerate(zip(self.route_nodes, self.route_oris)):
            # If we shouldn't shift turning paths in intersections:
            dont_shift_this_node = not shift_intersection_nodes and node in inters
            if dont_shift_this_node:
                # If we have nodes before and after with which to compute if the path is turning:
                interior = idx < len(self.route_nodes) - 2 and idx > 0
                if interior:
                    next_node = self.route_nodes[idx + 1]
                    prev_node = self.route_nodes[idx - 1]
                    x_changed = next_node[0] != prev_node[0]
                    y_changed = next_node[1] != prev_node[1]
                    route_is_turning = x_changed and y_changed
                    forward_vec = np.asarray(node) - np.asarray(prev_node)
                    next_vec = np.asarray(next_node) - np.asarray(prev_node)                    
                    # If the path is turning:
                    if route_is_turning:
                        mat = np.eye(2)
                        mat[:,0] = forward_vec
                        mat[:,1] = next_vec
                        right = np.linalg.det(mat) < 0
                        if right:
                            # Nullify the lane shift of left turns. TODO there's some weird coord backwardness going on.
                            ori = [0, 0]
                            
            if dont_shift_this_node and interior and route_is_turning and not right:
                # Skip the nodes in the center of right-turning intersections...
                continue
                            
            world = self._converter.convert_to_world(node)

            # lane shift
            lane_shift_direction = np.array(rot90right(ori))
            # drive in middle of lane
            lane_shift = lane_width_meters / 2.0  
            lane_shift_2d = lane_shift_direction * (lane_shift + self.waypointerconf.lane_fudge_factor)
            # world points are 3D
            world[:2] += lane_shift_2d  

            # map-corner shift
            if node in self._corner_nodes:
                corner_shift_meters = self.CORNER_SHIFT_PIXELS * self._city_track.get_pixel_density()
                corner_index = self._corner_nodes.index(node)
                corner_shift_direction = self._corner_shift_ori[corner_index]
                corner_shift_2d = corner_shift_direction * corner_shift_meters
                world[:2] += corner_shift_2d

            route.append(world)
        route = np.array(route)

        if not self._params.interpolate:
            log.info("Pruning waypointer route")
            route, self.keep_indices = self.prune_route(route, self._params.control_meters_per_waypoint)
        else:
            self.keep_indices = np.arange(route.shape[0])

        self._current_route = route  # for next call
        self._current_route_oris = np.asarray(self.route_oris)[self.keep_indices]  # TODO(rowan) warning: self.route_oris not interpolated
        return self._current_route

    def _get_route_nodes_and_oris(self, current_node, current_ori, goal_node, goal_ori):
        route_nodes = self._city_track.compute_route(current_node, current_ori, goal_node, goal_ori)
        route_oris = [current_ori]
        for i in range(1, len(route_nodes)):
            next_node = route_nodes[i+1] if i < len(route_nodes)-1 else route_nodes[i]
            prev_node = route_nodes[i-1]
            ori = np.array(next_node) - np.array(prev_node)
            ori = tuple(np.sign(ori)+0.)  # make float
            route_oris.append(ori)
        # TODO: am I missing the final ori?
        return route_nodes, route_oris

    def get_perturbed_route_nodes(self, mu_fwd=-15., sigma_fwd=1., mu_right=-1., sigma_right=3.,
                                  twins=False):
        # a route is a is one waypoint per route node, shifted to be in the middle of the right lane
        rot90right = lambda x: [-x[1], x[0]]
        perturbed_route = []
        lane_width_meters = self.LANE_WIDTH_PIXELS * self._city_track.get_pixel_density()
        for node, ori in zip(self._current_route, self._current_route_oris):
            node_perturbed = node.copy()
            right_direction = np.array(rot90right(ori))
            fwd_direction = ori
            fwd_shift_scale = np.random.normal(loc=mu_fwd, scale=sigma_fwd)
            right_shift_scale = np.random.normal(loc=mu_right, scale=sigma_right)
            # Assumes ori is a unit vec.
            fwd_shift = fwd_direction * fwd_shift_scale
            right_shift = right_direction * right_shift_scale
            # Perturb the (x, y) coordinates forward/backward and right/left
            node_perturbed[:2] = node[:2] + fwd_shift + right_shift
            # node_perturbed[:2] = node[:2] + right_shift
            perturbed_route.append(node_perturbed)
            if twins:
                left_direction = -1 * right_direction
                node_perturbed_twin = node_perturbed.copy()
                node_perturbed_twin[:2] += left_direction * lane_width_meters * 2
                perturbed_route.append(node_perturbed_twin)
        return np.stack(perturbed_route, axis=0)

    def prepare_waypoints_for_control(self, waypoints_local, current_obs, dimconf):
        log.info("Preparing waypoints for control.")
        countlog = Countlog()
        countlog(waypoints_local)
        metadata = {'route_2d': waypoints_local}
        waypointerconf = self.waypointerconf

        if waypoints_local is not None:
            waypoints_local_prepruned = waypoints_local.copy()
            log.debug("Prepruning... first waypoint: {}. last_waypoint: {}. N={}".format(
                waypoints_local_prepruned[0],
                waypoints_local_prepruned[-1],
                waypoints_local_prepruned.shape[0]))
        else:
            log.warning("No waypoints were received to prepare! (None)")
            waypoints_local_prepruned = waypoints_local

        countlog(waypoints_local)

        # Possibly ensure we have a stationary waypoint.
        if waypointerconf.ensure_one_stationary or not waypointerconf.have_planned:
            waypoints_local = self.ensure_zero_waypoint_exists(waypoints_local)

        countlog(waypoints_local)        

        # Drop the close waypoints first.
        waypoints_local = self.drop_close_waypoints(waypoints_local, waypointerconf.min_waypoint_distance)

        countlog(waypoints_local)

        # Possibly ensure we still have a stationary waypoint.
        if waypointerconf.ensure_one_stationary or not waypointerconf.have_planned:
            waypoints_local = self.ensure_zero_waypoint_exists(waypoints_local)

        countlog(waypoints_local)

        # Note this is done after ensuring stationary exist.
        if waypointerconf.drop_near_on_green:
            if current_obs.traffic_light_state == 'GREEN' or current_obs.traffic_light_state == 'INTERSECTION':
                log.info("Dropping nearby waypoints (<{:.2f}) due to green light.".format(waypointerconf.min_green_distance))
                waypoints_local = self.drop_close_waypoints(waypoints_local, waypointerconf.min_green_distance)
                waypoints_local = self.drop_behind_waypoints(waypoints_local)
        if waypointerconf.drop_near_on_yellow:
            if current_obs.traffic_light_state == 'YELLOW':
                log.info("Dropping nearby waypoints (<{:.2f}) due to yellow light.".format(waypointerconf.min_green_distance))
                waypoints_local = self.drop_close_waypoints(waypoints_local, waypointerconf.min_green_distance)
                waypoints_local = self.drop_behind_waypoints(waypoints_local)
        if waypointerconf.drop_far_on_red:
            if current_obs.traffic_light_state == 'RED':
                log.info("Dropping far waypoints (>={:.2f}) due to red light.".format(waypointerconf.max_red_distance))
                waypoints_local = self.drop_far_waypoints(waypoints_local, waypointerconf.max_red_distance)

        # Now drop the far waypoints.
        waypoints_local = self.drop_far_waypoints(waypoints_local, waypointerconf.max_waypoint_distance)

        countlog(waypoints_local)                

        if waypointerconf.drop_near_on_turn:
            if current_obs.is_turning:
                min_turn_thresh = self.waypointerconf.turn_waypoint_distance
                log.info("Vehicle is turning. Dropping waypoints closer than {} meters".format(min_turn_thresh))
                waypoints_local = self.drop_close_waypoints(waypoints_local, min_turn_thresh)

        countlog(waypoints_local)

        # Undo everything!
        if waypoints_local.size == 0:
            log.nerror("Pruned all waypoints! Unpruning!"*3)
            waypoints_local = waypoints_local_prepruned

        countlog(waypoints_local)                        

        log.info("First waypoint: {}. last_waypoint: {}".format(
            waypoints_local[0],
            waypoints_local[-1]))

        # Ensure minimum distances achieved. Note that there's no such thing as distance-based 'ordering' without a fixed reference point,
        # so we have to do pairwise computations. (We don't want to add points sequentially based on distance to origin, we want to add them
        #   sequentially based on distance to all other waypoint candidates)
        waypoints_local = self.ensure_waypoints_separated(waypoints_local, meters_per_waypoint=waypointerconf.control_meters_per_waypoint)
        countlog(waypoints_local)

        if waypointerconf.region_control:
            if waypointerconf.drop_far_on_red and current_obs.traffic_light_state == 'RED':
                route_2d_pp = waypoints_local[:, :2]
                polygon = geom_util.generate_stopping_polygon(d=self.waypointerconf.region_d)
                metadata = {'route_2d_pp': route_2d_pp,
                            'route_2d_pp_s': route_2d_pp,
                            'polygon': polygon.copy(),
                            'pp_polygon': polygon.copy()}
                z = waypoints_local[0, -1]
                waypoints_local = np.concatenate((polygon, z * np.ones(shape=(polygon.shape[0], 1))), axis=-1)
            else:
                route_2d = waypoints_local[:, :2]
                # Preprocess.
                route_2d_preprocess = geom_util.preprocess_route_2d_for_polygon(route_2d, clip_K=dimconf.region_clip_K)
                # Prune out duplicates.
                route_2d_preprocess_separate = self.ensure_waypoints_separated(route_2d_preprocess, meters_per_waypoint=1e-3)
                # Create polygons.
                pp_polygon, polygon, *_ = geom_util.create_region_from_route(
                    route_2d_preprocess_separate,
                    clip_K=dimconf.region_clip_K,
                    d=self.waypointerconf.region_d)
                metadata['route_2d_pp'] = route_2d_preprocess
                metadata['route_2d_pp_s'] = route_2d_preprocess_separate
                metadata['polygon'] = polygon
                metadata['pp_polygon'] = polygon
                z = waypoints_local[0, -1]
                waypoints_local = pp_polygon.copy()
                # Add on z=0 coordinates to the polygon so its Nx3
                waypoints_local = np.concatenate((waypoints_local, z * np.ones(shape=(waypoints_local.shape[0], 1))), axis=-1)
        log.info("Finished preparing waypoints")
        return waypoints_local, metadata

    def has_agent_diverged_from_route(self, waypoints_local):
        distances = np.linalg.norm(waypoints_local[:, :2], axis=-1)
        closest_waypoint_dist = distances.min()
        # Test divergence with a small buffer to allow for interpolation errors.
        diverged = closest_waypoint_dist > self.waypointerconf.divergence_distance + 1.
        if diverged:
            log.error("Agent diverged from route! (closest waypoint distance={:.2f} > {:.2f})".format(
                closest_waypoint_dist, self.waypointerconf.divergence_distance))
        return diverged

    def prune_route(self, route, meters_per_waypoint):
        log.info("Enforcing minimum distance between waypoints")
        keep_waypoints = [route[0]]
        keep_indices = [0]
        for idx, waypoint in enumerate(route):
            dist = np.linalg.norm(waypoint - keep_waypoints[-1])
            if dist > meters_per_waypoint:
                keep_waypoints.append(waypoint)
                keep_indices.append(idx)
        return np.stack(keep_waypoints, axis=0), np.asarray(keep_indices)

    def drop_close_waypoints(self, waypoints_local, min_waypoint_distance):
        return waypoints_local[np.where(np.linalg.norm(waypoints_local[..., :2], axis=1) >= min_waypoint_distance)[0]]

    def drop_far_waypoints(self, waypoints_local, max_waypoint_distance):
        return waypoints_local[np.where(np.linalg.norm(waypoints_local[..., :2], axis=1) <= max_waypoint_distance)[0]]

    def drop_behind_waypoints(self, waypoints_local):
        front_mask = waypoints_local[:, 0] >= 0.
        return waypoints_local[front_mask, :]

    def perturb_waypoints(self, waypoints_2d, mu=0., stddev=8.):
        noise = np.random.normal(loc=mu, scale=stddev, size=waypoints_2d.shape)
        waypoints_2d_noisy = waypoints_2d + noise
        return waypoints_2d_noisy

    def ensure_zero_waypoint_exists(self, waypoints_local, threshold_of_zeros=0.2):
        if waypoints_local is None or waypoints_local.size == 0:
            return np.asarray([0., 0., 0.], dtype=np.float32)[None]
        else:
            dists = np.linalg.norm(waypoints_local[:, :2], axis=-1)
            ad = np.argmin(dists)
            if (dists > threshold_of_zeros).all():
                wp = waypoints_local[ad]
                log.info("Adding a zero waypoint, closest is {} with dist {:.2f}".format(wp, dists[ad]))
                zero = np.asarray([0., 0., wp[2]], dtype=np.float32)[None]
                waypoints_local = np.concatenate((zero, waypoints_local))
            return waypoints_local

    def ensure_waypoints_separated(self, waypoints_local, meters_per_waypoint):
        assert(meters_per_waypoint >= 0)
        # Compute all pairwise distances.
        dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(waypoints_local[:, :2]))

        # Track the points to ignore
        toss_inds = set()

        # We'll always keep index 0, since toss_inds is initially empty.
        for i in range(waypoints_local.shape[0]):
            # If we've tossed the index, don't bother checking distances. 
            if i in toss_inds:
                continue
            # We will keep waypoint_i, and toss the indices of points that are too close to it.
            else:
                toss_inds.update(np.where(dists[i] <= meters_per_waypoint)[0])
                # The above step adds i, because it's on the diagonal. Remove it!
                toss_inds.remove(i)

        # All possible inds we could keep
        keep_inds = set(np.arange(waypoints_local.shape[0]))
        # Remove the violators.
        keep_inds = np.asarray(sorted(list(keep_inds - toss_inds)))
        return waypoints_local[keep_inds]

    def get_waypoints_from_transform_and_measurement(self, transform_now, measurement, max_tries=20):
        waypoints_valid = False
        n_tries = 0

        while not waypoints_valid and n_tries < max_tries:
            n_tries += 1
            try:
                # *** Create waypoint route if not created already OR its been reset. ***
                waypoints = npu.lock_nd(self.get_waypoints_from_measurement(measurement))
                log.info("Waypoints received: {}".format(waypoints.shape))
                # Transform waypoints to local coordinates.
                if waypoints.shape[0] == 0:
                    raise RuntimeError("should never happen! whats your max waypoint distance")
                else:
                    waypoints_local = npu.lock_nd(transform_now.transform_points(waypoints))                
                    diverged = self.has_agent_diverged_from_route(waypoints_local)
                    if diverged:
                        # *** We diverged. RESET THE WAYPOINT ROUTE. ***                            
                        log.warning("Resetting the route!\n"*10)
                        self.reset_plan_to_goal()
                    else:
                        waypoints_valid = True
                        return waypoints, waypoints_local
            except WaypointerError as w:
                log.error("Waypointer error! {}".format(w))
                return None, None
        if n_tries >= max_tries - 2:
            log.error("Waypointer error!")
            raise RuntimeError("Waypointer error")

    def get_unsticking_waypoints(self, waypoints_control, midlow_controller, current_obs):
        swd = self.waypointerconf.stuck_waypoint_distance                
        if (hasattr(midlow_controller, 'dim_planner') and
            midlow_controller.dim_planner.goal_likelihood.describe() == "RegionIndicator"):
            # Trim the closest part of the polygon
            log.warning("Generating unsticking polygon")
            waypoints_control = geom_util.generate_unsticking_polygon(polygon=waypoints_control)
        elif (hasattr(midlow_controller, 'dim_planner') and
              midlow_controller.dim_planner.goal_likelihood.describe() == "SegmentSetIndicator"):
            log.warning("Dropping waypoints of segments close to and behind car")
            waypoints_control = npu.lock_nd(
                self.waypointer.drop_close_waypoints(waypoints_control, min_waypoint_distance=swd))
            waypoints_control = self.waypointer.drop_behind_waypoints(waypoints_control)
        else:
            log.warning("Changing waypoints for stuck vehicle (dropping close waypoints <= {})!".format(swd))
            try:
                waypoints_control = npu.lock_nd(self.waypointer.drop_close_waypoints(waypoints_control, min_waypoint_distance=swd))
            except Exception as e:
                print(e)
                log.error("Caught exception when trying to drop close waypoints!")
                log.error(e)
                waypoints_control = npu.lock_nd(np.asarray([[10., 0., 0.]]))
            if waypoints_control is None or waypoints_control.size == 0:
                log.error("No waypoints left! Adding one")
                waypoints_control = npu.lock_nd(np.asarray([[10., 0., 0.]]))
        return waypoints_control

class Countlog:
    def __init__(self):
        self.count = 0

    def __call__(self, w):
        log.debug("Waypoint count={}. Calls={}".format(w.shape[0], self.count))
        self.count += 1
