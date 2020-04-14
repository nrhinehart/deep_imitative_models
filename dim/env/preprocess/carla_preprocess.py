
import attrdict
import collections
import json
import logging
import numpy as np
import os
import pdb
import scipy.spatial
import skimage.transform as skt

import carla.transform as transform
from carla.transform import Translation, Rotation

import precog.utils.tfutil as tfutil
import precog.utils.class_util as classu
import precog.utils.img_util as imgu
import precog.utils.tensor_util as tensoru
import precog.dataset.serialized_dataset as serialized_dataset

log = logging.getLogger(os.path.basename(__file__))
np.set_printoptions(suppress=True, precision=4)
    
# Convert protobuf messages to numpy before we dill/pickle them.
vector3_to_np = lambda v: np.array([v.x, v.y, v.z])
control_to_np = lambda c: np.array([c.steer, c.throttle, c.brake, c.hand_brake, c.reverse])
transforms_rotation_to_np = lambda c: np.array([c.rotation.roll, c.rotation.pitch, c.rotation.yaw])
transform_to_loc = lambda t: np.squeeze(np.asarray(t.matrix[:3, -1]))

class LidarParams(attrdict.AttrDict):
    def __init__(self, meters_max=50, pixels_per_meter=2, hist_max_per_pixel=25, val_obstacle=1.):
        super().__init__(meters_max=meters_max,
                         pixels_per_meter=pixels_per_meter,
                         hist_max_per_pixel=hist_max_per_pixel)

class BEVFeatureParams:
    @staticmethod
    def create_v1():
        # Keep this constructor around.
        return BEVFeatureParams(build_occupancy_grid=True, build_overhead_lidar=True, build_overhead_semantic=False,
                                occupancy_above=True, occupancy_below=True, expand_bev=False, second_occupancy_above=True)
    
    @classu.member_initialize
    def __init__(self, build_occupancy_grid=True, build_overhead_lidar=True, build_overhead_semantic=False,
                 occupancy_above=True, occupancy_below=True, expand_bev=False, second_occupancy_above=False):
        pass

    def __repr__(self):
        return "BEVFeatureParams(oc={}, ol={}, sem={}, above={}, below={}, expand={}, 2nd_above={})".format(
            self.build_occupancy_grid, self.build_overhead_lidar, self.build_overhead_semantic,
            self.occupancy_above, self.occupancy_below, self.expand_bev, self.second_occupancy_above)

class PlayerObservations:
    @classu.member_initialize
    def __init__(self,
                 measurements,
                 t_index,
                 radius,
                 A=None,
                 agent_id_ordering=None,
                 empty_val=200,
                 waypointer=None,
                 frame=None):
        """This objects holds some metadata and history about observations. It's used as a pre-preprocessing object to pass around 
        outside of the model.

        :param measurements: list of measurements
        :param t_index: 
        :param radius: if multiagent, radius inside which to include other agents
        :param multiagent: whether to include other agents
        :param A: 
        :param agent_id_ordering: optional, None or dict str id : index of prediction.
        :returns: 
        :rtype: 

        """
        self.t = t_index
        
        # The sequence of player measurements.
        self.player_measurement_traj = [_.player_measurements for _ in measurements]
        # The transform at 'now'
        self.tform_t  = transform.Transform(self.player_measurement_traj[t_index].transform)
        # The inverse transform at 'now'
        self.inv_tform_t = self.tform_t.inverse()
        # The sequence of player forward speeds.
        self.player_forward_speeds = np.asarray([_.forward_speed for _ in self.player_measurement_traj])
    
        # The sequence of player accelerations. These are in world coords.
        self.accels_world = np.asarray([vector3_to_np(_.acceleration)
                                        for _ in self.player_measurement_traj])
        # Apply transform, undo translation.
        self.accels_local = (self.inv_tform_t.transform_points(self.accels_world) -
                             self.inv_tform_t.matrix[:3, -1])
        # (T,3) sequence of player positions (CARLA world coords).
        self.player_positions_world = np.asarray([vector3_to_np(_.transform.location)
                                                  for _ in self.player_measurement_traj])
        # (T,3) sequence of player positions (CARLA local coords / R2P2 world frames of transform at 'now')
        self.player_positions_local = self.inv_tform_t.transform_points(self.player_positions_world)

        # global player rotations.
        self.rotations = np.asarray([transforms_rotation_to_np(_.transform)
                                     for _ in self.player_measurement_traj])

        # Get current measurement
        self.measurement_now = measurements[self.t]

        if agent_id_ordering is None:
            # Get agents currently close to us.
            self.all_nearby_agent_ids = get_nearby_agent_ids(self.measurement_now, radius=radius)
            self.nearby_agent_ids = self.all_nearby_agent_ids[:A-1]
            # Extract just the aid from (dist, aid) pairs.
            self.nearby_agent_ids_flat = [_[1] for _ in self.nearby_agent_ids]
            # Store our ordering of the agents.
            self.agent_id_ordering = dict(
                tuple(reversed(_)) for _ in enumerate(self.nearby_agent_ids_flat))
        else:
            # Expand radius so we make sure to get the desired agents.
            self.all_nearby_agent_ids = get_nearby_agent_ids(self.measurement_now, radius=9999999*radius)
            self.nearby_agent_ids = [None]*len(agent_id_ordering)
            for dist, aid in self.all_nearby_agent_ids:
                if aid in agent_id_ordering:
                    # Put the agent id in the provided index.
                    self.nearby_agent_ids[agent_id_ordering[aid]] = (dist, aid)

        # Extract all agent transforms. TODO configurable people.
        self.npc_tforms_unfilt = extract_nonplayer_transforms_list(measurements, False)

        # Collate the transforms of nearby agents.
        self.npc_tforms_nearby = collections.OrderedDict()
        self.npc_trajectories = []
        for dist, nid in self.nearby_agent_ids:
            self.npc_tforms_nearby[nid] = self.npc_tforms_unfilt[nid]
            self.npc_trajectories.append([transform_to_loc(_) for _ in self.npc_tforms_unfilt[nid]])
        self.all_npc_trajectories = []                
        for dist, nid in self.all_nearby_agent_ids:
            self.all_npc_trajectories.append([transform_to_loc(_) for _ in self.npc_tforms_unfilt[nid]])

        history_shapes = [np.asarray(_).shape for _ in self.all_npc_trajectories]
        different_history_shapes = len(set(history_shapes)) > 1
        if different_history_shapes:
            log.error("Not all agents have the same history length! Pruning smallest agents until there's just one shape")
            while len(set(history_shapes)) > 1:
                history_shapes = [np.asarray(_).shape for _ in self.all_npc_trajectories]
                smallest_agent_index = np.argmin(history_shapes,axis=0)[0]
                self.all_npc_trajectories.pop(smallest_agent_index)

        # The other agent positions in world frame.
        self.agent_positions_world = np.asarray(self.npc_trajectories)
        # N.B. This reshape will fail if histories are different sizes.
        self.unfilt_agent_positions_world = np.asarray(self.all_npc_trajectories)
        self.n_missing = max(A - 1 - self.agent_positions_world.shape[0], 0)

        # length-A list, indicating if we have each agent.
        self.agent_indicators = [1] + [1] * len(self.npc_trajectories) + [0] * self.n_missing

        if self.n_missing == 0:
            pass
        elif self.n_missing > 0:
            # (3,)
            faraway = self.player_positions_world[-1] + 500
            faraway_tile = np.tile(faraway[None, None], (self.n_missing, len(measurements), 1))
            if self.n_missing == 0:
                pass
            elif self.n_missing == A - 1:
                self.agent_positions_world = faraway_tile
            else:
                self.agent_positions_world = np.concatenate(
                    (self.agent_positions_world, faraway_tile), axis=0)

        self.player_position_now_world = self.player_positions_world[self.t]
        oshape = self.agent_positions_world.shape
        uoshape = self.unfilt_agent_positions_world.shape
        apw_pack = np.reshape(self.agent_positions_world, (-1, 3))
        uapw_pack = np.reshape(self.unfilt_agent_positions_world, (-1, 3))

        # Store all agent current positions in agent frame.
        self.unfilt_agent_positions_local = np.reshape(self.inv_tform_t.transform_points(uapw_pack), uoshape)

        if A == 1:
            self.agent_positions_local = np.array([])
            self.agent_positions_now_world = np.array([])
            self.all_positions_now_world = self.player_position_now_world[None]
        else:
            self.agent_positions_local = np.reshape(self.inv_tform_t.transform_points(apw_pack), oshape)
            self.agent_positions_now_world = self.agent_positions_world[:, self.t]
            self.all_positions_now_world = np.concatenate(
                (self.player_position_now_world[None], self.agent_positions_now_world), axis=0)
        assert(self.all_positions_now_world.shape == (A, 3))

        if self.n_missing > 0:
            log.warning("Overwriting missing agent local positions with empty_val={}!".format(empty_val))
            self.agent_positions_local[-self.n_missing:] = empty_val

        self.yaws_world = [self.tform_t.yaw]

        # Extract the yaws for the agents at t=now.
        for tforms in self.npc_tforms_nearby.values():
            self.yaws_world.append(tforms[self.t].yaw)
        self.yaws_world.extend([0]*self.n_missing)

        assert(self.agent_positions_world.shape[0] == A - 1)
        assert(self.agent_positions_local.shape[0] == A - 1)
        assert(len(self.yaws_world) == A)
        assert(len(self.agent_indicators) == A)

        if waypointer is not None:
            # Get the light state from the most recent measurement.
            self.traffic_light_state, self.traffic_light_data = waypointer.get_upcoming_traffic_light(measurements[-1], sensor_data=None)
        else:
            log.warning("Not recording traffic light state in observation!")

        self.player_destination_world = waypointer.goal_position
        
        if self.player_destination_world is not None:
            self.player_destination_local = self.inv_tform_t.transform_points(self.player_destination_world[None])[0]
        else:
            self.player_destination_local = None
        self.yaws_local = [yaw_world - self.yaws_world[0] for yaw_world in self.yaws_world]

    def get_sfa(self, t):
        """Return (2d position, scalar forward_speed, 3d acceleration) at the specified index"""
        return (self.player_positions_local[t,:2],  self.player_forward_speeds[t], self.accels_local[t])

    def copy_with_new_empty_val(self, empty_val):
        return PlayerObservations(measurements=self.measurements,
                                  t_index=self.t_index,
                                  radius=self.radius,
                                  A=self.A,
                                  agent_id_ordering=self.agent_id_ordering,
                                  empty_val=empty_val)

    def copy_with_new_ordering(self, agent_id_ordering):
        return PlayerObservations(measurements=self.measurements,
                                  t_index=self.t_index,
                                  radius=self.radius,
                                  A=self.A,
                                  agent_id_ordering=agent_id_ordering)
    

class StreamingCARLALoader:
    @classu.member_initialize
    def __init__(self, settings, T_past, T, bev_feature_params=None, sdt_params=None, with_sdt=False):
        """This object is used to populate feeds for the model.

        :param settings: 
        :param T_past: 
        :param T: 
        :param bev_feature_params: 
        :returns: 
        :rtype: 

        """
        sensor_names = [_.SensorName for _ in self.settings._sensors]
        lidar_idx = sensor_names.index('Lidar32')
        self.lidar_sensor = self.settings._sensors[lidar_idx]
        self.lidar_params = self.settings.lidar_params
        self.measurements = []
        self.player_trajectory = []
        self.feed_dicts_and_transforms = {}
        # Default to v1.
        if self.bev_feature_params is None: self.bev_feature_params = BEVFeatureParams.create_v1()
        self.bev_and_lidar_kwargs = {'lidar_sensor': self.lidar_sensor,
                                     'lidar_params': self.lidar_params,
                                     'bev_feature_params': self.bev_feature_params}
        
    @property
    def T_past(self):
        return self._T_past

    @T_past.setter
    def T_past(self, v):
        self._T_past = v

    @T_past.getter
    def T_past(self):
        return self._T_past

    def prune_old(self, frame):
        """Prune feeds older outside of the time horizon of the current frame. Needs to be done regularly to avoid O(N) memory growth.

        :param frame: 
        :returns: 
        :rtype: 

        """
        
        max_prune_frame = frame - self.T - 1
        for i in range(max_prune_frame): self.feed_dicts_and_transforms.pop(i, None)

    def populate_expert_feeds(self, observations, S_future_world_frame, frame):
        """Populate an earlier feed dict with expert data

        :param observations: 
        :param S_future_world_frame: 
        :param frame: 
        :returns: 
        :rtype: 

        """
        earlier_frame = frame - self.T
        assert(earlier_frame > 0)
        
        # Get feed dict and transform from an earlier frame
        fd, tform = self.feed_dicts_and_transforms[earlier_frame]
        # These are the past positions with the most recent one occuring at the current frame. 
        player_future_local = tform.transform_points(observations.player_positions_world)[1:self.T+1, :2][None, None]
        # TODO create for other agents too... assumes A=1.
        # other_future_local = tform.transform_points(observations.agent_positions_world)
        fd[S_future_world_frame] = player_future_local
        return fd
        
    def populate_phi_feeds(self,
                           phi,
                           sensor_data,
                           measurement_buffer,
                           observations,
                           frame,
                           with_bev=True,
                           with_lights=True):
        """Populates the feed_dict values of phi for the current frame.

        :param sensor_data: 
        :param with_bev: 
        :returns: 
        :rtype: 

        """
        assert(measurement_buffer.maxlen > self.T)
        feed_dict = tfutil.FeedDict()
        # Store these feeds for potential later expert population.
        self.feed_dicts_and_transforms[frame] = (feed_dict, observations.inv_tform_t)
        B, H, W, C = tensoru.shape(phi.overhead_features)
        _, A = tensoru.shape(phi.yaws)
            
        # Extract robot pasts.
        pasts = observations.player_positions_local[-self.T_past:, :2][None]
        
        # Extract other agent pasts.
        pasts_other = observations.agent_positions_local[:, -self.T_past:, :2]

        # Indicate all present
        agent_presence = np.ones(shape=tensoru.shape(phi.agent_presence), dtype=np.float32)

        # Combine ego and other pasts.
        pasts_joint = np.concatenate((pasts, pasts_other))[:A][None]
        
        # Tile pasts.
        pasts_batch = np.tile(pasts_joint, (B, 1, 1, 1))

        yaws = np.tile(np.asarray(observations.yaws_local[:A])[None], (B, 1))

        feed_dict[phi.S_past_world_frame] = pasts_batch                
        feed_dict[phi.yaws] = yaws
        feed_dict[phi.agent_presence] = agent_presence
        feed_dict[phi.is_training] = np.array(False)

        # Feed the BEV features.
        if with_bev:
            bevs = build_BEV(frame_data=sensor_data, measurements=measurement_buffer[-1], **self.bev_and_lidar_kwargs)[None]
            assert(bevs.ndim == 4)
            
            if bevs.shape[1] > H:
                # Center crop the bev's preserving the pixels/meter of input.
                log.debug("Center-cropping...")
                # TODO move inside BEV building?
                bevs = imgu.batch_center_crop(bevs, target_h=H, target_w=H)

            if self.with_sdt:
                # TODO hardcoded and bad! Currently set to match the SDT params from the precog repo.
                for b in range(B):                
                    serialized_dataset._create_sdt(
                        bevs=bevs,
                        sdt_filename=None,
                        b=0,
                        H=H,
                        W=W,
                        C=C,
                        sdt_clip_thresh=0.5,
                        stamp=True,
                        save=False,
                        sdt_zero_h=3,
                        sdt_zero_w=8,
                        sdt_params={'clip_top': 1, 'clip_bottom': -3, 'normalize': True})

            # TODO make the model more efficient by using B=1.
            bevs = np.tile(bevs, (B, 1, 1, 1))            

            # Add the bevs to the feed dict.
            feed_dict[phi.overhead_features] = bevs

        if with_lights:
            light_string = observations.traffic_light_state.upper()
            light_string_batch = np.tile(np.asarray(light_string), (B,))
            feed_dict[phi.light_strings] = light_string_batch
            log.debug("Feeding traffic light state '{}' to model".format(light_string))
        else:
            log.warning("Not feeding traffic light state to model!")
        return feed_dict

class NumpyEncoder(json.JSONEncoder):
    """
    The encoding object used to serialize np.ndarrays
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
def dict_to_json(dict_datum, out_fn, b=0):
    """Used to serialize model feeds into json that can be used to train the model.

    :param dict_datum: 
    :param out_fn: 
    :returns: 
    :rtype: 

    """
    assert(not os.path.isfile(out_fn))
    # Assume that the input dict has keys with .name attributes (e.g. tf.Tensors)
    preproc_dict = {k.name.split(':')[0]: np.squeeze(v[b]) for k, v in dict_datum.items()}
    with open(out_fn, 'w') as f:
        json.dump(preproc_dict, f, cls=NumpyEncoder)
    return out_fn

def build_BEV(frame_data,
              measurements,
              lidar_sensor,
              lidar_params,
              bev_feature_params):
    """

    :param frame_data: 
    :param measurements: 
    :param lidar_sensor: 
    :param lidar_params: 
    :param bev_feature_params: 
    :returns: e.g. (splat, above, below, second_above)
    :rtype: 

    """
    
    # Build lidar features.
    if bev_feature_params.build_overhead_lidar:
        # (W, H)
        overhead_lidar = splat_lidar(lidar_sensor,
                                     frame_data.Lidar32,
                                     measurements.player_measurements,
                                     lidar_params=lidar_params)
        overhead_lidar_features = overhead_lidar[..., None]
        log.debug("Overhead LIDAR features shape: {}".format(overhead_lidar_features.shape))
        if bev_feature_params.build_occupancy_grid:
            ogrid = get_occupancy_grid(
                lidar_sensor=lidar_sensor,
                lidar_measurement=frame_data.Lidar32,
                player_measurements=measurements.player_measurements,
                lidar_params=lidar_params,
                above=bev_feature_params.occupancy_above,
                below=bev_feature_params.occupancy_below,
                second_above=getattr(bev_feature_params, 'second_occupancy_above', False))
            log.debug("Occupancy grid shape: {}".format(ogrid.shape))
            overhead_lidar_features = np.concatenate((overhead_lidar_features, ogrid), axis=-1)
    else:
        overhead_lidar_features = None

    # Build semantic features.
    if bev_feature_params.build_overhead_semantic:
        raise NotImplementedError        
    else:
        overhead_semantic_features = None

    # Build joint features. Always put semantic features at the end.
    if overhead_lidar_features is not None:
        if overhead_semantic_features is not None:
            overhead_features = np.concatenate((overhead_lidar_features, overhead_semantic_features), axis=-1)
        else:
            overhead_features = overhead_lidar_features
    elif overhead_semantic_features is not None:
        overhead_features = overhead_semantic_features
    else:
        overhead_features = None
    log.debug("Built BEV shape: {}".format(overhead_features.shape))
    return overhead_features

def get_rectifying_player_transform(player_measurements):
    """ Build transform to correct for pitch or roll of car. """
    ptx = player_measurements.transform
    p_rectify = transform.Transform()
    pT = Translation(0, 0, 0)
    # Undo the pitch and roll (not yaw!)
    pR = Rotation(-ptx.rotation.pitch, 0, -ptx.rotation.roll)
    # pR = Rotation(ptx.rotation.pitch, 0, ptx.rotation.roll)
    p_rectify.set(pT, pR)
    return p_rectify

def get_rectified_player_transform(player_measurements):
    return (get_rectifying_player_transform(player_measurements) *
            transform.Transform(player_measurements.transform))

def get_rectified_sensor_transform(lidar_sensor, player_measurements):
    p_rectify = get_rectifying_player_transform(player_measurements)
    
    # Transform to car frame, then undo the pitch and roll of the car frame.
    lidar_transform = p_rectify * lidar_sensor.get_transform()
    return lidar_transform

def get_rectified_depth_transform(lidar_sensor, player_measurements):
    p_rectify = get_rectifying_player_transform(player_measurements)
    
    # Transform to car frame, then undo the pitch and roll of the car frame.
    lidar_transform = p_rectify * lidar_sensor.get_transform().inverse()
    return lidar_transform

def rectify_rigid_rotation_of_sensor(image, player_measurements, K, K_inv):
    """TODO not sure if correct.
    :param image: Image from rigidly rotated sensor
    :param player_measurements: Measurements of the player to build the rectifying rotation
    :returns: rectified image
    :rtype: ndarray
    """

    R = get_rectifying_player_transform(player_measurements).matrix[:3, :3]
    return skt.warp(image.data, K @ R @ K_inv)

def flip_light_state(light_state):
    if light_state == "GREEN":
        return "RED"
    elif light_state == "YELLOW":
        return "YELLOW"
    elif light_state == "RED":
        return "GREEN"
    elif light_state == "INTERSECTION" or light_state == "NONE":
        return light_state
    else:
        raise ValueError("Unknown light state: '{}'".format(light_state))

def splat_lidar(lidar_sensor, lidar_measurement, player_measurements, lidar_params):
    """Convert a point cloud to a one-channel image centered around the car
    by projecting points to a ground plane histogram and normalizing it.

    :param lidar_sensor: sensor.Lidar
    :param lidar_measurement: a lidar measurement
    :param player_measurements: a player measurement
    :param lidar_params: dict parameters for the splatting
    :returns: 2d ndarray, size goverened by lidar parameters
    :rtype: ndarray
    """
    
    lidar_point_cloud = lidar_measurement.point_cloud

    # Get world -> rectified lidar transform
    lidar_transform = get_rectified_sensor_transform(lidar_sensor, player_measurements)

    # Transform points to the car frame
    lidar_points_at_car = np.asarray(lidar_transform.transform_points(lidar_point_cloud._array))

    permute = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32)
    lidar_points_at_car = (permute @ lidar_points_at_car.T).T
    return splat_points(lidar_points_at_car, lidar_params)

def splat_points(points, splat_params, nd=2):
    meters_max = splat_params['meters_max']
    pixels_per_meter = splat_params['pixels_per_meter']
    hist_max_per_pixel = splat_params['hist_max_per_pixel']
    
    # Allocate 2d histogram bins. Todo tmp?
    ymeters_max = meters_max
    xbins = np.linspace(-meters_max, meters_max+1, meters_max * 2 * pixels_per_meter + 1)
    ybins = np.linspace(-meters_max, ymeters_max+1, ymeters_max * 2 * pixels_per_meter + 1)
    hist = np.histogramdd(points[..., :nd], bins=(xbins, ybins))[0]
    # Compute histogram of x and y coordinates of points
    # hist = np.histogram2d(x=points[:,0], y=points[:,1], bins=(bins, ybins))[0]

    # Clip histogram 
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel

    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground
    return overhead_splat

def get_occupancy_grid(lidar_sensor, lidar_measurement, player_measurements,
                       lidar_params, above=True, below=False, second_above=False):
    """Get occupancy grid(s) (indicators of occupancy) at various heights

    :param lidar_sensor: 
    :param lidar_measurement: 
    :param player_measurements: 
    :param lidar_params: 
    :returns: (H, W, C) array of occupancy masks (C = 2 currently)
    :rtype: 

    """

    assert(sum([above, below]) >= 1)
    lidar_point_cloud = lidar_measurement.point_cloud

    # Get world -> rectified lidar transform
    lidar_transform = get_rectified_sensor_transform(lidar_sensor, player_measurements)

    # Transform points to the car frame
    lidar_points_at_car = np.asarray(lidar_transform.transform_points(lidar_point_cloud._array))

    permute = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32)
    lidar_points_at_car = (permute @ lidar_points_at_car.T).T

    z_threshold = -4.5  # sees low objects, but sometimes the lidar rolls and sees the road as an obstacle
    # z_threshold = -3.0  # doesn't see low objects, but the lidar is fine
    z_threshold_second_above = -2.0

    above_mask = lidar_points_at_car[:, 2] > z_threshold

    def get_occupancy_from_masked_lidar(mask):
        masked_lidar = lidar_points_at_car[mask]
        meters_max = lidar_params['meters_max']
        pixels_per_meter = lidar_params['pixels_per_meter']
        xbins = np.linspace(-meters_max, meters_max, meters_max * 2 * pixels_per_meter + 1)
        ybins = xbins
        grid = np.histogramdd(masked_lidar[..., :2], bins=(xbins, ybins))[0]
        grid[grid > 0.] = lidar_params['val_obstacle']
        return grid

    feats = ()
    if above:
        feats += (get_occupancy_from_masked_lidar(above_mask),)
    if below:
        feats += (get_occupancy_from_masked_lidar((1 - above_mask).astype(np.bool)),)
    if second_above:
        second_above_mask = lidar_points_at_car[:, 2] > z_threshold_second_above
        feats += (get_occupancy_from_masked_lidar(second_above_mask),)
    return np.stack(feats, axis=-1)

def extract_nonplayer_transforms_list(measurement_list, include_people=False):
    """Extract a dictionary of lists for each agent of their transforms.

    :param measurement_list: list Measurements
    :param include_people: whether to include people
    :rtype: dict : str agent id -> list Transform

    """
    transforms = collections.defaultdict(list)
    for m in measurement_list:
        tforms = extract_nonplayer_transforms(m, include_people)
        for k, v in tforms.items():
            transforms[k].append(v)
    return transforms

def extract_nonplayer_transforms(measurement, include_people=False):
    """Extract the other vehicle (and maybe people) Transforms

    :param measurements: Measurments inst
    :param include_people: whether to include people
    :returns: dict : agentkey -> transform.Transform or None
    :rtype: dict

    """
    transforms = {}
    
    for npmes in measurement.non_player_agents:
        aid = npmes.id
        agentkey = '{}'.format(aid)
        a_transforms = extract_nonplayer_transform(npmes, include_people)
        if a_transforms is not None:
            transforms[agentkey] = a_transforms
    return transforms

def extract_nonplayer_transform(npmes, include_people=False):
    if npmes.HasField('vehicle'):
        return transform.Transform(npmes.vehicle.transform)
    elif npmes.HasField('pedestrian') and include_people:
        return transform.Transform(npmes.pedestrian.transform)
    else:
        return None

def get_nearby_agent_ids(measurement, radius):
    """Get sorted agent distances and ids within a radius

    :param measurement: Measurement inst
    :param radius: float clip radius
    :returns: sorted agent distances and ids
    :rtype: list tuple (float distance, str agentid)

    """
    loc = vector3_to_np(measurement.player_measurements.transform.location)
    nearby = []
    transforms = extract_nonplayer_transforms(measurement)

    #H ACK
    # log.error("Strange criteria for nearby agents!")
    # acceptor = lambda x,y: y[0] < 115. and y[1] > 52
    for aid, tform in transforms.items():
        if tform is None: continue
        aloc = transform_to_loc(tform)
        dist = scipy.spatial.distance.euclidean(loc, aloc)
        if dist < radius:
            nearby.append((dist, aid))
    nearest = sorted(nearby)
    # pdb.set_trace()
    return nearest
