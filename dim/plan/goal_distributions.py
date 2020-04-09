
# Common 3rd-party
from abc import ABCMeta, abstractproperty, abstractmethod
import six
import logging
import numpy as np
import os
import pdb
import tensorflow as tf
import tensorflow_probability as tfp
# In-package, matplotlib setup first.

import precog.interface
import precog.utils.tfutil as tfutil
import precog.utils.class_util as classu
import precog.utils.np_util as npu
import precog.utils.rand_util as randu
import precog.utils.tensor_util as tensoru

import dim.env.util.tfutils as dimtfutil
import dim.env.util.geom_util as geom_util

log = logging.getLogger(os.path.basename(__file__))
tfd = tfp.distributions

def create(model, dimconf):
    if dimconf.goal_likelihood in ("BatchedMixtureGoalNormal", "GaussianMixture"):
        goal_likelihood = BatchedGaussianMixture(model, **dimconf.batched_gaussian_mixture)
    elif dimconf.goal_likelihood in ('empty', 'none', 'None'):
        goal_likelihood = EmptyGoalLikelihood(model)
    elif dimconf.goal_likelihood == "SegmentSetIndicator":
        goal_likelihood = SegmentSetIndicator(model)
    elif dimconf.goal_likelihood == "RegionIndicator":
        goal_likelihood = RegionIndicator(model, waypoint_count=dimconf.region_clip_K * 2)
    else:
        raise ValueError("Unknown goal likelihood: {}".format(dimconf.goal_likelihood))
    return goal_likelihood

@six.add_metaclass(ABCMeta)
class GoalLikelihood:
    @abstractmethod
    def log_prob(self, trajectories):
        """

        :param trajectories: (K, A, B, T, d)
        :returns: 
        :rtype: 

        """
        
    @abstractmethod
    def set_feeds(self, feed_dict, waypoint_sequence, **kwargs):
        pass

    @abstractproperty
    def goal_likelihood_feeds(self):
        pass

    @abstractmethod
    def describe(self):
        pass

    @property
    def current_waypoints(self):
        return self._current_waypoints

class EmptyGoalLikelihood:
    @classu.member_initialize
    def __init__(self, model, **kwargs):
        self._current_waypoints = []
        
    def log_prob(self, trajectories, *args, **kwargs):
        return tf.squeeze(tf.zeros(shape=tensoru.shape(trajectories)[:-2], dtype=tf.float64))

    def set_feeds(self, feed_dict, *args, **kwargs):
        return feed_dict

    @property
    def goal_likelihood_feeds(self):
        return []

    def describe(self):
        return "EmptyGoalLikelihood"

    @property
    def current_waypoints(self):
        return self._current_waypoints

class BatchedSingleGoalNormal(GoalLikelihood):
    @classu.member_initialize
    def __init__(self, model, scale_meters=None):
        if scale_meters is None:
            self.scale_meters = dimcfg.goal_likelihood_scale
        self.batch_shape = (model.consts.K, model.consts.B)
        self.event_shape = (model.consts.d,)
        self.full_shape = self.batch_shape + self.event_shape
        self._single_goal_ph = tf.compat.v1.placeholder(dtype=tf.float64, shape=self.event_shape, name='waypoint0')
        self._loc = tf.tile(self._single_goal_ph[None, None], self.batch_shape + (1,))
        self._scale_diag = tf.constant(self.scale_meters * np.ones(self.full_shape, dtype=np.float64))
        self._dist = tfp.distributions.MultivariateNormalDiag(loc=self._loc, scale_diag=self._scale_diag)
        self._gl_feeds = [self._single_goal_ph]
        self._current_waypoints = None
        
    def log_prob(self, trajectories):
        assert(tfutil.rank(trajectories) == 5)
        # (K, A, B, T, d) -> (K, B, d)
        s_T = trajectories[:, 0, :, -1, :]
        log_prob = self._dist.log_prob(s_T)
        assert(tfutil.shape(s_T) == self.full_shape)
        return log_prob
        
    def set_feeds(self, feed_dict, waypoint_sequence, **kwargs):
        assert(waypoint_sequence.ndim == 2)
        first_waypoint = waypoint_sequence[0, :2]
        self._current_waypoints = first_waypoint.copy()
        feed_dict[self._single_goal_ph] = self.current_waypoints
        return feed_dict

    @property
    def goal_likelihood_feeds(self):
        return self._gl_feeds

    def describe(self):
        return "Isonormal, scale={:.2f}".format(self.scale_meters)

class BatchedGaussianMixture(GoalLikelihood):
    @classu.member_initialize
    def __init__(self, model, waypoint_count, goal_likelihood_scale, weight_scheme):
        assert(self.weight_scheme in ('uniform', 'linear', 'exponential'))
        self.M = waypoint_count
        md = model.metadata
        self.batch_shape = (md.B, md.K)
        self.event_shape = (md.D,)
        self.full_shape = self.batch_shape + self.event_shape

        # Right-most dimension indexes components. (M, 2). The input positions will define the centers of the distributions.
        self._multi_goal_ph = tf.compat.v1.placeholder(dtype=tf.float64, shape=(self.M,) + self.event_shape, name='waypoint0')
        # Right-most dimension indexes components. (K, B, 2, M)
        self._multi_goal_batch = tf.tile(self._multi_goal_ph[None, None], self.batch_shape + (1, 1))
        # Right-most dimension indexes components. (K, B, M)
        Imx = np.tile(np.asarray(self.goal_likelihood_scale, dtype=np.float64)[None, None, None], self.batch_shape + (self.M,))
        # Right-most dimension indexes components. (M,)
        if self.weight_scheme == 'uniform':
            self.p = 1./self.M * np.ones(shape=(self.M,), dtype=np.float64)
        else:
            raise ValueError("unhandled weight scheme")
        
        # Right-most dimension indexes components. (K, B, M)
        p_batch = np.tile(self.p[None, None], self.batch_shape + (1,))
        self._components = tfd.MultivariateNormalDiag(loc=self._multi_goal_batch, scale_identity_multiplier=Imx)
        self._mixture = tfd.Categorical(probs=p_batch)
        self._dist = tfp.distributions.MixtureSameFamily(mixture_distribution=self._mixture, components_distribution=self._components)
        self._current_waypoints = None
        self._gl_feeds = [self._multi_goal_ph]

    def _traj_to_s_T(self, trajectories):
        assert(tensoru.rank(trajectories) == 5)
        s_T = tf.squeeze(trajectories[..., -1, :])
        assert(tensoru.shape(s_T) == self.full_shape)
        return s_T
    
    def prob(self, trajectories):
        return self._dist.prob(self._traj_to_s_T(trajectories))
        
    def log_prob(self, trajectories):
        # [TODO] HACK
        return self._dist.log_prob(self._traj_to_s_T(trajectories))
        
    def set_feeds(self, feed_dict, waypoint_sequence, **kwargs):
        assert(waypoint_sequence.ndim == 2)
        assert(waypoint_sequence.shape[1] >= 2)
        log.debug("Setting waypoint feeds. Shape={}".format(waypoint_sequence.shape))
        
        if waypoint_sequence.shape[0] < self.M:
            log.warning("Waypoint sequence contained less positions than the distribution expects! {}<{}".format(waypoint_sequence.shape[0], self.M))
            # Tile the data M times. This handles all cases of 1 <= waypoint.shape[0] < self.M
            waypoint_sequence = np.tile(waypoint_sequence, (self.M, 1))
        self._current_waypoints = npu.lock_nd(waypoint_sequence[:self.M, :2].copy())
        feed_dict[self._multi_goal_ph] = self._current_waypoints
        return feed_dict

    @property
    def goal_likelihood_feeds(self):
        return self._gl_feeds

    def describe(self):
        return "MoG, sig={:.2f}, M={}, weights={}".format(self.goal_likelihood_scale, self.M, self.weight_scheme)
        
class BatchedComponentsMixture(GoalLikelihood):
    @classu.member_initialize
    def __init__(self, graph_input, model_components, M=None, goal_likelihood_scale=None, weight_scheme=None):
        assert(self.weight_scheme in ('uniform', 'linear', 'exponential'))
            
        self.batch_shape = (graph_input.consts.K, graph_input.consts.B)
        self.event_shape = (graph_input.consts.d,)
        self.full_shape = self.batch_shape + self.event_shape

        # Right-most dimension indexes components. (M, 2). The input positions will define the centers of the distributions.
        self._multi_goal_ph = tf.compat.v1.placeholder(dtype=tf.float64, shape=(self.M,) + self.event_shape, name='waypoint0')
        # Right-most dimension indexes components. (K, B, M, 2)
        self._multi_goal_batch = tf.tile(self._multi_goal_ph[None, None], self.batch_shape + (1, 1))

        simple = False
        if simple:
            self._components = model_components.get_components()
            self._log_prob = 1 / self.M * tf.reduce_sum(self._components.log_prob(self._multi_goal_batch), axis=-1)
            self._prob = 1 / self.M * tf.reduce_prod(self._components.prob(self._multi_goal_batch), axis=-1)
        else:
            self._components_x = model_components.get_components()
            self._q_mu = self._components_x.mean()
            self._q_sigma = self._components_x.covariance()
            self._components = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=self._multi_goal_batch, covariance_matrix=self._q_sigma)
            
            # Right-most dimension indexes components. (K, B, M)
            # Imx = np.tile(np.asarray(self.goal_likelihood_scale, dtype=np.float64)[None, None, None], self.batch_shape + (self.M,))

            # Right-most dimension indexes components. (M,)
            if self.weight_scheme == 'uniform':
                self.p = 1./self.M * np.ones(shape=(self.M,), dtype=np.float64)
            elif self.weight_scheme == 'linear':
                linear_weights = np.linspace(1, self.M, self.M, dtype=np.float64)
                self.p = linear_weights / linear_weights.sum()
            elif self.weight_scheme == 'exponential':
                exp_weights = np.exp(np.linspace(1, self.M, self.M, dtype=np.float64))
                self.p = exp_weights / exp_weights.sum()
            else:
                raise ValueError("unhandled weight scheme")

            # Right-most dimension indexes components. (K, B, M)
            p_batch = np.tile(self.p[None, None], self.batch_shape + (1,))
            self._mixture = tfd.Categorical(probs=p_batch)
            self._dist = tfp.distributions.MixtureSameFamily(mixture_distribution=self._mixture, components_distribution=self._components)
            self._log_prob = self._dist.log_prob(self._q_mu[...,0,:])
            
            # self._current_waypoints = None
        self._gl_feeds = [self._multi_goal_ph]

    def _traj_to_s_T(self, trajectories):
        assert(tfutil.rank(trajectories) == 5)
        # (K, A, B, T, d) -> (K, B, d)
        s_T = trajectories[:, 0, :, -1]
        assert(tfutil.shape(s_T) == self.full_shape)
        return s_T
    
    def prob(self, trajectories):
        return self._prob
    
    def log_prob(self, trajectories):
        return self._log_prob
        
    def set_feeds(self, feed_dict, waypoint_sequence, **kwargs):
        """

        :param feed_dict: 
        :param waypoint_sequence: waypoints in LOCAL coordinates!
        :returns: 
        :rtype: 

        """
        assert(waypoint_sequence.ndim == 2)
        assert(waypoint_sequence.shape[1] >= 2)
        log.debug("Setting waypoint feeds. Shape={}".format(waypoint_sequence.shape))
        
        if waypoint_sequence.shape[0] < self.M:
            log.warning("Waypoint sequence contained less positions than the distribution expects! {}<{}".format(waypoint_sequence.shape[0], self.M))
            # Tile the data M times. This handles all cases of 1 <= waypoint.shape[0] < self.M
            waypoint_sequence = np.tile(waypoint_sequence, (self.M, 1))
        self._current_waypoints = npu.lock_nd(waypoint_sequence[:self.M, :2].copy())
        feed_dict[self._multi_goal_ph] = self._current_waypoints
        return feed_dict

    @property
    def goal_likelihood_feeds(self):
        return self._gl_feeds

    def describe(self):
        return "ComponentsMixture, M={}, weights={}".format(self.M, self.weight_scheme)

class SegmentSetIndicator(GoalLikelihood):
    @classu.member_initialize
    def __init__(self, model, planning_graph):
        raise NotImplementedError("needs to be updated to new code")
        self.M = dimcfg.mixture_waypoint_count
        self.do_shifts = False
        if self.do_shifts:
            M_comp = 3 * self.M
        else:
            M_comp = self.M
            
        # This will tile mu and sigma M times.
        self._components = ModelComponentsAtT(planning_graph, model.graph_input.consts.T, M=M_comp)

        self.batch_shape = (model.graph_input.consts.K, model.graph_input.consts.B)
        self.event_shape = (model.graph_input.consts.d,)
        self.full_shape = self.batch_shape + self.event_shape

        # K, A, B, M, d -> K, B, M, d
        mu = self._components.mu[:, 0]
        # K, A, B, M, d, d -> K, B, M, d, d. Sigma = sig @ sig^T.
        Sig = self._components.Sig[:, 0]
        # K, B, M, d, d
        Prec = tf.linalg.inv(Sig)
        # K, B, M-1, d
        mu_short = mu[..., :-1, :]
        # K, B, M-1, d, d
        Prec_short = Prec[..., :-1, :, :]

        # Right-most dimension indexes components. (M, 2). The input positions will define the vertices of the segments.
        self._vertices = tf.compat.v1.placeholder(dtype=tf.float64, shape=(self.M,) + self.event_shape, name='waypoint0')
        if self.do_shifts:
            vertices_pretile = tf.concat((self._vertices, self._vertices + [0, .5], self._vertices + [0, -.5]), axis=-2)
        else:
            vertices_pretile = self._vertices
        
        # Right-most dimension indexes components. (K, B, M, 2)        
        self._vertices_batch = tf.tile(vertices_pretile[None, None], self.batch_shape + (1, 1))
        
        # K, B, M-1, d
        start = self._vertices_batch[..., :-1, :]
        # K, B, M-1, d
        end = self._vertices_batch[..., 1:, :]
        # K, B, M-1, d
        diffs = end - start
        # K, B, M
        u_numerator = -1 * tf.einsum('kbmd,kbmde,kbme->kbm', diffs, Prec_short, start - mu_short)
        # K, B, M
        u_denominator = tf.einsum('kbmd,kbmde,kbme->kbm', diffs, Prec_short, diffs)
        # K, B, M

        # Don't divide where denominator is 0.
        self.u = dimtfutil.safe_divide(u_numerator, u_denominator)

        old = False

        q_T = tfd.MultivariateNormalFullCovariance(loc=self.planning_graph.mus_verlet[-1], covariance_matrix=self.planning_graph.sigmas[-1])
        # K, A, B -> K, B. Will account for the extra q_T term in the prior by subtracting this.
        log_q_T_x = q_T.log_prob(self.planning_graph.x[..., -1, :])[:, 0]
        
        self.u = tf.minimum(tf.maximum(self.u, 0.), 1.)
        # K, B, M, d
        self.xy = start + tf.einsum('kbm,kbmd->kbmd', self.u, diffs)
        # K, B, M
        self.losses = tf.einsum('kbmd,kbmde,kbme->kbm', self.xy - mu_short, Prec_short, self.xy - mu_short)
        self._log_probs = -1 * self.losses
        # K, B
        self._log_prob = tf.reduce_max(self._log_probs, axis=-1) - log_q_T_x

    def log_prob(self, trajectories):
        return self._log_prob

    def set_feeds(self, feed_dict, waypoint_sequence, **kwargs):
        assert(waypoint_sequence.ndim == 2)
        assert(waypoint_sequence.shape[1] >= 2)
        log.debug("Setting waypoint feeds. Shape={}".format(waypoint_sequence.shape))
        
        if waypoint_sequence.shape[0] < self.M:
            log.warning("Waypoint sequence contained less positions than the distribution expects! {}<{}".format(waypoint_sequence.shape[0], self.M))
            # Make the same segments, but in reverse. Avoids creating spurious segments
            # because segment_i = (waypoint_sequence[i], waypoint_sequence[i+1]))
            waypoint_sequence_dup = np.concatenate((waypoint_sequence, waypoint_sequence[-2::-1]), axis=0)
            # Tile the data M times. This handles all cases of 1 <= waypoint.shape[0] < self.M
            waypoint_sequence = np.tile(waypoint_sequence_dup, (self.M, 1))
        self._current_waypoints = npu.lock_nd(waypoint_sequence[:self.M, :2].copy())
        feed_dict[self._vertices] = self._current_waypoints
        return feed_dict        

    def goal_likelihood_feeds(self):
        return []

    def describe(self):
        return "SegmentSetIndicator"

    @property
    def current_waypoints(self):
        return self._current_waypoints

class RegionIndicator(GoalLikelihood):
    @classu.member_initialize
    def __init__(self, model, waypoint_count, T=None):
        # self.M = dimcfg.mixture_waypoint_count
        self.do_shifts = False
        self.M = waypoint_count
        M_comp = self.M + 1
        if T is None: T = model.metadata.T
        
        mu_T = model.sampled_output.rollout.mu[..., T-1, :]
        self._sigma_T_in = model.sampled_output.rollout.sigma[..., T-1, :, :]
        # sigel_T = model.sampled_output.rollout.sigel[..., T-1, :, :]
        # # expm(X) expm(-X) = I -> (expm(X))^{-1} = expm(-X). Avoids computing any inverses explicitly.
        # prec_T = tf.linalg.expm(-1 * sigel_T)

        # This should still be sigma_T (since it should already be symmetric, but it ensures it's actually symmetric.
        sigma_T_sym = .5 * (self._sigma_T_in + tensoru.backswap(self._sigma_T_in))
        # Perform eigendecomposition. AQ = Q L  
        eig_vals, Q = tf.linalg.eigh(sigma_T_sym)
        # Ensure the eigenvalues have minimum values.
        eps = 1e-2
        eig_vals_clip = tf.where(eig_vals > eps, eig_vals, eps*tf.ones_like(eig_vals))
        Linv = tf.matrix_diag(1. / eig_vals_clip)
        L = tf.matrix_diag(eig_vals_clip)
        QT = tensoru.backswap(Q)
        # A = Q L Q^{-1} = Q L Q^T
        self._sigma_T_out = tf.einsum('...ij,...jk,...kh->...ih', Q, L, QT)
        # A^{-1} = Q^T L^{-1} Q
        self._prec_T_out = tf.einsum('...ij,...jk,...kh->...ih', QT, Linv, Q)

        # prec_T = tf.linalg.inv(sigma_T)

        vectile = lambda x: tf.tile(x[..., None, :], (1, 1, 1, M_comp, 1))
        mattile = lambda x: tf.tile(x[..., None, :, :], (1, 1, 1, M_comp, 1, 1))
        mu_T_tile_vertices = vectile(mu_T)
        prec_T_tile_vertices = mattile(self._prec_T_out)
        sigma_T_tile_vertices = mattile(self._sigma_T_out)
        q_T_dist_edges = tfd.MultivariateNormalFullCovariance(
            loc=mu_T_tile_vertices[..., :-1, :],
            covariance_matrix=sigma_T_tile_vertices[..., :-1, :, :],
            validate_args=True)
        prec_T_tile_edges = prec_T_tile_vertices[..., :-1, :, :]

        # (B, K)
        log_q_x = model.sampled_output.base_and_log_q.log_q_samples
        # (B, K, A, T)
        log_q_T_x = log_q_x.op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[0][..., T-1]

        self.batch_shape = (model.metadata.B, model.metadata.K, 1)
        self.event_shape = (model.metadata.D,)
        self.full_shape = self.batch_shape + self.event_shape
    
        # K, B, M+1, d, d -> K, B, M, d
        mu_T_tile_edges = mu_T_tile_vertices[..., :-1, :]
        # K, B, M, d, d
        # Prec_edges = Prec

        # Right-most dimension indexes components. (M, 2). The input positions will define the vertices of the edges.
        self.vertices = tf.compat.v1.placeholder(dtype=tf.float64, shape=(self.M,) + self.event_shape, name='waypoint0')
        first_vertex = self.vertices[..., 0, :]

        # Close the polygon by connecting the last vertex to the first.
        _sequential_polygon_edges = tf.concat((self.vertices, first_vertex[..., None, :]), axis=-2)
        _sequential_polygon_edges_batch = tensoru.tile_to_batch(_sequential_polygon_edges, self.batch_shape)

        # K, B, M, d
        start = _sequential_polygon_edges_batch[..., :-1, :]
        # K, B, M, d
        end = _sequential_polygon_edges_batch[..., 1:, :]
        # K, B, M, d
        segment_vector = end - start

        # K, B, M
        u_numerator = -1 * tf.einsum('bkamd,bkamde,bkame->bkam', segment_vector, prec_T_tile_edges,  start - mu_T_tile_edges)
        # K, B, M
        u_denominator = tf.einsum('bkamd,bkamde,bkame->bkam', segment_vector, prec_T_tile_edges, segment_vector)

        # K, B, M
        u_edges = dimtfutil.safe_divide(u_numerator, u_denominator)

        # Clip to constrain each point to each segment.
        u_edges = tf.minimum(tf.maximum(u_edges, 0.), 1.)

        # K, B, M, d. The best xy points along the polygon for each segment.
        xy_edges = start + tf.einsum('bkam,bkamd->bkamd', u_edges, segment_vector)

        # These are the log probs of the xy points along the edges (best points of exterior). They're relevant when the mu's are outside.
        # The max is applied across edges.
        log_q_T_outside = tf.reduce_max(q_T_dist_edges.log_prob(xy_edges), axis=-1)
        # (K, A, B) -> (K, B)
        log_q_T_inside = q_T_dist_edges.log_prob(mu_T_tile_edges)[...,0]
        
        # Bools of whether the gaussian is inside the region.
        inside_indicators = geom_util.inside_polygon(edges=_sequential_polygon_edges_batch, point=mu_T)
        # (B, K) the final log GoalLikelihood = log Indicator(s_T \in poly)
        self._log_prob_As = tf.where(inside_indicators, log_q_T_inside, log_q_T_outside) - log_q_T_x
        # Assumes A=1.
        self._log_prob = tf.squeeze(self._log_prob_As)
        
    def log_prob(self, trajectories):
        return self._log_prob

    def set_feeds(self, feed_dict, waypoint_sequence, **kwargs):
        assert(waypoint_sequence.ndim == 2)
        assert(waypoint_sequence.shape[1] >= 2)
        log.debug("Setting waypoint feeds. Shape={}".format(waypoint_sequence.shape))
        # assert(waypoint_sequence.shape[0] == self.M)        
        if waypoint_sequence.shape[0] < self.M:
            log.warning("Waypoint sequence contained less positions than polygon expects! {}<{}".format(waypoint_sequence.shape[0], self.M))
            # Make the same edges, but in reverse. Avoids creating spurious edges
            # because segment_i = (waypoint_sequence[i], waypoint_sequence[i+1]))
            dups = np.tile(waypoint_sequence[[-1]], (self.M-waypoint_sequence.shape[0], 1))
            waypoint_sequence = np.concatenate((waypoint_sequence, dups), axis=0)
        self._current_waypoints = npu.lock_nd(waypoint_sequence[:self.M, :2].copy())
        log.info("Final polygon vertices to be fed: {}".format(self._current_waypoints))
        feed_dict[self.vertices] = self._current_waypoints
        return feed_dict        

    def goal_likelihood_feeds(self):
        return []

    def describe(self):
        return "RegionIndicator"

    @property
    def current_waypoints(self):
        return self._current_waypoints
