
import copy
import logging
import os
import numpy as np
import pdb
import tensorflow as tf

import precog.utils.class_util as classu
import precog.utils.np_util as npu
import precog.utils.tensor_util as tensoru

log = logging.getLogger(os.path.basename(__file__))

class PosteriorComponents:
    @classu.member_initialize
    def __init__(self, log_prior, log_goal_likelihood, log_posterior, dlog_prior_dz, dlog_goal_likelihood_dz, dlog_posterior_dz):
        """Stores posterior inference targets and some gradients of the posterior.

        :param log_prior: 
        :param log_goal_likelihood: 
        :param log_posterior: 
        :param dlog_prior_dz: 
        :param dlog_goal_likelihood_dz: 
        :param dlog_posterior_dz: 
        :returns: 
        :rtype: 

        """
        self.inference_targets = (self.log_prior, self.log_goal_likelihood, self.log_posterior)
        self.gradient_step_targets = self.inference_targets + (self.dlog_posterior_dz,)

    def validate_shape(self, objective_shape):
        # Ensure each term matches our shape expectations.        
        assert(tensoru.shape(self.log_goal_likelihood) == objective_shape)
        assert(tensoru.shape(self.log_prior) == objective_shape)
        assert(tensoru.shape(self.log_posterior) == objective_shape)

class DIMPlanner:    
    @classu.member_initialize
    def __init__(self, model, goal_likelihood, dimconf, sess, check_gradients=True):
        """ Stores the model, constructs the posterior, enables gradient-ascent on the posterior

        :param model: 
        :param goal_likelihood: 
        :param dimconf: 
        :param sess: 
        :param check_gradients: whether to check gradients exist.

        """
        log.info("Instantiating {}.".format(self.__class__.__name__))
        # --- Define some useful functions.
        # (B, K, A, T, D) -> (B, K, T, D)
        partial_derivative = lambda t, dv: tf.gradients(t, dv)[0]

        # --- Extract and shape the trajectories and planning variables from the model.
        sampled_output = model.sampled_output
        sampled_rollout = sampled_output.rollout
        self.Z = sampled_output.base_and_log_q.Z_sample
        self.objective_shape = (model.metadata.B, model.metadata.K)
        # (B, K, A, T, D)
        self.S_world_frame = sampled_rollout.S_world_frame
        # (B, K)
        log_prior = tf.identity(sampled_output.base_and_log_q.log_q_samples, "log_prior")
        
        # --- Compute the log goal likelihood and form the planning criterion.
        goal_log_likelihood_values = tf.identity(goal_likelihood.log_prob(self.S_world_frame), "log_goal_likelihoods")
        log_posteriors = tf.identity(log_prior + goal_log_likelihood_values, "log_posteriors")

        # --- Compute the gradients.
        log.info("Computing planning log-posterior gradients...")
        dlogq_dz = tf.identity(partial_derivative(log_prior, self.Z), "dlogq_dz")

        if goal_likelihood.describe() == "EmptyGoalLikelihood": dloggoal_dz = tf.zeros_like(dlogq_dz)
        else: dloggoal_dz = partial_derivative(goal_log_likelihood_values, self.Z)
        dloggoal_dz = tf.identity(dloggoal_dz, "dloggoal_dz")
        
        # Note gradient is computed by "hand".
        # (K, B, T, D)
        dlog_posterior_dz = tf.identity(dlogq_dz + dloggoal_dz, "dlog_posterior_dz")

        # -- Package the components of the posterior.
        self.posterior_components = PosteriorComponents(
            log_prior=log_prior,
            log_goal_likelihood=goal_log_likelihood_values,
            log_posterior=log_posteriors,
            dlog_prior_dz=dlogq_dz,
            dlog_goal_likelihood_dz=dloggoal_dz,
            dlog_posterior_dz=dlog_posterior_dz)
        log.info("Done computing gradients.")
        self.posterior_components.validate_shape(self.objective_shape)
        log.info("Done instantiating {}.".format(self.__class__.__name__))
    
    def check_planning_gradients(self, feed_dict=None):
        if getattr(self, 'has_checked_gradients', False): 
            self.has_checked_gradients = True
        else:
            return

        log.info("Checking planning gradients.")

        if feed_dict is not None:
            log.info("Checking cross-z gradients.")
            z = self.sess.run(self.Z)
            z_init = z.copy()
            feed_dict[self.Z] = z_init
            grad0 = self.sess.run(self.posterior_components.dlog_posterior_dz, feed_dict)

            # Check that changing z changes the gradients correctly.
            for k in [0, 1, 2, -1, -2, -3]:
                z = z_init.copy()
                z[:, k] += np.random.normal(size=z[:, k].shape)
                feed_dict[self.Z] = z
                grad1 = self.sess.run(self.posterior_components.dlog_posterior_dz, feed_dict)
                grad0_k = grad0[:, k]
                grad1_k = grad1[:, k]
                grad0_notk = np.delete(grad0, k, 1)
                grad1_notk = np.delete(grad1, k, 1)
                
                gradx_k_close = np.isclose(grad0_k, grad1_k)
                gradx_notk_close = np.isclose(grad0_notk, grad1_notk)

                # Ensure changing z changes the correct gradient.
                assert(not gradx_k_close.any())
                # Ensure other zs don't change the incorrect gradients.
                assert(gradx_notk_close.all())
                
            for b in [0, 1, 2, -1, -2, -3]:
                z = z_init.copy()
                z[b] += np.random.normal(size=z[b].shape)
                feed_dict[self.Z] = z
                grad1 = self.sess.run(self.posterior_components.dlog_posterior_dz, feed_dict)
                grad0_b = grad0[b]
                grad1_b = grad1[b]
                grad0_notb = np.delete(grad0, b, 0)
                grad1_notb = np.delete(grad1, b, 0)
                
                gradx_b_close = np.isclose(grad0_b, grad1_b)
                gradx_notb_close = np.isclose(grad0_notb, grad1_notb)

                # Ensure changing z changes the correct gradient.
                # This can fail randomly, however, if the gradients magically align from the z's being too similar.
                assert(not gradx_b_close.any())
                # Ensure other zs don't change the incorrect gradients.
                # This can fail randomly, however, if the gradients magically align from the z's being too similar.
                assert(gradx_notb_close.all())

    def _prepare_feed_dict(self, feed_dict, waypoint_states):
        waypoint_states = (waypoint_states)
        self.goal_likelihood.set_feeds(feed_dict, waypoint_states)

    def initialize_z(self, feed_dict, n=15, reuse=False):
        # if self.dimconf.use_templates:
        #     log.info("Initializing z zimization with template.".format(n))
        #     fd_inv = copy.copy(feed_dict)
        #     fd_inv.update({self.planning_graph.x_inverse_input.data: self.centers_X})
        #     best_z = self.sess.run(self.Z_inverse_output, fd_inv)
        #     sample_z = self.sess.run(self.Z)
        #     # Throw in some random z too.
        #     log.info("Initializing z zimization with some sampled z's.".format(n))
        #     best_z[:self.planning_graph.K//3, :, self.planning_graph.B//2] = sample_z[:self.planning_graph.K//3, :, self.planning_graph.B//2]
        #     # Throw in some stationary z too.
        #     stat_z = self.get_stationary_z(feed_dict)
        #     if self.dimconf.add_stationary_zs:
        #         log.info("Initializing z zimization with some stationary z's.".format(n))
        #         best_z[-self.planning_graph.K//3:, :, self.planning_graph.B//2] = stat_z[-self.planning_graph.K//3:, :, self.planning_graph.B//2]
        #     return best_z
        
        if self.Z in feed_dict:
            del feed_dict[self.Z]

        if reuse and hasattr(self, '_best_z_init'):
            log.debug("Re-using the initial z")
            return self._best_z_init.copy()

        log.info("Initializing z optimization by sampling {} times.".format(n))
        best_z = None
        best_log_posterior = None
        
        for i in range(n):
            log_posterior, z = self.sess.run([self.posterior_components.log_posterior, self.Z], feed_dict)
            if best_z is None:
                best_log_posterior = log_posterior
                best_z = z
            log_posterior[np.logical_not(np.isfinite(log_posterior))] = -np.inf
            best_better_mask = log_posterior > best_log_posterior
            if best_better_mask.any():
                kbetter, bbetter = np.where(best_better_mask)
                best_z[kbetter, :, bbetter] = z[kbetter, :, bbetter]
        if self.dimconf.initialize_some_stationary_z:
            ksome = slice(0, self.K//3)
            bsome = slice(0, self.B//2)
            best_z[ksome, :, bsome] = self.get_stationary_z(feed_dict)[ksome, :, bsome]
        self._best_z_init = best_z.copy()
        return best_z

    def get_stationary_z(self, feed_dict):
        raise NotImplementedError("Haven't reimplemented inverse-graph computation")
    
        fd2 = copy.copy(feed_dict)
        if self.Z_inverse_output in feed_dict:
            del feed_dict[self.Z_inverse_output]
        xzero = np.zeros(tensoru.shape((self.planning_graph.x_inverse_input)), dtype=np.float32)
        xzero += np.random.normal(size=xzero.shape, loc=0., scale=1e-3)
        fd2[(self.planning_graph.x_inverse_input)] = xzero
        return self.sess.run(self.Z_inverse_output, fd2)

    def plan(self,
             waypoint_states,
             feed_dict,
             logq_thresh=125,
             criterion_thresh=125.1,
             loggoal_thresh=-3,
             clip_norm=5): 
        """

        :param waypoint_states: ndarray, in local coordinates!
        :param sess: 
        :param feed_dict:
        :returns: 
        :rtype: 

        """
        lr = self.dimconf.planning_lr
        max_steps = self.dimconf.max_planning_steps
        min_steps = self.dimconf.min_planning_steps
        assert(min_steps <= max_steps)
        self._prepare_feed_dict(feed_dict, waypoint_states)
        if self.check_gradients: self.check_planning_gradients(feed_dict=feed_dict)
        
        top_k = self.dimconf.max_top_k
        ss = "Beginning planning. \nWaypoint(s): {}".format(npu.tabformat(waypoint_states))
        ss += "\n\tGoal likelihood: {}".format(self.goal_likelihood.__class__.__name__)
        ss += "\n\tMin steps: {:.3f}".format(min_steps)
        ss += "\n\tMax steps: {:.3f}".format(max_steps)
        ss += "\n\tTop K: {:.3f}".format(top_k)
        ss += "\n\tLearning rate: {:.3g}".format(lr)
        log.info(ss)
        log.debug("Randomizing z initialization")

        # Initialize the current and best planning variables and their log_posterior.
        # best_z = z_cur = self.sess.run(self.Z)
        best_z = z_cur = self.initialize_z(feed_dict, n=1)
            
        log_posterior = best_log_posterior = -np.inf * np.ones(self.objective_shape, dtype=np.float32)
        n_best_stable = 0

        i = 0

        for i in range(max_steps):
            feed_dict[self.Z] = z_cur
                        
            try:
                log_prior, log_prob_goal, log_posterior, zgrad = self.sess.run(self.posterior_components.gradient_step_targets, feed_dict)
            except tf.errors.InvalidArgumentError as e: 
                log.error("Caught error when performing inference: {}".format(e))
                z_cur = best_z.copy()
                continue

            finite_mask = np.isfinite(zgrad)
            nonfinite_mask = np.logical_not(finite_mask)
            if nonfinite_mask.any():
                log_posterior_mask = nonfinite_mask.sum(axis=(-2, -1)) > 0
                # Zero junk gradients.
                zgrad[nonfinite_mask] = 0.
                # Reset z's that led to junk gradients.
                z_cur[nonfinite_mask] = self.sess.run(self.Z)[nonfinite_mask]
                # Trash the log_posterior of the junk gradients.
                log_posterior[log_posterior_mask] = -np.inf
                log_prior[log_posterior_mask] = -np.inf
                log_prob_goal[log_posterior_mask] = -np.inf

                if nonfinite_mask.all():
                    log.warning("All gradients are non-finite!")
            assert(np.isfinite(zgrad).all())

            # The mask of where log_posterior are better than before.
            bests_better_mask = log_posterior > best_log_posterior
            any_better = bests_better_mask.any()

            if any_better:
                # Update the best log_posterior according to the better mask.
                best_log_posterior[bests_better_mask] = log_posterior[bests_better_mask]
                # Update the best z's from the current z's according to the best mask.
                bbetter, kbetter = np.where(bests_better_mask)
                best_z[bbetter, kbetter] = z_cur[bbetter, kbetter]
                best_top_k = np.sort(best_log_posterior.ravel())[-top_k:][::-1]

            # Order the plans by their log_posterior.
            log_posterior_ordering = np.argsort(log_posterior.ravel())
            # Ordering of the best plans by their log_posterior.
            top_k_inds = log_posterior_ordering[-top_k:][::-1]
            any_top_k_better = bests_better_mask.ravel()[top_k_inds].any()

            # Compute stability of top K log_posterior.
            if any_top_k_better: n_best_stable = 0
            else: n_best_stable += 1

            if i >= min_steps:
                # Check if log probs diffs have converged.
                # TODO deprecate these thresholds
                log_prior_clear = (log_prior > logq_thresh)
                goal_likelihood_clear = (log_prob_goal > loggoal_thresh)
                criterion_clear = (log_posterior > criterion_thresh)
                useful_mask = np.logical_and(log_prior_clear, criterion_clear, goal_likelihood_clear)
                top_k_useful = useful_mask.ravel()[top_k_inds].all()
                # Convergence criterion: log q and planning criterion thresholds pass, min iters pass, and stable for a few iters.
                convergence_criterion = i > min_steps and top_k_useful
                if convergence_criterion:
                    log.info("Satisfied planning convergence criterion")
                    break
            else:
                pass

            # Gradient ascent on the log_posterior. Reshape to (..., Td)
            zgradvec = lr * zgrad.reshape((-1, zgrad.shape[-2] * zgrad.shape[-1]))
            # Compute norms across vectors
            zgradnorm = np.linalg.norm(zgradvec, axis=-1)
            zgradvec_clipped = (clip_norm * zgradvec.T / np.maximum(zgradnorm, clip_norm)).T
            zgrad_clipped = zgradvec_clipped.reshape(zgrad.shape)

            # APPLY THE GRADIENT.
            z_cur += zgrad_clipped

            # Print current planning statistics.
            period = i % 1 == 0
            logfunc = log.info if period else log.debug
            logfunc("Planning goal at {}/{}:\n\tLog prob: {}\n\tLog_Posterior: {}.\n\tGoal likelihoods: {}\n\tBest log_posterior: {}".format(
                i, max_steps, log_prior.ravel()[top_k_inds], log_posterior.ravel()[top_k_inds], log_prob_goal.ravel()[top_k_inds], best_top_k))

        # Set to the best z we've had.
        feed_dict[self.Z] = best_z

        # Evaluate nodes we don't evaluate during gradient descent.
        # (T, d)
        targets = (self.model.phi.S_past_world_frame, self.S_world_frame) + self.posterior_components.inference_targets
        past_states_local, x_planned, log_prior, log_goal_likelihoods, log_posteriors = self.sess.run(targets, feed_dict)
        log_prior_clear = (log_prior > logq_thresh)
        goal_likelihood_clear = (log_prob_goal > loggoal_thresh)
        criterion_clear = (log_posteriors > criterion_thresh)
        useful_mask = np.logical_and(log_prior_clear, criterion_clear, goal_likelihood_clear)

        finite_mask = np.isfinite(log_posteriors)
        nonfinite_mask = np.logical_not(finite_mask)
        if nonfinite_mask.any():
            raise RuntimeError("all log_posterior should be finite...")
        
        plan = Plan(planned_trajectories=(x_planned),
                    planned_trajectories_tf=self.S_world_frame,
                    z_planned=feed_dict[self.Z],
                    waypoint_target=waypoint_states,
                    past_states_local=(past_states_local),
                    log_prior=log_prior,
                    log_posterior=log_posteriors,                    
                    goal_log_likelihoods=log_goal_likelihoods,
                    useful_plans_mask=useful_mask,
                    planner=self,
                    steps=i)
        return plan
        
class Plan:
    @classu.member_initialize
    def __init__(self,
                 planned_trajectories,
                 waypoint_target,
                 past_states_local,
                 z_planned,
                 log_prior,
                 log_posterior,
                 goal_log_likelihoods,
                 useful_plans_mask,
                 planner,
                 steps=0,
                 to_waypoints=True,
                 planned_trajectories_tf=None,
                 valid_plan_indicators=None,
                 best_log_posterior=None):
        """The object that holds plan(s) and their metadata.

        :param planned_trajectories: 
        :param waypoint_target: local coordinates
        :param past_states_local: 
        :param z_planned: 
        :param log_prior: 
        :param log_posterior: 
        :param goal_log_likelihoods: 
        :param useful_plans_mask: 
        :param planner: 
        :param steps: 
        :param to_waypoints: 
        :param planned_trajectories_tf: 
        :param valid_plan_indicators: 
        :param best_log_posterior: 
        :returns: 
        :rtype: 

        """
        self.planned_trajectories = planned_trajectories
        self.past_states_local = past_states_local
        self.planned_trajectories_tf = planned_trajectories_tf
        self.waypoint_target = waypoint_target
        self._lock()
        assert(z_planned.shape == self.planned_trajectories.shape)
        self.current_target_forward_speed = np.nan
        self.current_forward_speed_error = np.nan            
        # The count of the number of plans.
        self.size = z_planned.shape[0] * z_planned.shape[1]
        # The shape of a single plan.
        self.single_plan_shape = z_planned.shape[-2:]
        # Reshape the trajectories to be indexed by flat plan ordering.
        self.planned_trajectories_flat = self.planned_trajectories.reshape((self.size,) + self.single_plan_shape)
        # Compute the ordering of planned trajectories from highest log_posterior to lowest log_posterior.
        self.plan_ordering_flat = np.argsort(log_posterior.ravel())[::-1]
        # Store the best plan.
        self.best_planned_traj = self.planned_trajectories_flat[self.plan_ordering_flat[0]]
        # Store the flat mask.
        self.useful_plans_mask_flat = useful_plans_mask.ravel()
        # Count the number of useful plans.
        self.n_useful = self.useful_plans_mask_flat.sum()
        # If the best plan's not useful, it's risky.
        self.is_risky = not self.useful_plans_mask_flat[self.plan_ordering_flat[0]]
        # Store values of the best plan.
        self.best_plan_log_prior = self.log_prior.ravel()[self.plan_ordering_flat[0]]
        self.best_plan_log_posterior = self.log_posterior.ravel()[self.plan_ordering_flat[0]]
        self.best_plan_loggoal = self.goal_log_likelihoods.ravel()[self.plan_ordering_flat[0]]

        # (2,) ?
        self.best_planned_traj_goal = self.best_planned_traj[..., -1, :]
        self.chosen_waypoint_idx = np.argmin(np.linalg.norm(self.waypoint_target[:, :2] - self.best_planned_traj_goal, axis=-1))
        self.chosen_waypoint = self.waypoint_target[self.chosen_waypoint_idx]

    def _lock(self):
        # Lock important data.
        npu.lock_nd(self.log_posterior)
        npu.lock_nd(self.z_planned)
        npu.lock_nd(self.log_prior)
        npu.lock_nd(self.useful_plans_mask)
        npu.lock_nd(self.past_states_local)
        npu.lock_nd(self.planned_trajectories)
        npu.lock_nd(self.waypoint_target)

    def get_plan_position(self, t_future_index):
        """Returns the t'th position into the planned trajectories (in coordinates of original frame)

        :param t_future_index: 
        :returns: 
        :rtype: 

        """
        # Get the first trajectory from the ordering.        
        # Get the t'th index of the trajectory.
        return self.best_planned_traj[t_future_index, :]
