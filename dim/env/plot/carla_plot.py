
import dill
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from PIL import Image
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches

import precog.utils.class_util as classu
import dim.env.preprocess.carla_preprocess as carla_preprocess

log = logging.getLogger(os.path.basename(__file__))
np.set_printoptions(suppress=True, precision=3)

def online_plot(model,
                measurements,
                sensor_data,
                overhead_lidar,
                overhead_semantic,
                fd,
                axes,
                fig,
                plot_state,
                plottables=[]):
    # TODO make Plottable classes for these.
    # if plot_state['overhead_lidar_im'] is not None: plot_state['overhead_lidar_im'].remove()
    if plot_state['frontal_im'] is not None: plot_state['frontal_im'].remove()
    if plot_state['overhead_im'] is not None: plot_state['overhead_im'].remove()
    if plot_state['frontal_sem'] is not None: plot_state['frontal_sem'].remove()
    if plot_state['overhead_sem'] is not None: plot_state['overhead_sem'].remove()
    if plot_state['overhead_im_rect'] is not None: plot_state['overhead_im_rect'].remove()

    kwargs = {'overhead_lidar': overhead_lidar}
    purged = set()
    for plottable in plottables:
        # If the plottable exists and has things to be removed.
        if plottable:
            if plottable.removable():
                # For every thing in the plottable
                for key in plottable.plot_keys:
                    # If the object exists (has been plotted before), remove it.
                    if key in purged: continue
                    else: purged.add(key)
                    if plot_state.purge_key(key) == 'continue': continue
                    
            # Plot the plottable.
            plottable.plot(plot_state, axes=axes, **kwargs)

    # Center-crop the image
    if sensor_data is not None:
        fim = sensor_data.CameraRGB.data
        h, w, *_ = fim.shape
        diff = w - h
        assert(diff >= 0)
        # plot_state['frontal_im'] = axes[0,1].imshow(im)
        plot_state['frontal_im'] = axes[0,1].imshow(fim)
        
        if plot_state.plotconf.plot_overhead_im:
            # plot_state['overhead_im'] = axes[1,1].imshow(sensor_data.CameraRGBBEV.data)
            oim = sensor_data.CameraRGBBEV.data
            H,W,C = oim.shape
            extent = (0, 100, 100, 0)
            plot_state['overhead_im'] = axes[1,0].imshow(
                sensor_data.CameraRGBBEV.data,
                extent=extent,
                zorder=1)

    if plot_state.plotconf.plot_zoomed:
        axes[0, 0].axis(plot_state.plotconf.zoom_out_bounds)
        if not plot_state.plotconf.plot_overhead_im:
            axes[1, 0].axis(plot_state.plotconf.zoom_bounds)
        axes[1, 1].axis(plot_state.plotconf.zoom_bounds)
    
    for ax in axes.ravel(): ax.axis('off')
    plt.gcf().canvas.flush_events()

class PlotState(dict):
    @classu.member_initialize
    def __init__(self, plotconf): pass
    
    @staticmethod
    def from_plottables(plotconf, plottables):
        ps = PlotState(plotconf)
        state = {
            # 'overhead_lidar_im': None,
            'frontal_im': None,
            'overhead_im_rect': None,
            'overhead_sem': None,
            'overhead_im': None,
            'frontal_sem': None
        }
        for plottable in plottables:
            for key in plottable.plot_keys:
                state[key] = None
        ps.update(state)
        return ps

    def purge_key(self, key):
        if key not in self: return
        obj = self[key]
        if hasattr(obj, 'remove') and not isinstance(obj, list):
            obj.remove()
            return
        elif obj is None:
            return 'continue'
        else:
            for o in obj:
                try: o.remove()
                except ValueError as e: log.debug("ValueError! {}".format(e))
            return
        
class GenericPerFramePlottingData:
    @classu.member_initialize
    def __init__(self,
                 pilot,
                 hires,
                 measurement,
                 feed_dict,
                 waypoints_local,
                 waypoints_control,
                 waypoint_metadata,
                 transform_now,
                 current_obs,
                 control,
                 direction_string='empty'):
        pass

    def get_text_strings(self):
        strings = []
        m = self.measurement
        p = m.player_measurements
        t = p.transform
        loc = t.location
        acc = p.acceleration
        strings.append("Local frame={}.".format(self.current_obs.frame) + " Fwd speed={:.2f}m/s".format(p.forward_speed))
        strings.append("Pos=({:.2f},{:.2f},{:.2f}).".format(loc.x, loc.y, loc.z) + " Acc=({:.2f},{:.2f},{:.2f})".format(acc.x, acc.y, acc.z))
        strings.append("Light state={}. Turning={}".format(self.current_obs.traffic_light_state, self.current_obs.is_turning))
        strings.append("Stationary={}. Stuck={}".format(self.current_obs.is_stationary, self.current_obs.is_stuck))
        strings.append("Direction string={}".format(self.direction_string))
        
        if hasattr(self.current_obs, 'npc_trajectories'):
            npcs_local_tm0 = self.current_obs.unfilt_agent_positions_local[:, -1]
            if npcs_local_tm0.shape[0] == 0: log.error("No unfiltered agents???")
        return strings

class DIMPlottingData:
    @classu.member_initialize
    def __init__(self, plan, midlow_controller): pass
    
class PlottableManager:
    @classu.member_initialize
    def __init__(self, plotconf):
        self.base_plottables = []
        self.base_plottables.extend(
            [PlottableChosenWaypoint,
             PlottableLegend,
             PlottablePast,
             PlottablePlan,
             PlottablePlanQuality,
             PlottableTextBlock,
             PlottableDIMTextBlock,
             PlottableControlWaypoints,
             PlottablePolygon,
             PlottableMixtureWaypoints,
             PlottablePolygonSeed,
             PlottableControlRegion,
             PlottableRegionVertices,
             PlottableSegmentSet,
             PlottableDestination,
             PlottableRoute])

        self.p2i = dict([reversed(_) for _ in enumerate(self.base_plottables)])
        self.reset()

    def _i(self, o):
        """Stash the plottable in the plottables by its class"""
        self.plottables[self.p2i[o.__class__]] = o

    def update_zero(self, plot_data):
        self.update_from_observation(plot_data=plot_data)
        self.update_regions(plot_data)

    def update_regions(self, plot_data):
        # Assuming region control
        self._i(PlottableControlRegion(
            plot_data.waypoints_control, transform_now=plot_data.transform_now, transform_orig=plot_data.transform_now))
        self._i(PlottableRegionVertices(
            plot_data.waypoints_control, transform_now=plot_data.transform_now, transform_orig=plot_data.transform_now))
        self._i(PlottablePolygonSeed(
            plot_data.waypoints_local, transform_now=plot_data.transform_now, transform_orig=plot_data.transform_now))

    def update_from_observation(self, plot_data, text=False, plot_control_points=True, plot_regions=False):
        # These are current.
        self._i(PlottablePast(plot_data.current_obs.player_positions_local))
        if plot_regions:
            self.update_regions(plot_data)
        if text and self.plotconf.plot_text_block:
            self._i(PlottableTextBlock(plot_data.get_text_strings() + [format_control_string(plot_data.control)]))
        if hasattr(plot_data.current_obs, 'player_destination_local') and plot_data.current_obs.player_destination_local is not None:
            self._i(PlottableDestination(plot_data.current_obs.player_destination_local[None]))

    def update_dim(self, plot_data, dim_plotting_data):
        dim_plan = dim_plotting_data.plan
        transform_orig = dim_plotting_data.midlow_controller.transform_plan_start
        self.update_from_observation(plot_data, text=False, plot_control_points=False, plot_regions=False)

        if len(dim_plan.planner.goal_likelihood.current_waypoints) > 0:
            if dim_plan.planner.goal_likelihood.describe() == "SegmentSetIndicator":
                self._i(PlottableSegmentSet(
                    dim_plan.planner.goal_likelihood.current_waypoints, transform_now=plot_data.transform_now, transform_orig=transform_orig))
            elif dim_plan.planner.goal_likelihood.describe() == "RegionIndicator":
                self._i(PlottableControlRegion(
                    dim_plan.planner.goal_likelihood.current_waypoints,
                    transform_now=plot_data.transform_now, transform_orig=transform_orig))
                # self._i(PlottableRegionVertices(
                #     dim_plan.waypoint_metadata['pp_polygon'].copy(), transform_now=plot_data.transform_now, transform_orig=transform_orig))
                self._i(PlottableRoute(
                    dim_plan.waypoint_metadata['route_2d'].copy(), transform_now=plot_data.transform_now, transform_orig=transform_orig))
            else:
                self._i(PlottableMixtureWaypoints(
                    dim_plan.planner.goal_likelihood.current_waypoints, transform_now=plot_data.transform_now, transform_orig=transform_orig))
        self._i(PlottablePlan(dim_plan,
                              waypoints_local=plot_data.waypoints_local,
                              transform_orig=transform_orig,
                              transform_now=plot_data.transform_now))
        
        if self.plotconf.plot_text_block:
            self._i(PlottableDIMTextBlock(dim_plan, transform_orig, plot_data))
        if self.plotconf.plot_legend:
            self._i(PlottableLegend(plot_data))
        if self.plotconf.plot_chosen_waypoint and dim_plan.planner.goal_likelihood.describe() != "RegionIndicator":
            self._i(PlottableChosenWaypoint(dim_plan.chosen_waypoint[None], transform_orig=transform_orig, transform_now=plot_data.transform_now))
        self._i(PlottableDestination(plot_data.current_obs.player_destination_local[None]))

    def reset(self):
        self.plottables = [None] * len(self.base_plottables)

class Plottable:
    """An object that holds data to be plotted"""
    def plot(self, plot_state, axes, *args, **kwargs): raise NotImplementedError

    def removable(self): raise NotImplementedError

class Nonremovable:
    """Indicates object has nothing to remove between plotting (e.g. no images)"""
    def removable(self): return False

class Removable:
    """Indicates object has something to remove between plotting (e.g. images)"""
    def removable(self): return True

class PlottableLegend(Removable):
    plot_keys = ['legend0', 'legend1', 'legend2', 'legend3']

    @classu.member_initialize
    def __init__(self, plot_data):
        pass

    def plot(self, plot_state, axes, **kwargs):
        markersize = 12
        p0 = mlines.Line2D([],[],
                           linewidth=2,
                           marker=PlottableMixtureWaypoints.marker,
                           markersize=markersize,
                           color=PlottableMixtureWaypoints.color,
                           alpha=0.4)
        p1 = mlines.Line2D([],[],linewidth=0,
                           marker='o',
                           markersize=markersize,
                           color=plot_state.plotconf.cw_colors[0],
                           markeredgecolor='k',markeredgewidth=2,alpha=0.4)
        p2 = mlines.Line2D([],[],
                           linewidth=0,
                           marker=PlottablePast.marker,
                           markersize=markersize,
                           color=PlottablePast.color,
                           alpha=0.5)

        fontsize = 12 if self.plot_data.hires else 5
        plot_state[self.plot_keys[0]] = axes[0,0].legend(
            (p0,p1,p2), ('Goal Centers', 'Plan Points','Past Positions'),fontsize=fontsize,loc='upper left', bbox_to_anchor=(0,1))
        plot_state[self.plot_keys[2]] = axes[1,0].legend(
            (p0,p1,p2), ('Goal Centers', 'Plan Points','Past Positions'),fontsize=fontsize,loc='upper left', bbox_to_anchor=(0,1))
        plot_state[self.plot_keys[3]] = axes[1,1].legend(
            (p0,p1,p2), ('Goal Centers', 'Plan Points','Past Positions'),fontsize=fontsize,loc='upper left', bbox_to_anchor=(0,1))

def cmap_discretize(cmap, N):
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

class PlottableWaypoints(Nonremovable, Plottable):    
    @classu.member_initialize
    def __init__(self, waypoints, transform_orig=None, transform_now=None, axes=None):
        """

        :param waypoints: (N, 2) robot coordinates
        :param transform_orig: 
        :param transform_now: 
        :returns: 
        :rtype: 

        """
        self.waypoints = transform(waypoints, transform_now=transform_now, transform_orig=transform_orig)
    
    def plot(self, plot_state, axes, **kwargs):
        if self.waypoints is not None:
            # (N, 2)
            waypoints_grid = traj2grid(self.waypoints[:,0], self.waypoints[:,1])
            # Plot as single trajectory (1, N, 2)
            if self.axes is not None:
                axes = self.axes

            axes = np.asarray([axes[0,0], axes[1,0], axes[1,1]])
            for axidx, ax in enumerate(axes.ravel()):
                plot_state[self.plot_keys[axidx]] = plot_trajectories(
                    # Use the "plot_kwargs_key" of the inherited class to retrieve the kwargs for the object.
                    waypoints_grid[None], ax=ax, lines=plot_state[self.plot_keys[axidx]], **getattr(plot_state.plotconf, self.plot_kwargs_key))

    @property
    def plot_kwargs_key(self):
        raise ValueError("Preventing instantiation of base class")

class PlottableWaypointsDefault(Nonremovable, Plottable):
    plot_keys = ['waypoints0', 'waypoints1', 'waypoints2', 'waypoints3']
    plot_kwargs_key = 'waypoints'

class PlottablePast(PlottableWaypoints):
    plot_keys = ['past_waypoints0', 'past_waypoints1', 'past_waypoints2', 'past_waypoints3']
    plot_kwargs_key = 'past'

class PlottableRegionVertices(PlottableWaypoints):
    plot_keys = ['region_waypoints0', 'region_waypoints1', 'region_waypoints2', 'region_waypoints3']
    plot_kwargs_key = 'region_vertices'
    
class PlottablePolygonSeed(PlottableWaypoints):
    plot_keys = ['seed_waypoints0', 'seed_waypoints1', 'seed_waypoints2', 'seed_waypoints3']
    plot_kwargs_key = 'polygon_seed'
    marker = 'd'
    color = 'yellow'

class PlottableControlWaypoints(PlottableWaypoints):
    plot_keys = ['control_waypoints0', 'control_waypoints1', 'control_waypoints2', 'control_waypoints3']
    plot_kwargs_key = 'control_waypoints'

class PlottableMixtureWaypoints(PlottableWaypoints):
    plot_keys = ['mixture_waypoints0', 'mixture_waypoints1', 'mixture_waypoints2', 'mixture_waypoints3']
    plot_kwargs_key = 'mixture_waypoints'

class PlottableSegmentSet(PlottableWaypoints):
    plot_keys = ['segmentset_waypoints0', 'segmentset_waypoints1', 'segmentset_waypoints2', 'segmentset_waypoints3']
    plot_kwargs_key = 'segment_set'

class PlottableRoute(PlottableWaypoints):
    plot_keys = ['route_waypoints0', 'route_waypoints1', 'route_waypoints2', 'route_waypoints3']
    plot_kwargs_key = 'route'

class PlottableDestination(PlottableWaypoints):
    plot_keys = ['dest_0', 'dest_1', 'dest_2', 'dest_3']
    plot_kwargs_key = 'destination'

class PlottableChosenWaypoint(PlottableWaypoints):
    plot_keys = ['chosen_waypoints0', 'chosen_waypoints1', 'chosen_waypoints2', 'chosen_waypoints3']
    plot_kwargs_key = 'chosen'

class PlottablePolygon(Removable, Plottable):
    plot_keys = ['polygon0', 'route1', 'polygon2', 'route3', 'polygon3', 'route1', 'polygon4', 'route3']
    plot_kwargs_key = 'pp_polygon'
    color = 'green'
    alpha = 0.05
    marker = 'x'
    linewidth = 2
    zorder = 50
    
    @classu.member_initialize
    def __init__(self, polygon, plot_data, transform_orig=None, transform_now=None, axes=None):
        """

        :param polygon: (N, 2) robot coordinates
        :param transform_orig: 
        :param transform_now: 
        :returns: 
        :rtype: 

        """
        self.polygon = transform(polygon, transform_now=transform_now, transform_orig=transform_orig)
    
    def plot(self, plot_state, axes, **kwargs):
        if self.polygon is not None:
            # (N, 2)
            polygon_grid = traj2grid(self.polygon[:,0], self.polygon[:,1])
            # Plot as single trajectory (1, N, 2)
            if self.axes is not None:
                axes = self.axes

            axes = np.asarray([axes[0,0], axes[1,0], axes[1,1]])
            for axidx, ax in enumerate(axes.ravel()):
                polygon = matplotlib.patches.Polygon(polygon_grid, **plot_state.plotconf.polygon)
                ax.add_patch(polygon)
                plot_state[self.plot_keys[axidx]] = polygon
    
class PlottableControlRegion(PlottablePolygon):
    plot_keys = ['polygon0_cr', 'route1_cr', 'polygon2_cr', 'route3_cr', 'polygon3_cr', 'route1_cr', 'polygon4_cr', 'route3_cr']
    @classu.member_initialize
    def __init__(self, polygon, transform_orig=None, transform_now=None, axes=None):    
        self.polygon = transform(polygon, transform_now=transform_now, transform_orig=transform_orig)

    def plot(self, plot_state, axes, **kwargs):
        # (N, 2)
        polygon_grid = traj2grid(self.polygon[:,0], self.polygon[:,1])
        
        # Plot as single trajectory (1, N, 2)
        if self.axes is not None:
            axes = self.axes

        axes = np.asarray([axes[0,0], axes[1,0], axes[1,1]])
        # pdb.set_trace()
        for axidx, ax in enumerate(axes.ravel()):
            polygon = matplotlib.patches.Polygon(polygon_grid, facecolor=plot_state.plotconf.colors[1], edgecolor='k', alpha=0.4, zorder=49)
            ax.add_patch(polygon)
            plot_state[self.plot_keys[2*axidx]] = polygon

class PlottablePlan(Nonremovable, Plottable):
    plot_keys = ['overhead_wp_target', 'overhead_planned_line0',
                 'foo',
                 'bar',
                 'thesenamesaremeaningless',
                 'baz',
                 'overhead_planned_line1',
                 'overhead_wps', 'overhead_planned_line2',
                 'overhead_chosen_target0',
                 'overhead_chosen_target1',
                 'overhead_chosen_target2',
                 'aa',
                 'bb',
                 'cc',
                 'dd',
                 'ee',
                 'ff']
    
    @classu.member_initialize
    def __init__(self, plan, waypoints_local, transform_orig, transform_now, top_k_max=12):
        # Normal alphas for the plans we'll plot, in plan ordering (high at beginning, low at end)
        # self.alphas = np.linspace(0.5, .1, plan.n_useful)
        self.alphas = np.linspace(0.5, .4, plan.n_useful)
        self.alphas_topk = np.linspace(0.5, .4, top_k_max)

        # From bright to dark for the useful plans.
        self.colors = cm.coolwarm(np.linspace(1., 0., plan.size))
        
        # Transform from local at t -> world -> local at t + tau
        # self.composed = self.transform_now.inverse() * self.transform_orig
        self.composed = self.transform_now * self.transform_orig.inverse()
        self.alpha = 0.2

    def plot(self, plot_state, axes, **kwargs):
        # Waypoints in current frame
        # waypoints_local = self.waypoints_local

        # Points in a possibly old frame
        x_planned = self.plan.planned_trajectories_flat
        waypoint_targets = self.plan.waypoint_target

        target_axes = [axes[0,0], axes[1,0]]
        if plot_state.plotconf.plot_zoomed:
            target_axes.append(axes[1,1])

        # Execute temporal transforms.
        waypoint_targets = self.composed.transform_points(waypoint_targets)
        # Add z coordinates (they're zero in the local frame of the original transform)
        x_planned_3d = np.concatenate((x_planned, np.zeros_like(x_planned[..., [0]])), axis=-1)
        x_planned_tx = (
            self.composed.transform_points(x_planned_3d.reshape((-1, 3))).reshape(x_planned_3d.shape))
        # Clip irrelevant z.
        x_planned = x_planned_tx[..., :2]
        
        # Execute traj to grid coordinate transforms.
        waypoint_targets_grid = traj2grid(waypoint_targets[..., 0], waypoint_targets[..., 1])
                
        # x_planned_world = self.planning_transform.transform_points(x_planned)
        x_planned_grid = traj2grid(x_planned[...,0], x_planned[...,1])

        if waypoint_targets_grid.ndim == 2:
            waypoint_targets_grid = waypoint_targets_grid[None]

        def plot_planned_line(key, ax, alpha_mx=1., top_k=12, plot_useless=False):
            # Plot the plan.
            plot_state[key] = plot_trajectories(x_planned_grid, ax=ax, lines=plot_state[key], **plot_state.plotconf.plan)
            planlines = plot_state[key]
            if top_k == self.top_k_max or top_k == 1: alphas = self.alphas_topk
            else: alphas = self.alphas

            # Iterate plotted plans from best to worst.
            for planline_idx in range(len(planlines)):
                po = self.plan.plan_ordering_flat[planline_idx]
                # Adjust the next best plan.
                planline = plot_state[key][po]                    

                useless = not self.plan.useful_plans_mask_flat[planline_idx]
                hide_useless = useless and not plot_useless and planline_idx > 0
                if hide_useless or planline_idx >= top_k:
                    planline.set_alpha(0.0)
                    continue
                else:
                    # Set alpha 
                    planline.set_alpha(alphas[planline_idx])

                    # Set the color corresponding to the color ordering
                    planline.set_color(self.colors[planline_idx])

                    # Make z's higher for the better plans.
                    planline.set_zorder(2*len(self.plan.plan_ordering_flat) - planline_idx)

                if planline_idx == 0:
                    planline.set_alpha(self.alpha)
            return plot_state[key]

        pk_idx = 0
        for ta_idx, ta in enumerate(target_axes):
            top_k = self.top_k_max if ta_idx != 1 else 1
            plot_useless = ta_idx != 1
            _ = plot_planned_line(self.plot_keys[pk_idx], ax=ta, top_k=top_k, plot_useless=plot_useless)
            pk_idx += 1

class PlottableTextBlock(Removable, Plottable):
    plot_keys = ['text_str']
    xax = 0
    yax = 1

    @classu.member_initialize
    def __init__(self, text_strings): pass
    
    def plot(self, plot_state, axes, **kwargs):
        xloc = min(plot_state.plotconf.zoom_bounds[0:2]) + 1
        yloc = min(plot_state.plotconf.zoom_bounds[2:4]) + 1
        joint_text = '\n'.join(self.text_strings)
        bbox = dict(facecolor='w', edgecolor='k', alpha=0.8)
        pconf = plot_state.plotconf
        fontsize = pconf.hires_dim_font_size if pconf.hires_plot else pconf.lores_dim_font_size
        plot_state[self.plot_keys[0]] = axes[self.yax, self.xax].text(
            xloc, yloc, s=joint_text, zorder=10, ha='left', va='top', fontsize=fontsize, bbox=bbox)

class PlottableDIMTextBlock(PlottableTextBlock):
    plot_keys = ['dimtext']
    
    def __init__(self, dim_plan, transform_orig, plot_data):
        text_strings = []
        steps = dim_plan.steps
        m = plot_data.current_obs.measurements[-1]
        p = m.player_measurements
        f = p.forward_speed
        
        pd_text_strings = plot_data.get_text_strings()
        text_strings.extend(pd_text_strings)
        text_strings.append(format_control_string(plot_data.control))
        text_strings.append("Target fwd pos={:.2f}, vel={:.2f}m/s. Cur fwd={:.2f}m/s. Err={:.2f}".format(
            dim_plan.current_target_forward_displacement, dim_plan.current_target_forward_speed, f, dim_plan.current_forward_speed_error))
        text_strings.append(
            "log_posterior={:.2f}, log q(x)={:.2f}, log p(goal|x)={:.2f}".format(
                dim_plan.best_plan_log_posterior, dim_plan.best_plan_log_prior, dim_plan.best_plan_loggoal))
        text_strings.append("Plan count={}. Plan step={}. Opt steps={:.2f}".format(dim_plan.total_plan_count, dim_plan.plan_step, steps))
        text_strings.append("p(goal|x)={}".format(dim_plan.planner.goal_likelihood.describe()))
        super().__init__(text_strings)

class PlottableText(Removable, Plottable):
    plot_keys = ['text_str']
    xax = 0
    yax = 0
    xloc = 50
    yloc = 10

    @classu.member_initialize
    def __init__(self, text):
        pass

    def plot(self, plot_state, axes, **kwargs):
        plot_state[self.plot_keys[0]] = axes[self.xax,self.yax].text(self.xloc, self.yloc, s=self.text,
                                                                     zorder=10)

class PlottablePlanQuality(PlottableText):
    xax = 1
    yax = 0
    xloc = 50
    yloc = 20
    
    def __init__(self, log_prob, steps):
        text = 'Log prob: {:.3f}. Steps: {}'.format(log_prob, steps)
        super().__init__(text)        

def create_agent_figure(figsize=(6,6), image_size=None, n_wide=2, dpi=100., remove_second_row=False):
    fig, axes = plt.subplots(2, n_wide, figsize=figsize, dpi=dpi)
    # fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    for ax in axes.ravel(): ax.axis('off')
    plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

    # fig.suptitle('CARLA plots')
    plt.pause(1e-4)

    for ax in axes.ravel(): ax.figure.canvas.draw()
        
    if remove_second_row:
        log.debug("removing second row of plot")
        ax00_pos = axes[0,0].get_position()
        ax10_pos = axes[1,0].get_position()
        log.info("Original (0,0) and (1,0) positions: {}, {}".format(ax00_pos, ax10_pos))
        axes[1,0].set_position(ax00_pos)
        axes[0,0].set_position(ax10_pos)
        for ax in axes.ravel(): ax.figure.canvas.draw()
        log.info("New (0,0) and (1,0) positions: {}, {}".format(axes[0,0].get_position(), axes[1,0].get_position()))
    fig.canvas.draw()
    return fig, axes

def plot_trajectories(trajectories, ax, alphas=None, lines=None, **kwargs):
    """

    :param trajectories: (K, T, d) Trajectories in ***GRID / OVERHEAD IMAGE FRAME*** (not world frame).
    :param ax: plt.AxesSubplot
    :returns: 
    :rtype: 

    """
    assert(trajectories.ndim == 3)
    assert(trajectories.shape[-1] == 2)
    kwa = {'marker': '.'}
    kwa.update(kwargs)

    if lines is not None:
        for i, (line, tx, ty) in enumerate(zip(lines, trajectories[..., 0], trajectories[..., 1])):
            log.debug("updating new line")
            # Apparently this causes things to complain
            # line.set_xdata(tx)
            line._xorig = tx
            line._invalidx = True
            line.set_ydata(ty)
            line.set_alpha(kwargs.get('alpha', 1.0))
    else:
        lines = ax.plot(trajectories.T[0], trajectories.T[1], **kwa)
    if alphas is not None:
        for lidx, line in enumerate(lines):
            line.set_alpha(alphas[lidx])
    return lines
        
def update_pilots_plot(pilot,
                       plottable_manager,
                       generic_plotting_data,
                       approach_plotting_data):    
    if pilot == 'dim':
        plottable_manager.update_dim(plot_data=generic_plotting_data, dim_plotting_data=approach_plotting_data)
    elif pilot == 'auto':
        plottable_manager.update_from_observation(plot_data=generic_plotting_data, text=True, plot_regions=False)
    else:
        # N.B. 'auto' pilot is not in the conditional block.
        raise ValueError("Unknown pilot '{}'".format(pilot))

def transform(points, transform_now, transform_orig=None):
    assert(points.ndim == 2)
    if points.shape[1] == 2:
        # 2d -> 3d
        points = np.concatenate((points, np.zeros_like(points[..., [0]])), axis=-1)
    if transform_orig is not None:
        composed = transform_now * transform_orig.inverse()
        return composed.transform_points(points)
    else:
        return points
    
def get_unique_output_dir(parent, prefix='movie'):
    """Find a suitable directory to store things into"""
    out_fmt = lambda counter: '{}/{}{:03d}'.format(parent, prefix, counter)

    for i in range(100):
        output_directory = out_fmt(i)
        if not os.path.isdir(output_directory):
            print("Found unmade dir: '{}'".format(output_directory))
            break
    return output_directory

def get_settings(directory):
    """ Load the camera settings from an episode directory (child directory of parent with settings)"""
    return dill.load(open(os.path.dirname(os.path.realpath(directory))  + '/settings.dill', 'rb'))

def plot_item_legend(n=6):
    plt.figure(figsize=(6,6), dpi=180)
    plt.rcParams['font.family'] = 'serif'
    colors = plt.get_cmap('coolwarm')(np.linspace(0,1, n))

    plt.axis([0,2,0,2])

    plt.scatter(0,0, color='green', marker='x', s=60, label='Goal Centers')
    plt.scatter(0,0, color='orangered', marker='x', s=60, label='Past positions')
    plt.scatter(0,0, marker='o', facecolor=colors[-1], edgecolor='k', s=60, label='Chosen plan positions')
    # plt.scatter(0,0, color=colors[n-1], marker='s', s=60, label='Chosen goal')
    # plt.scatter(0,0, color='gold', marker='+', s=60, label='Controller setpoint', linewidth=1)

    ###
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.arange(len(handles))[::-1]
    order = np.roll(order, -3)

    legfig = plt.figure(figsize=(2,1), dpi=180)
    legfig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    legfig.savefig('plotlegend.png', dpi=180)

def plot_plan_colormap():
    
    nw = 256
    rat = 2 ** 3
    nh = nw // rat
    gradient = np.linspace(0, 1, nw)

    gradient = np.stack((gradient,)*nh, axis=0) 
    w = 9
    rat = nh / nw
    plt.figure(figsize=(w, w // rat))
    #plt.axis('off')
    plt.imshow(gradient, cmap=plt.get_cmap('coolwarm'))
    plt.xlabel("Plan preference (low to high)", fontsize=40, fontname='serif')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # plt.gcf().get_axis
    #plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig("test.png", bbox_inches='tight')

def format_control_string(control):
    return "Steer={:.2f}, Throttle={:.2f}, Brake={:.2f}".format(control.steer, control.throttle, control.brake)

def traj2grid(trajx, trajy, **kwargs):
    # TODO this global function is a hack.
    if len(kwargs): log.warning("Extra traj2grid kwargs ignored!")
    return np.stack((trajx*2 + 50, trajy*2 + 50), axis=-1)

def full_extent(axes, pad=0.0):
    from matplotlib.transforms import Bbox

    items = []
    for ax in axes: items += [ax]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)
