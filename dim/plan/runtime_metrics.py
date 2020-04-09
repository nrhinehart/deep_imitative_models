
from collections import defaultdict
import logging
import os
import numpy as np
import scipy.stats
import precog.utils.class_util as classu

log = logging.getLogger(os.path.basename(__file__))

class FrameMetrics:
    keys = ['collision_vehicles', 'collision_pedestrians', 'collision_other',
            'intersection_otherlane', 'intersection_offroad']
    
    def __init__(self, measurements, nonmeasurement_metrics={}):
        """Extract keys from the measurements that describe the player's current performance"""
        pm = measurements.player_measurements
        for key in self.keys:
            # Retrieve the value from the measurements.
            try:
                setattr(self, key, getattr(pm, key))
            except AttributeError:
                # Likely due to a nonmeasurement metric key we added before.
                pass

        # These are things that are quantified outside the normal measurement metrics, e.g. intersections and red lights.
        for nm_key, nm_value in nonmeasurement_metrics.items():
            # Get the attributes from the provided dictionary.
            setattr(self, nm_key, nm_value)
            # If the attribute doesn't exist in the keys, add it
            if nm_key not in FrameMetrics.keys:
                FrameMetrics.keys.append(nm_key)

        self.keys = FrameMetrics.keys
        self.rstr = ',\n\t'.join(['{}: {:.3f}'.format(k, getattr(self, k)) for k in self.keys])

    def __repr__(self):
        return "{}:\n\t{}".format(self.__class__.__name__, self.rstr)

    @staticmethod
    def quantify_table(table, keys):
        """Quantify means, stds, and nnz for each key"""
        assert(table.ndim == 2)
        jkeys = list(enumerate(keys))
        # if table.shape[0] > jkeys:
        nnzs = {k: np.count_nonzero(table[:, j]) for j,k in jkeys}
        means = {k: np.mean(table[:, j]) for j,k in jkeys}
        stds = {k: np.std(table[:, j]) for j,k in jkeys}
        maxs = {k: np.max(table[:, j]) for j,k in jkeys}
        sums = {k: np.sum(table[:, j]) for j,k in jkeys}
        return {'nnzs': nnzs, 'means': means, 'stds': stds, 'maxs': maxs, 'sums': sums}
                               
class EpisodeMetrics:
    def __init__(self, episode_params):
        self.episode_params = episode_params
        self.frame_metrics = []
        self.success = False
        self.concluded = False
        self.passenger_comfort = defaultdict(list)
        self.passenger_comfort_percentiles = None
        self.intersection_count = 0
        self.red_light_violations = 0

    def update(self, measurements, extras={}):
        """Track new measurements"""
        self.frame_metrics.append(FrameMetrics(measurements, extras))
        self.intersection_count = self.frame_metrics[-1].intersection_count
        self.red_light_violations = self.frame_metrics[-1].red_light_violations

    def update_passenger_comfort(self, obs):
        Tp = len(obs.measurements)
        Hz = 10
        states = np.array([obs.get_sfa(x)[0] for x in range(-Tp, 0)])  # (7,2)
        speeds = (states[1:] - states[:-1]) * Hz  # (6,)
        accels = (speeds[1:] - speeds[:-1]) * Hz  # (5,)
        jerks = (accels[1:] - accels[:-1]) * Hz  # (4,)
        snaps = (jerks[1:] - jerks[:-1]) * Hz  # (3,)
        cracks = (snaps[1:] - snaps[:-1]) * Hz  # (2,)
        pops = (cracks[1:] - cracks[:-1]) * Hz  # (1,)
        speeds = np.linalg.norm(speeds[-1])
        accels = np.linalg.norm(accels[-1])
        try: jerks = np.linalg.norm(jerks[-1])
        except IndexError: jerks = None
        try: snaps = np.linalg.norm(snaps[-1])
        except IndexError: snaps = None
        try: cracks = np.linalg.norm(cracks[-1])
        except IndexError: cracks = None
        try: pops = np.linalg.norm(pops[-1])
        except IndexError: pops = None
        info = {'speed': speeds, 'acceleration': accels, 'jerk': jerks,
                'snap': snaps, 'crackle': cracks, 'pop': pops}
        for key, value in info.items():
            self.passenger_comfort[key].append(value)

    def conclude(self, success=False, summary=''):
        assert(self.concluded is False)
        self.success = success
        self.summary = summary
        
        self.passenger_comfort_percentiles = {}
        # for key, value in self.passenger_comfort.items():
        #     try:
        #         pp = np.percentile(value, range(101))
        #     except TypeError:
        #         pp = -999
        #     self.passenger_comfort_percentiles[key] = pp
        self.concluded = True

    def quantify(self):
        """Quantify performance of this episode"""
        keys = self.frame_metrics[0].keys
        # keys = FrameMetrics.keys
        self.table = np.zeros((len(self.frame_metrics), len(keys)), dtype=np.float32)
        jkeys = list(enumerate(keys))

        for i, fm in enumerate(self.frame_metrics):
            for j, key in jkeys:
                self.table[i, j] = getattr(fm, key)

        self.intersection_count = getattr(self, 'intersection_count', 0)
        self.red_light_violations = getattr(self, 'red_light_violations', 0)

        # self.table_stats = FrameMetrics.quantify_table(self.table)
        return self.table


# TODO HACK
CLIP = 25
    
class MultiepisodeMetrics:
    @staticmethod
    def from_episode_metrics(epmets):
        mm = MultiepisodeMetrics(None)
        mm.all_episode_metrics = epmets
        mm.quantify()
        return mm
    
    @classu.member_initialize
    def __init__(self, carla_agent_args):
        self.n_successes = 0
        self.n_failures = 0
        self.n_total = 0
        self.episode_to_observed_frame_count = {}
        self.all_episode_metrics = []
        # Track whether all metrics have been quantified (now we have one, unconcluded, unquantified)
        self.quantified = False

    def update(self, measurements, extras={}):
        """Track new measurements"""
        self.all_episode_metrics[-1].update(measurements, extras)

    def update_passenger_comfort(self, obs):
        self.all_episode_metrics[-1].update_passenger_comfort(obs)

    def get_comfort_metrics(self):
        data = defaultdict(list)
        for episode_metrics in self.all_episode_metrics[:CLIP]:
            if getattr(episode_metrics, 'ignore', False):
                # print("ignoring metrics at index {}".format(self.all_episode_metrics.index(episode_metrics)))
                continue
            for key, value in episode_metrics.passenger_comfort.items():
                data[key] += value
        return data

    def plot_passenger_comfort(self):
        data = self.get_comfort_metrics()
        keys = data.keys()

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3)
        axxys = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for i in range(len(keys)):
            key = keys[i]
            x, y = axxys[i]
            axes[x, y].set_title(key)
            axes[x, y].hist(data[key], bins=20)

        # print to screen expectations
        print('')
        print('carla metrics')
        print(self)
        print('')
        print('Passenger comfort 99 percentile:')
        print(self.format_comfort_metrics(data, tex=False))
        print()
        print('')

        plt.show()

    def format_comfort_metrics(self, data, tex=False, percentile=99, sep=','):
        ss = ""
        if tex:  # TODO: fix
            sep = '&'
            ss = ('{:.3f} {{0}} '.format(np.percentile(data['acceleration'], percentile)) +
                  '{:.3f} {{0}} '.format(np.percentile(data['jerk'], percentile)) +
                  '{:.2f} {{0}} '.format(np.percentile(data['snap'], percentile)) +
                  '{:.1f} {{0}} '.format(np.percentile(data['crackle'], percentile)) +
                  '{:.0f}'.format(np.percentile(data['pop'], percentile))).format(sep)
        else:
            for key, value in data.items():
                if key == 'speed' and isinstance(value[0], np.ndarray):
                    # this was because of an earlier error in some of our metric files
                    value = np.reshape(value, -1)
                value = [_ for _ in value if _]  # remove potential None's
                if value:
                    ss += '\t\t{}: {:.3f}\n'.format(key, np.percentile(value, percentile))
        return ss

    def begin_new_episode(self, episode_params):
        self.all_episode_metrics.append(EpisodeMetrics(episode_params))

    def conclude_episode(self, success=False, summary=''):
        """Conclude the current episode with success or failure"""
        self.all_episode_metrics[-1].conclude(success, summary=summary)
        self.quantified = False

    def print_conclusion_statii(self):
        for eidx, em in enumerate(self.all_episode_metrics):
            # Print conclusions as ints so less visually confusing
            if hasattr(em, 'episode_params'):
                true_index = em.episode_params.episode
            else:
                true_index = None
            skipped = eidx >= CLIP
            print("Episode {} (true_index={}). Success={}. Metrics-skipped={}, Concluded={}. Summary='{}'".format(
                eidx, true_index, int(em.success), skipped, em.concluded, getattr(em, 'summary', '')))

    def quantify(self):
        """Quantify performance of all episodes"""
        successes = []
        tables = []
        print("CLIPPING TO {} max episodes (have {})".format(CLIP, len(self.all_episode_metrics)))
        for em in self.all_episode_metrics[:CLIP]:
            # Only quantify concluded episodes.
            if not em.concluded:
                print("Skipping unconcluded episode!")
                continue
            if getattr(em, 'ignore', False):
                print("Skipping for some reason?")
                continue
            tables.append(em.quantify())
            successes.append(em.success)

        self.n_total = len(successes)
        self.n_successes = sum(successes)
        self.n_failures = self.n_total - self.n_successes

        if len(tables) == 0:
            log.warning("No episodes completed yet to quantify...")
            self.joint_table_stats = {'means': {}, 'stds': {}, 'nnzs': {}}
            self.success_percentage = 0.0
        else:
            self.joint_table = np.concatenate(tables, axis=0)
            # These four members quantify the overall performance.
            self.joint_table_stats = FrameMetrics.quantify_table(self.joint_table,
                                                                 keys=self.all_episode_metrics[0].frame_metrics[0].keys)
            self.success_percentage = np.mean(successes)
            
        self.quantified = True

    def get_mean_stats_list(self):
        # Sort that jawn.
        mean_stats = sorted(["{}={}".format(k, str(v)) for k, v in self.joint_table_stats['means'].items()])
        return mean_stats

    def __repr__(self):
        if not self.quantified: self.quantify()
        mean_stats = "\n\t\t".join(k+": "+str(v) for k, v in self.joint_table_stats['means'].items())
        ss = ("MultiepisodeMetrics:\n" +
              "\tn_total={}\n" +
              "\tn_successes={}\n" +
              "\tn_failures={}\n" +
              "\tsuccess_percentage={:.3f}\n").format(self.n_total,
                                                      self.n_successes,
                                                      self.n_failures,
                                                      self.success_percentage)
        ss += "\tMean_time (seconds)={}\n".format(self.compute_average_time_to_complete_successful_episodes())
        ss += "\tMean stats:\n\t\t{}\n".format(mean_stats)
        ss += "\tComfort: \n{}\n".format(self.format_comfort_metrics(self.get_comfort_metrics()))
        ss += self.format_intersection_metrics()
        # except Exception as e:
        #     log.error(e)
        #     raise e
        # return "failedcomfortprint"
        return ss

    def format_table(self, sep=','):
        # print(self.all_episode_metrics[0].frame_metrics[0].keys)

        header_str = 'Successes {0} Collision Impulse {0} Wrong Lane {0} Off road {0} Time {0} Accel {0} Jerk {0} Snap {0} Crackle {0} Pop'.format(sep)
        try:
            mainstr = ("{} / {} {{0}} ".format(self.n_successes, self.n_total) +
                       "{:.3f} {{0}} ".format(self.joint_table_stats['means']['collision_other']) +
                       "{:.3f} {{0}} ".format(100 * self.joint_table_stats['means']['intersection_otherlane']) +
                       "{:.3f} {{0}} ".format(100 * self.joint_table_stats['means']['intersection_offroad']) +
                       "{:.3f} {{0}} ".format(self.compute_average_time_to_complete_successful_episodes())).format(sep)
        except KeyError as e:
            mainstr = 'format_table() KeyError: {}'.format(e)
        
        cstr = self.format_comfort_metrics(data=self.get_comfort_metrics(), tex=False)
        extra = ''
        try:
            extra += "{:.2f}\\% {{}} ".format(100 * self.joint_table_stats['means']['collision_pothole']).format(sep)
            extra += "{0}/{1} {{}} ".format(*compute_pothole_totals(self.all_episode_metrics[:CLIP])).format(sep)
            header_str += ' {} Potholes '.format(sep)
            try:
                extra += self.format_intersection_metrics()
            except Exception as e:
                print(e)
        except KeyError as e:
            print(e)
        finally:
            header_str += '\n'
        return header_str + mainstr + cstr + extra

    def format_intersection_metrics(self):
        intersections, red_light_violations, vio_perc = get_red_light(self)
        ss = "\tTotal intersections={}\n".format(intersections)
        ss += "\tTotal red light violations={}\n".format(red_light_violations)
        ss += "\tRed light violation percentage={:.2f}%\n".format(vio_perc)
        return ss

    def compute_average_time_to_complete_successful_episodes(self):
        if self.n_successes == 0:
            return np.inf
        time = 0.
        Hz = 10
        for episode in self.all_episode_metrics[:CLIP]:
            if episode.success:
                time += len(episode.frame_metrics)
        time /= self.n_successes
        time /= Hz
        return time

def get_red_light(mm):
    self = mm
    intersections = 0
    red_light_violations = 0
    for ep in self.all_episode_metrics[:CLIP]:
        intersections += ep.intersection_count
        red_light_violations += ep.red_light_violations
    vio_perc = 100 * red_light_violations / (intersections + np.finfo(np.float32).eps)
    return intersections, red_light_violations, vio_perc
    

def print_tex(mm, sep='&'):
    self = mm
    intersections, red_light_violations, vio_perc = get_red_light(self)
    # header_str = 'Successes {0} Collision Impulse {0} Wrong Lane {0} Off road {0} Time {0} Accel {0} Jerk {0} Snap {0} Crackle {0} Pop'.format(sep)
    header_str = 'Success  & Ran Red Light & Success & Collision Impulse & Wrong lane & Off road & Accel & Jerk & Snap & Crackle & Pop'
    try:
        mainstr = ("{}\\% & ".format(int(round(100 * self.n_successes / self.n_total))) +
                   '{:.3f}\\% & '.format(vio_perc) +
                   "{}\\% & ".format(int(round(100 * self.n_successes / self.n_total))) +
                   "{:.3f} & ".format(self.joint_table_stats['means'].get('collision_other', 'ERROR')) +
                   "{:.3f}\\% & ".format(100 * self.joint_table_stats['means'].get('intersection_otherlane', 'ERROR')) +
                   "{:.3f}\\% & ".format(100 * self.joint_table_stats['means'].get('intersection_offroad', 'ERROR')))
                   # "{:.3f}\\% & ".format(self.compute_average_time_to_complete_successful_episodes()))
    except KeyError as e:
        mainstr = 'format_table() KeyError: {}'.format(e)

    cstr = self.format_comfort_metrics(data=self.get_comfort_metrics(), tex=True)
    extra = ''
    try:
        extra += "{:.2f}\\% {{}} ".format(100 * self.joint_table_stats['means']['collision_pothole']).format(sep)
        extra += "{0}/{1} {{}} ".format(*compute_pothole_totals(self.all_episode_metrics[:CLIP])).format(sep)
        header_str += ' {} Potholes '.format(sep)
        try:
            extra += self.format_intersection_metrics()
        except Exception as e:
            print(e)
    except KeyError as e:
        print(e)
    finally:
        header_str += '\n'
    print(header_str + mainstr + cstr)# + extra)

def print_joint_tex(mms, sep='&'):

    # header_str = 'Successes {0} Collision Impulse {0} Wrong Lane {0} Off road {0} Time {0} Accel {0} Jerk {0} Snap {0} Crackle {0} Pop'.format(sep)
    header_str = 'Success  & Ran Red Light  & Wrong lane & Off road'
    all_numbers = []
    for self in mms:
        intersections, red_light_violations, vio_perc = get_red_light(self)
        numbers = [100 * self.n_successes / self.n_total,
                   vio_perc,
                   100 * self.joint_table_stats['means'].get('intersection_otherlane', 'ERROR'),
                   100 * self.joint_table_stats['means'].get('intersection_offroad', 'ERROR')]
        all_numbers.append(numbers)
    all_numbers = np.asarray(all_numbers)
    # print(all_numbers.shape)
    means = np.mean(all_numbers, axis=0)
    stds = scipy.stats.sem(all_numbers, axis=0, ddof=0)
    # print(np.std(all_numbers,axis=0))
    # for m, s in zip(means,stds):
    #     print('${:.3f}\%\pm{:.3f}$'.format(m, s), end='& ')

    print(header_str)
    ir = lambda x: int(round(x))
    ir = lambda x: x
    ss  = "${:.1f}\%{{\\scriptstyle\pm{:.1f}}}$ &".format(ir(means[0]), stds[0])
    ss += "${:.1f}\%{{\\scriptstyle\pm{:.1f}}}$ &".format(ir(means[1]), stds[1]) 
    ss += "NULL0 & "
    ss += "NULL1 & "
    ss += "${:.3f}\%{{\\scriptstyle\pm{:.3f}}}$ & ".format((means[2]), stds[2])
    ss += "${:.3f}\%{{\\scriptstyle\pm{:.3f}}}$ & ".format((means[3]), stds[3])
    ss += "NULL2 & "
    ss += "NULL3 & "
    ss += "NULL4 & "
    ss += "NULL5 & "
    ss += "NULL6 & "
    print(ss)
          
    
    # print(means)
    # print(stds)



# # TODO move inside class (trying to make metrics printing backwards-compatible for now).
# def format_intersection_metrics(self):

def compute_pothole_totals(episode_metrics):
    collisions = sum([_.frame_metrics[-1].count_pothole_collisions for _ in episode_metrics
                      if len(_.frame_metrics) > 0])
    total_potholes = sum([_.frame_metrics[-1].total_potholes for _ in episode_metrics
                          if len(_.frame_metrics) > 0])
    return collisions, total_potholes

def multiep_from_concluded(list_of_metrics):
    allms = []
    for mm in list_of_metrics:
        for ep in mm.all_episode_metrics:
            if ep.concluded:
                allms.append(ep)

    mjoint = MultiepisodeMetrics(None)
    mjoint.all_episode_metrics = allms
    mjoint.quantify()
    return mjoint

