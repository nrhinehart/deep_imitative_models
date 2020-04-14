
import socket

import precog.utils.class_util as classu
import precog.utils.gsheets_util as gsheets_util

if gsheets_util.have_pygsheets:
    class MultiepisodeResults(gsheets_util.GSheetsResults):
        @classu.member_initialize
        def __init__(self, tag, multiep_metrics, root_dir):
            super().__init__(sheet_name="carla_agent_results", worksheet_name="multiepisode_results")
            # Claim a row.
            self.claim_row(tag)
            self.conf_tags = ['host-{}'.format(socket.gethostname()), root_dir]

        def update(self):
            self.multiep_metrics.quantify()
            # Don't do anything until we've actually populated some metrics
            if self.multiep_metrics.n_total == 0:
                mean_stats_list = ['None']*7
            else:
                mean_stats_list = self.multiep_metrics.get_mean_stats_list()
            # If we've populated metrics, update our row.
            row = [self.multiep_metrics.success_percentage, self.multiep_metrics.n_total] + mean_stats_list + self.conf_tags
            self.update_claimed_row(row)

    class EpisodeResults(gsheets_util.GSheetsResults):
        def __init__(self):
            super().__init__(sheet_name="carla_agent_results", worksheet_name="episode_results")
