
from abc import ABCMeta, abstractmethod
import logging
import os
import six

log = logging.getLogger(os.path.basename(__file__))

@six.add_metaclass(ABCMeta)
class TrafficLightDecider:
    @abstractmethod
    def get_upcoming_traffic_light(self, *args, **kwargs):
        pass
