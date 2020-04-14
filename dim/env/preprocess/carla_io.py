
from __future__ import print_function

import attrdict
import dill
import glob
import logging
import os
import os.path as op

import carla.carla_server_pb2 as carla_protocol

log = logging.getLogger(os.path.basename(__file__))

def save_data_at_frame(measurements, sensor_data, control, directory, frame, settings, blacklist=[]):
    """Save sensor and measurement data to disk

    :param measurements: Measurements object
    :param sensor_data: sensor_data dict
    :param directory: str output directory
    :param frame: int frame index
    """
    log.info("Saving data at frame {:06d}".format(frame))
    if not os.path.isdir(directory): os.makedirs(directory)

    # Serialize the sensor data to disk.
    for name, datum in sensor_data.items():
        if name in blacklist: continue
        filename = op.join(directory, '{}_{:06d}'.format(name, frame))
        with open(filename + '.dill', 'wb') as f: dill.dump(datum, f)

    # Serialize the Measurements and Controls to disk.
    filename = op.join(directory, 'Measurements_{:06d}.pb'.format(frame))
    filename_c = op.join(directory, 'Control_{:06d}.pb'.format(frame))
    with open(filename, 'wb') as f: f.write(measurements.SerializeToString())
    with open(filename_c, 'wb') as f: f.write(control.SerializeToString())

def load_data_at_frame(directory, frame,
                       sensor_names=['CameraRGB', 'CameraDepth', 'CameraRGBBEV', 'Lidar32',
                                     'CameraSemantic']):
    """Load sensor and measurement data from disk

    :param directory: str location of data
    :param frame: int frame index
    :returns: dictionary of loaded data
    :rtype: AttrDict
    """
    # TODO these are hardcoded names... based on the collection.
    data = attrdict.AttrDict()
    for sensor_name in sensor_names:
        filename = op.join(directory, '{}_{:06d}'.format(sensor_name, frame)) + '.dill'
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                data[sensor_name] = dill.load(f)
    data.measurements = load_measurements_at_frame(directory, frame)
    data.control = load_control_at_frame(directory, frame)
    return data

def is_frame_index_valid(directory, frame_index):
    try:
        filename = op.join(directory, 'Measurements_{:06d}.pb'.format(frame_index))
        assert(os.path.isfile(filename))
        filename = op.join(directory, 'Control_{:06d}.pb'.format(frame_index))
        assert(os.path.isfile(filename))

        # TODO check for sensor data.
        return True
    except AssertionError:
        return False
    
def load_measurements_at_frame(directory, frame):
    filename = op.join(directory, 'Measurements_{:06d}.pb'.format(frame))            
    with open(filename, 'rb') as f:
        measurements = carla_protocol.Measurements()
        measurements.ParseFromString(f.read())
    return measurements

def load_control_at_frame(directory, frame):
    filename = op.join(directory, 'Control_{:06d}.pb'.format(frame))            
    with open(filename, 'rb') as f:
        control = carla_protocol.Control()
        control.ParseFromString(f.read())
    return control

def get_max_frame(directory):
    measurements = sorted(glob.glob(directory + '/Measurements*.pb'))
    last_frame = int(op.splitext(op.basename(measurements[-1]).split('_')[1])[0])
    return last_frame
