
import logging
import os

from carla.settings import CarlaSettings
from carla.sensor import Camera, Lidar

import dim.env.preprocess.carla_preprocess as carla_preprocess

log = logging.getLogger(os.path.basename(__file__))

def create_settings(n_vehicles, n_pedestrians, quality_level,
                    image_size, use_lidar, root_dir, lidar_params, z_BEV=25):
    # Create a CarlaSettings object. This object is a wrapper around
    # the CarlaSettings.ini file. Here we set the configuration we
    # want for the new episode.
    settings = CarlaSettings()

    # Make sure that the server is running with '-benchmark' flag
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=n_vehicles,
        NumberOfPedestrians=n_pedestrians,
        # WeatherId=4,
        WeatherId=1,
        # WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=quality_level)
    settings.randomize_seeds()
    log.info("Vehicle seed: {}".format(settings.SeedVehicles))

    # TODO make lidar params pay attention to the H x W of BEV?
    settings.lidar_params = lidar_params

    # Now we want to add a couple of cameras to the player vehicle.
    # We will collect the images produced by these cameras every
    # frame.

    # The default camera captures RGB images of the scene.
    front_cam_rgb = Camera('CameraRGB')
    front_cam_position = (0., 0., 1.5)

    # angel_cam_position  = (-5.5, 0.0, 2.8)
    # angel_cam_position  = (-5.5, 0.0, 3.8)
    angel_cam_position  = (-7.5, 0.0, 4.8)    

    # For visualization.
    front_cam_rgb.FOV = 90

    # Set image resolution in pixels.
    front_cam_rgb.set_image_size(*image_size)

    # Set its position relative to the car in meters.
    # Default forward-facing
    # camera0.set_position(0.30, 0, 1.30)
    # front_cam_rgb.set_position(*front_cam_position)
    front_cam_rgb.set_position(*angel_cam_position)
    # front_cam_rgb.set_rotation(roll=0., pitch=-15., yaw=0.0)
    front_cam_rgb.set_rotation(roll=0., pitch=-15., yaw=0.0)
    settings.add_sensor(front_cam_rgb)

    # Bird's eye view.
    bev_cam_rgb = Camera('CameraRGBBEV')
    # Note that at FOV=90, image width across ground plane is z*2 (if camera is parallel to ground)
    # z_BEV = 50
    bev_cam_rgb.set_position(x=0, y=0, z=z_BEV)
    bev_cam_rgb.set_rotation(pitch=-90, roll=0, yaw=-90)
    bev_cam_rgb.set_image_size(400, 400)
    # bev_cam_rgb.set_image_size(200, 200)
    
    # bev_cam_rgb.set_image_size(*image_size)
    settings.add_sensor(bev_cam_rgb)

    # bev_cam_sem = Camera('CameraSemanticBEV', PostProcessing='SemanticSegmentation')
    # # Note that at FOV=90, image width across ground plane is z*2 (if camera is parallel to ground)
    # bev_cam_sem.set_position(x=0, y=0, z=z_BEV)
    # bev_cam_sem.set_rotation(pitch=-90, roll=0, yaw=0)
    # bev_cam_sem.set_image_size(*image_size)
    # settings.add_sensor(bev_cam_sem)

    # Let's add another camera producing ground-truth depth.
    front_cam_depth = Camera('CameraDepth', PostProcessing='Depth')
    front_cam_depth.set_image_size(*image_size)
    front_cam_depth.set_position(*front_cam_position)
    settings.add_sensor(front_cam_depth)

    # Semantic segmentation is black! Even in the demo code of manual_control.py ... Might be an issue with graphics card / Unreal support?
    # See https://github.com/carla-simulator/carla/issues/151
    camera = Camera('CameraSemantic', PostProcessing='SemanticSegmentation')
    camera.set(FOV=90.0)
    camera.set_image_size(*image_size)
    camera.set_position(*front_cam_position)
    camera.set_rotation(pitch=0, yaw=0, roll=0)
    settings.add_sensor(camera)

    if use_lidar:
        settings.add_sensor(create_lidar())
    else:
        log.info("Not adding lidar")
    return settings

def create_lidar(lidar_position=(0., 0., 2.5)):
    log.info("Adding Lidar")
    lidar = Lidar('Lidar32')
    lidar.set_position(*lidar_position)
    lidar.set_rotation(0, 0, 0)
    lidar.set(
        Channels=32,
        Range=50,
        PointsPerSecond=100000,
        RotationFrequency=10,
        UpperFovLimit=10,
        LowerFovLimit=-30)
    return lidar
