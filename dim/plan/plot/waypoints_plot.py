#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Connects with a CARLA simulator and displays the available start positions
for the current map."""

from __future__ import print_function

import argparse
import logging
import sys
import time
import os
import pkgutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

import dim.env.config.waypoint_config as wpcfg

from carla.client import make_carla_client
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from dim.waypointer import Waypointer, WaypointerParams


def waypoints_plot(args, route_start_node, route_start_ori, route_finish_node, route_finish_ori, params):
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. The same way as in the client example.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        # We load the default settings to the client.
        scene = client.load_settings(CarlaSettings())
        print("Received the start positions")

        try:
            carla_module = pkgutil.get_loader("carla")
            image_path = os.path.join(os.path.dirname(carla_module.get_filename()), 'planner', '%s.png' % scene.map_name)
            image = mpimg.imread(image_path)
            carla_map = CarlaMap(scene.map_name, 0.1653, 50)  # TODO: 0.1643? https://github.com/carla-simulator/carla/issues/644
        except IOError as exception:
            logging.error(exception)
            logging.error('Cannot find map "%s"', scene.map_name)
            sys.exit(1)

        # our waypoint generating object
        waypointer = Waypointer(params, scene.map_name)

        # route-independent things to plot
        intersection_nodes = waypointer.get_intersection_nodes()
        road_nodes = waypointer.get_road_nodes()

        waypoints = waypointer.get_waypoints(route_start_node, route_start_ori,
                                             route_finish_node, route_finish_ori)

        # all carla nodes between a start-node and finish-node (inclusive)
        route_nodes = waypointer.route_nodes

        # for the more important nodes, we'll plot their x/y values in the plot
        route_nodes_important = waypointer.get_route_nodes_at_intersections(route_nodes)
        route_nodes_important.add(route_start_node)
        route_nodes_important.add(route_finish_node)

        plot_types = [('intersection_nodes', intersection_nodes),
                      ('road_nodes', road_nodes),
                      ('route_nodes', route_nodes),
                      ('waypoints', waypoints)]

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 12))
        for i, (name, positions_to_plot) in enumerate(plot_types):
            ix, iy = i // 2, i % 2
            axs[ix, iy].imshow(image)
            axs[ix, iy].set_title(name)

            for j, position in enumerate(positions_to_plot):

                pixel = carla_map.convert_to_pixel(position)

                circle = Circle((pixel[0], pixel[1]), 12, color='r', label='A point')
                axs[ix, iy].add_patch(circle)

                if name == 'intersection_nodes':
                    axs[ix, iy].text(pixel[0], pixel[1], str(position), size='x-small')
                elif name == 'route_nodes':
                    if j == 0:
                        axs[ix, iy].text(pixel[0], pixel[1], "start " + str(position), size='x-small')
                    elif j == len(positions_to_plot) - 1:
                        axs[ix, iy].text(pixel[0], pixel[1], "finish " + str(position), size='x-small')
                    elif position in route_nodes_important:
                        axs[ix, iy].text(pixel[0], pixel[1], str(position), size='x-small')
                elif name == 'waypoints':
                    format_pos = lambda x: 'waypoint [{0:0.2f}, {1:0.2f}]'.format(x[0], x[1])
                    if j == 0:
                        axs[ix, iy].text(pixel[0], pixel[1], "first " + format_pos(position), size='x-small')
                    elif j == len(positions_to_plot) - 1:
                        axs[ix, iy].text(pixel[0], pixel[1], "last " + format_pos(position), size='x-small')

        fig.savefig('waypoints.pdf', orientation='landscape', bbox_inches='tight')


def main(route_start_node, route_start_ori, route_finish_node, route_finish_ori, params):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-pos', '--positions',
        metavar='P',
        default='all',
        help='Indices of the positions that you want to plot on the map. '
             'The indices must be separated by commas (default = all positions)')
    argparser.add_argument(
        '--no-labels',
        action='store_true',
        help='do not display position indices')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:
            waypoints_plot(args, route_start_node, route_start_ori, route_finish_node, route_finish_ori, params)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
        except RuntimeError as error:
            logging.error(error)
            break


if __name__ == '__main__':

    route_start_node = (11, 30)
    route_start_ori = "NORTH"
    route_finish_node = (45, 0)
    route_finish_ori = "EAST"

    params = WaypointerParams({})
    params.meters_per_waypoint = 1.0
    params.min_waypoint_distance = 1.0

    try:
        main(route_start_node, route_start_ori, route_finish_node, route_finish_ori, params)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
