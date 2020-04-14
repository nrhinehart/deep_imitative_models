
import logging
import os
import random
import pdb
import signal
import string
import subprocess

import dim.env.config

log = logging.getLogger(os.path.basename(__file__))

def get_pid(name):
    return subprocess.check_output(["pidof", name])

def randstring(N):
    # Re-seed the random number generator for a pseudorandom string (across processes at same random state)
    randstate = random.getstate()
    random.seed()
    ss = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
    # Reset the rng back to its previous state
    random.setstate(randstate)
    return ss

def start_server(name, carla_dir, server_log_dir, port=2000):
    # kill_old_carla()
    
    """Start the server for the given map name. Assumes no server is already started."""
    server_log_fn = '{}/server_log_p{}_{}.txt'.format(server_log_dir, port, randstring(10))
    assert(os.path.isdir(carla_dir))
    assert(os.path.isdir(server_log_dir))
    file_pointer = open(server_log_fn, 'w')

    # Build the server starting command.
    config_dir = os.path.dirname(dim.env.config.__file__)
    
    binary = '{}/CarlaUE4.sh'.format(carla_dir)
    assert(os.path.isfile(binary))
    settings_fn = '{}/CarlaSettingsP{}.ini'.format(config_dir, port)
    assert(os.path.isfile(settings_fn))

    scene = ' /Game/Maps/{}'.format(name)
    # [TODO] why does actually passing the settings file correctly break Carla???
    cwd = os.getcwd()
    settings = ' -carla-settings="{}"'.format(os.path.relpath(settings_fn, carla_dir))
    args = ' -carla-server -benchmark -fps={} -windowed -ResX=500 -ResY=500 -nocore'.format(10)
    cmd = binary + scene + settings + args
    log.info("Server startup command: '{}'".format(cmd))
    # Run the server in a subprocess.
    os.chdir(carla_dir)
    foo = subprocess.Popen(cmd.split(), stdout=file_pointer, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    os.chdir(cwd)
    return foo, server_log_fn

def kill_server(group_pid):
    return os.killpg(group_pid, signal.SIGKILL)

def kill_old_carla():
    name = "CarlaUE4"
    max_tries = 100
    n_tries = 0
    while n_tries < max_tries:
        n_tries += 1
        try:
            pid = subprocess.check_output(["pidof", name])
            try:
                os.kill(int(pid.strip()), signal.SIGTERM)
            except ProcessLookupError:
                break
        except subprocess.CalledProcessError:
            break
    if n_tries >= max_tries:
        log.error("Couldn't kill old carla!")
