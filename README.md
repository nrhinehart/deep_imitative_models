[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

# Purposes

1. Collect data to train a Deep Imitative Model in CARLA
2. Apply a Deep Imitative Model to CARLA

Openreview: [https://openreview.net/pdf?id=Skl4mRNYDr](https://openreview.net/pdf?id=Skl4mRNYDr)

<img src="example_ims/mixture.gif" width="400"/><img src="example_ims/region.gif" width="400"/>

# Primary files

1. `carla_agent.py` : Interface to perform both purposes above
2. `dim/plan/dim_plan.py` : The core planning module that assembles the prior from the model and the goal likelihood.
3. `dim/plan/goal_distributions.py` : Implementations of various goal likelihood distributions.
4. `dim/env/run_carla_episode.py` : Runs a single CARLA episode with autopilot or a DIM.

# Setup
## You'll need to install CARLA. This was tested with CARLA 0.8.4, release version.

0. Install the PRECOG repo: [https://github.com/nrhine1/precog](https://github.com/nrhine1/precog)
1. Set the path to CARLA in `env.sh` as the `PYCARLA` variable.
2. Create and activate a conda environment (you can use virtualenv instead, if you'd like)
3. Source the environment for the repo
4. Install the dependencies.
5. Update the CARLA devkit `transform.py`
6. Update the checkpoint file of the model

```bash
export DIMCONDAENV=dim3
conda create -n $DIMCONDAENV python=3.6.6
conda activate $DIMCONDAENV
source env.sh
pip install -r requirements.txt
cp $DIMROOT/ext/transform.py $PYCARLA/carla/
python $DIMROOT/scripts/create_checkpoint_file.py $DIMROOT/models/model_0/
```

# Data collection with the autopilot:

```bash
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 python $DIMROOT/carla_agent.py \
main.pilot=auto \
data.save_data=True \
experiment.scene="Town01" \
experiment.n_vehicles=50 \
plotting.plot=False experiment.frames_per_episode=5060 \
experiment.n_episodes=1000
```

This command will collect 1000 episodes. With the current defaults, each episodes will consist of 100 serialized data points

# Applying a Deep Imitative Model. 
It starts the server and runs it offscreen and applies the default model (in `dim_config.yaml`) with the RegionIndicator goal likelihood to `Town01`.
```bash
export CUDA_VISIBLE_DEVICES=0; SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 python -m pdb -c c $DIMROOT/carla_agent.py \
main.log_level=DEBUG \
main.pilot=dim \
dim.goal_likelihood=RegionIndicator \
waypointer.interpolate=False \
experiment.scene=Town01 \
experiment.n_vehicles=50
```

## Generate high-res plots without debug info:
Add the following options: `plotting.plot_text_block=False plotting.hires_plot=1`

# Citation
```
@inproceedings{Rhinehart2020Deep,
title={Deep Imitative Models for Flexible Inference, Planning, and Control},
author={Nicholas Rhinehart and Rowan McAllister and Sergey Levine},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Skl4mRNYDr}
}
```
