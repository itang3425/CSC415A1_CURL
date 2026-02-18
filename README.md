# CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning

This repository is a fork of [the official implementation of CURL](https://github.com/MishaLaskin/curl). This repo uses different package versions (see curl_new_env_export.yml) than the original (see conda_env.yml).
An example Colab notebook is included for reference.

## Installation 

Open colab, run the following commands:

```
git clone https://github.com/itang3425/CSC415A1_CURL.git
cd CSC415A1_CURL

!mamba env create -f curl_new_env_export.yml

!/usr/local/envs/curl_new/bin/pip install "pip<24" "setuptools<65" "wheel<0.38"

!/usr/local/envs/curl_new/bin/pip install gym==0.19.0 --no-build-isolation

!/usr/local/envs/curl_new/bin/pip install git+https://github.com/1nadequacy/dmc2gym.git

!/usr/local/envs/curl_new/bin/python -m pip install -r CURL_pip_reqs.txt

!apt-get update -qq
!apt-get install -y -qq libgl1-mesa-dri libgl1-mesa-glx libegl1-mesa mesa-utils

import os
os.environ["MUJOCO_GL"] = "egl"           # headless GPU rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"   # helps PyOpenGL pick EGL

!mkdir curl_runs
```

## Instructions
To train a CURL agent on the `cartpole swingup` task from image-based observations run `bash script/run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / hyperparamters.
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_model \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./curl_runs \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 
```

To run the ablation test, add --detach_encoder flag.

In your console, you should see printouts that look like:

```
| train | E: 221 | S: 28000 | D: 18.1 s | R: 785.2634 | BR: 3.8815 | A_LOSS: -305.7328 | CR_LOSS: 190.9854 | CU_LOSS: 0.0000
| train | E: 225 | S: 28500 | D: 18.6 s | R: 832.4937 | BR: 3.9644 | A_LOSS: -308.7789 | CR_LOSS: 126.0638 | CU_LOSS: 0.0000
| train | E: 229 | S: 29000 | D: 18.8 s | R: 683.6702 | BR: 3.7384 | A_LOSS: -311.3941 | CR_LOSS: 140.2573 | CU_LOSS: 0.0000
| train | E: 233 | S: 29500 | D: 19.6 s | R: 838.0947 | BR: 3.7254 | A_LOSS: -316.9415 | CR_LOSS: 136.5304 | CU_LOSS: 0.0000
```

For reference, the maximum score for cartpole swing up is around 845 pts, so CURL has converged to the optimal score. This takes about an hour of training depending on your GPU. 

Log abbreviation mapping:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - mean episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
CU_LOSS - average loss of the CURL encoder
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh. 

For GPU accelerated rendering, make sure EGL is installed on your machine and set `export MUJOCO_GL=egl`. For environment troubleshooting issues, see the DeepMind control documentation.
