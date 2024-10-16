import os
import torch

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]

# directory to store the results
results_dir = '/media/aryansingh/58A7-AC9A/results/CNNSpot_vqdm'
mkdir(results_dir)

# root to the testsets
dataroot = '/media/aryansingh/58A7-AC9A/RawGenImage'

# list of synthesis algorithms
#vals = ['BigGAN', 'glide', 'midjourney', 'vqdm', 'stable_diffusion_v_1_4']
#vals = ['ADM', 'BigGAN', 'glide', 'vqdm', 'stable_diffusion_v_1_4']
vals = ['ADM', 'BigGAN', 'glide', 'stable_diffusion_v_1_4', 'midjourney']

# indicates if corresponding testset has multiple classes
multiclass = [0, 0, 0, 0, 0]

# model
#model_path ='/media/aryansingh/58A7-AC9A/GenImageModels/ADM/checkpoints/experiment_name/model_epoch_latest.pth'
model_path ='/media/aryansingh/58A7-AC9A/GenImageModels/vqdm/checkpoints/CNNSpot/model_epoch_latest.pth'
