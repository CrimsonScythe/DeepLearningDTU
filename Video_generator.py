# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:35:41 2021

@author: Théo
"""
#!pip install procgen
#!wget https://raw.githubusercontent.com/nicklashansen/ppo-procgen-utils/main/utils.py
#!wget https://raw.githubusercontent.com/MishaLaskin/rad/1246bfd6e716669126e12c1f02f393801e1692c1/TransformLayer.py

#if running on windows, download the files above and uncomment the line under
import utils



file = "Stackrun7/"


# Environment hyperparameters
num_envs = 32
num_levels = 100
n_stack = 3 # =1 for default

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *



def make_env(
	n_envs=32,
	env_name='starpilot',
	start_level=0,
	num_levels=100,
	use_backgrounds=True,
	normalize_obs=False,
	normalize_reward=True,
	seed=81,

	):
	"""Make environment for procgen experiments"""
	set_global_seeds(seed)
	set_global_log_levels(40)
	env = ProcgenEnv(
		num_envs=n_envs,
		env_name=env_name,
		start_level=start_level,
		num_levels=num_levels,
		distribution_mode='easy',
		use_backgrounds=use_backgrounds,
		restrict_themes=not use_backgrounds,
		render_mode='rgb_array',
		rand_seed=seed
	)
    
  
	env = VecExtractDictObs(env, "rgb")
	if n_stack >=2 :
		env = VecFrameStack(env, n_stack)
	env = VecNormalize(env, ob=normalize_obs, ret=normalize_reward)
	env = TransposeFrame(env)
	env = ScaledFloatFrame(env)
	env = TensorEnv(env)
	

	return env

#################################################################################
#Defining Encoders



def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


#################################################################################



import imageio

# Make evaluation environment
eval_env = make_env(num_envs, env_name = 'starpilot',use_backgrounds=True ,start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
in_channels=eval_env.observation_space.shape[0]
encoder =  ImpalaModel(in_channels=in_channels)
policy = Policy(encoder, 256, 15)
policy.load_state_dict(torch.load(file+"checkpoint_Final.pt"))
policy.eval()
policy.cuda()
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames).numpy()


#Produce and save a video, 512 frames long at 25fps
imageio.mimsave('vid_stack.mp4', frames, fps=25)


