import os
import gym
import d4rl
import scipy
import tqdm
import functools
import argparse
import time
start_time = time.time()

import faulthandler
faulthandler.enable()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Bidirectional_Car_Env
from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, MlpScoreNet
from utils import get_args
from dataset.dataset import Diffusion_buffer


def train_behavior(args):
    for dir in ["./models", "./logs", "./results"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", str(args.expid))):
        os.makedirs(os.path.join("./models", str(args.expid)))
    if not os.path.exists(os.path.join("./results", str(args.expid))):
        os.makedirs(os.path.join("./results", str(args.expid)))
    writer = SummaryWriter("./logs/" + str(args.expid))
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.actor_type == "large":
        score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, 
                              marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    elif args.actor_type == "small":
        score_model= MlpScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, 
                                 marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)
    dataset = Diffusion_buffer(args)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = Adam(score_model.parameters(), lr=args.lr)

    print("training diffusion")
    n_epochs = args.n_behavior_epochs
    tqdm_epoch = tqdm.trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        
        for x, condition in data_loader:
            x = x[:, 1:] # action
            x = x.to(args.device)
            condition = condition.to(args.device)
            score_model.condition = condition
            loss = loss_fn(score_model, x, args.marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            score_model.condition = None
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Print the averaged training loss so far.
            
        # Update the checkpoint after each epoch of training.
        if (epoch+1) % 50 == 0 and args.save_model:
            torch.save(score_model.state_dict(), os.path.join("./models", str(args.expid), "ckpt{}.pth".format(epoch+1)))
            print('Saved the model!')
            print('Running time: {} minutes.'.format(round((time.time()-start_time)/60., 2)))
            
        if args.writer:
            args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)

    print("Training finished...")


if __name__ == "__main__":

    args = get_args()
    train_behavior(args)


    print('Running time: {} minutes.'.format(round((time.time()-start_time)/60., 2)))