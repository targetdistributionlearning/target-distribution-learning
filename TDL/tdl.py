from utils import ZFilter, Render, Schduler, to_cuda
from policies import ActorCritic
from rollout import Rollout
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from os.path import join as joindir
from os import makedirs as mkdir
import numpy as np
import math
import pdb


NUM_TOTAL_GPUS = 8
RESULT_DIR = joindir('../result', '.'.join(__file__.split('.')[:-1]))
mkdir(RESULT_DIR, exist_ok=True)


def runner(args):
    # cuda id 
    cid = mp.current_process().pid % NUM_TOTAL_GPUS
    
    # environment
    env = gym.make(args.env_name)
    dim_obs = env.observation_space.shape[0]
    dim_act = env.action_space.shape[0]
    max_episode_steps = env._max_episode_steps

    # seeding
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # policy and value network
    network = ActorCritic(dim_obs, dim_act, layer_norm=args.layer_norm, append_time=args.append_time, init_std=args.init_std)
    optimizer = opt.Adam(network.parameters(), lr=args.lr)

    # running state normalization
    running_state = ZFilter((dim_obs,), clip=5.0)

    # total number of steps of interaction with environment
    global_steps = 0
    
    # render to mp4
    render = Render(env.env.sim, args.env_name)
    video_folder = joindir(RESULT_DIR, args.label)
    mkdir(video_folder, exist_ok=True)

    step_size = Schduler(args.step_size, args.schedule_stepsize, args.num_episode, end_val=0.01)
    lr = Schduler(args.lr, args.schedule_adam, args.num_episode)
    y2_max = Schduler(args.y2_max, args.schedule_y2max, args.num_episode, end_val=0.05)
    mean_ratio = Schduler(args.mean_ratio, args.schedule_meanratio, args.num_episode)

    rollout = Rollout()

    # for direct method
    # y2_max = compute_allowed_mu2(0, args.epsilon, args.delta, dim_act)

    for i_episode in range(args.num_episode):
        # step0: validation 
        if i_episode % args.val_num_episode == 0:
            if args.record_KL:
                meanepreward_val, meaneplen_val = rollout.rollout_validate_KL(env, network, args, running_state, max_episode_steps)
            else:
                meanepreward_val, meaneplen_val = rollout.rollout_validate(env, network, args, running_state, max_episode_steps)
        else:
            meanepreward_val, meaneplen_val = np.nan, np.nan

        # step0: save mp4
        if (i_episode + 1) == args.num_episode:
            rollout.rollout_render(env, network, args, running_state, render, video_folder)

        # step1: perform current policy to collect on-policy transitions
        memory, meanepreward, meaneplen, num_steps = rollout.rollout_train(env, network, args, running_state, max_episode_steps)
        global_steps += num_steps
        
        # step2: extract variables from trajectories
        batch_size = len(memory)
        states, values, action_means, actions, ys, masks, next_states, rewards = memory.tsample()
        returns, deltas, advantages = Tensor(batch_size), Tensor(batch_size), Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        # we cannot do advantages normalization here, moreover, only the sign matters 
        if args.advantage_norm:
            advantages = advantages - advantages.median()

        # step3: set targets

        if args.method == 'ES':
            target_means = Variable(action_means + step_size.value() * (actions - action_means))
        elif args.method == 'direct':
            y2_max_value = y2_max.value()
            ys_norm = ys.pow(2).sum(dim=1, keepdim=True)
            ys_scale = ys / ys_norm.sqrt() * np.sqrt(y2_max_value)
            ys_target = torch.where(ys_norm > y2_max_value, ys_scale, ys)
            ys_target = torch.where(advantages.unsqueeze(1) >= 0, ys_target, -ys_target)
            target_means = Variable(action_means + ys_target * network.action_std)
        elif args.method == 'ES-MA1':
            mask_advantages = advantages.clone().masked_fill_(advantages < 0, 0)
            half_width = args.n_points
            ratio = mean_ratio.value()
            new_ys = torch.zeros(ys.shape)
            start_ind = 0
            for i in range(batch_size):
                if mask_advantages[i] > 0:
                    norm = 0.0
                    for j in range(max(start_ind, i - half_width), i + half_width + 1):
                        weight = mask_advantages[j] * args.gamma ** np.abs(i - j)
                        norm += weight
                        new_ys[i] += ys[j] * weight
                        if masks[j] == 0:
                            break
                    if norm > 0:
                        new_ys[i] = new_ys[i] / norm * ratio + ys[i] * (1 - ratio)
                if masks[i] == 0:
                    start_ind = i + 1
            target_means = Variable(action_means + step_size.value() * new_ys * network.action_std)
        elif args.method == 'ES-MA2':
            mask_advantages = advantages.clone().masked_fill_(advantages < 0, 0)
            half_width = args.n_points
            new_ys = torch.zeros(ys.shape)
            start_ind = 0
            for i in range(batch_size):
                if mask_advantages[i] > 0:
                    norm = 0.0
                    for j in range(max(start_ind, i - half_width), i + half_width + 1):
                        weight = mask_advantages[j] * args.beta ** np.abs(i - j)
                        norm += weight
                        new_ys[i] += ys[j] * weight
                        if masks[j] == 0:
                            break
                    if norm > 0:
                        new_ys[i] = new_ys[i] / norm
                if masks[i] == 0:
                    start_ind = i + 1
            target_means = Variable(action_means + step_size.value() * new_ys * network.action_std)

        # self adaptation
        multiplier = ys.pow(2)
        multiplier = multiplier.masked_fill_(advantages.unsqueeze(1) < 0, 1.0).mean(dim=0).sqrt()
        network.action_std = nn.Parameter((network.action_std * multiplier))

        # step4: learn
        # load the network to GPU and train on GPU, then use the network on CPU
        if args.use_cuda:
            states, target_means, returns, network_train = to_cuda(cid, states, target_means, returns, network)
        else:
            network_train = network

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_action_means = network_train._forward_actor(minibatch_states[:, :-1])
            minibatch_target_means = target_means[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = network_train._forward_critic(minibatch_states).flatten()

            loss_policy = torch.mean((minibatch_target_means - minibatch_action_means).pow(2))

            # not sure the value loss should be clipped as well 
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value 
            if args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
            else:
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))
            if i_epoch > int(30 * batch_size / args.minibatch_size):
                loss_coeff_value = 0
            else:
                loss_coeff_value = args.loss_coeff_value

            # https://en.wikipedia.org/wiki/Differential_entropy
            # entropy of normal distribution should be ln(sig * sqrt(2 * pi * e))
            # torch.mean(network.actor_logstd + .5 * math.log(2.0 * math.pi * math.e))
            # the const term is of no use, ignored
            if args.loss_coeff_entropy > 0:
                loss_entropy = torch.mean(torch.log(network_train.action_std))
                total_loss = loss_policy + loss_coeff_value * loss_value - args.loss_coeff_entropy * loss_entropy
            else:
                loss_entropy = np.nan
                total_loss = loss_policy + loss_coeff_value * loss_value
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if lr.schedule != 'none':
            lr_now = lr.value()
            # ref: https://stackoverflow.com/questions/48324152/
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if args.use_cuda:
            network = network_train.cpu()

        if (i_episode % args.val_num_episode == 0) and (args.record_KL):
            mean_KL, max_KL = rollout.calculate_KL(network)
        else:
            mean_KL, max_KL = np.nan, np.nan

        if (i_episode + 1) % args.log_num_episode == 0:
            print('==============================================')
            print('[{}] #ep={} Reward: {:.4f} ({:.4f})  Len: {:.0f}({:.0f})'.format(
                args.label, i_episode, 
                meanepreward, meanepreward_val, 
                meaneplen, meaneplen_val, 
            ))
            print('advantages_positive_ratio={:3f} action_std={} mean_KL={} max_KL={}'.format(
                float((advantages > 0).sum()) / batch_size,
                network.action_std,
                mean_KL, max_KL
            ))
            network_action_std = ['{:.2f}'.format(i) for i in list(network.action_std.detach().numpy()[0])]
            print('action_std={}'.format(network_action_std))

        lr.step()
        step_size.step()
        y2_max.step()
        mean_ratio.step()

    return network
