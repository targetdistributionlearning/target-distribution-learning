from utils import Memory
import torch
from torch import Tensor
from torch.distributions.normal import Normal
from os.path import join as joindir
import numpy as np


class Rollout(object):
    def __init__(self):
        self.memory = None

    def rollout_render(self, env, network, args, running_state, render, video_folder):
        counter = 0
        state = env.reset()
        if args.state_norm:
            state = running_state(state)
        render.render()
        while counter < 5:
            action_mean = network._forward_actor(Tensor(state).unsqueeze(0))
            action = action_mean.data.numpy()[0]
            next_state, _, done, _ = env.step(action)
            next_state = running_state(next_state)
            render.render()
            if done:
                counter += 1
                state = env.reset()
                if args.state_norm:
                    state = running_state(state)
                render.render()
            state = next_state
        render.to_mp4(joindir(video_folder, '{}-{}.mp4'.format(args.label, args.seed)))

    def rollout_train(self, env, network, args, running_state, max_episode_steps):
        return self._rollout_with_memory(env, network, args, running_state, max_episode_steps, keep_memory=False)

    def rollout_validate_KL(self, env, network, args, running_state, max_episode_steps):
        return self._rollout_with_memory(env, network, args, running_state, max_episode_steps, keep_memory=True)

    def rollout_validate(self, env, network, args, running_state, max_episode_steps):
        return self._rollout_no_memory(env, network, args, running_state, max_episode_steps)

    def _rollout_with_memory(self, env, network, args, running_state, max_episode_steps, keep_memory=False):
        memory = Memory()
        num_steps = 0
        reward_list = []
        len_list = []
        while num_steps < args.batch_size:
            state = env.reset()
            if args.state_norm:
                state = running_state(state)
            if args.append_time:
                state = np.append(state, 1.0)
            reward_sum = 0
            for t in range(max_episode_steps):
                action_mean, action_std, value = network(Tensor(state).unsqueeze(0))
                action_mean = action_mean[0]
                action_std = action_std[0]
                action, y = network.select_action(action_mean, action_std)
                action_mean = action_mean.data.numpy()
                action = action.data.numpy()
                y = y.data.numpy()
                next_state, reward, done, info = env.step(action)
                reward_sum += reward
                if args.state_norm:
                    next_state = running_state(next_state)
                if args.append_time:
                    next_state = np.append(next_state, 1 - (t + 1) / max_episode_steps)
                mask = 0 if (done or ((t + 1) == max_episode_steps)) else 1
                memory.push(state, value, action_mean, action, y, mask, next_state, reward)

                if done:
                    break
                    
                state = next_state
                
            num_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)

            meanepreward = np.mean(reward_list)
            meaneplen = np.mean(len_list)

        if keep_memory:
            self.memory = memory
            self.old_std = network.action_std.data
            return meanepreward, meaneplen
        else:
            return memory, meanepreward, meaneplen, num_steps

    def _rollout_no_memory(self, env, network, args, running_state, max_episode_steps):
        num_steps = 0
        reward_list = []
        len_list = []
        while num_steps < args.batch_size:
            state = env.reset()
            if args.state_norm:
                state = running_state(state)
            reward_sum = 0
            for t in range(max_episode_steps):
                action_mean = network._forward_actor(Tensor(state).unsqueeze(0))
                action = action_mean.data.numpy()[0]
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                if args.state_norm:
                    next_state = running_state(next_state)
                if done:
                    break
                state = next_state
            num_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
        meanepreward_val = np.mean(reward_list)
        meaneplen_val = np.mean(len_list)
        return meanepreward_val, meaneplen_val

    def calculate_KL(self, network):
        old_std = self.old_std
        new_std = network.action_std.data

        states, _, old_mu, _, _, _, _, _ = self.memory.tsample()
        new_mu = network._forward_actor(states[:, :-1])

        d1 = Normal(old_mu, old_std)
        d2 = Normal(new_mu, new_std)
        kl = torch.distributions.kl.kl_divergence(d1, d2)
        kls = np.linalg.norm(kl.data.numpy(), axis=1)

        return kls.mean(), kls.max()
        