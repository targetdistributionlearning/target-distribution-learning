import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, dim_obs, dim_act, layer_norm=True, append_time=True, init_std=1.0):
        super(ActorCritic, self).__init__()

        self.append_time = append_time
        self.action_dim = dim_act
        
        self.actor_fc1 = nn.Linear(dim_obs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, dim_act)
        self.action_std = nn.Parameter(init_std * torch.ones(1, dim_act))

        if self.append_time:
            self.critic_fc1 = nn.Linear(dim_obs + 1, 64)
        else:
            self.critic_fc1 = nn.Linear(dim_obs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        """
        if self.append_time:
            action_mean = self._forward_actor(states[:, :-1])
        else:
            action_mean = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, self.action_std, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        return action_mean

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_std):
        y = torch.normal(torch.zeros(self.action_dim), torch.ones(self.action_dim))
        action = action_mean + y * action_std
        return action, y
        