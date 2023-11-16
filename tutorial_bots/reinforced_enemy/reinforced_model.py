import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl
from botbowl.ai.env import EnvConf, BotBowlEnv
from examples.a2c.a2c_env import a2c_scripted_actions
from botbowl.ai.layers import *


class CNN(nn.Module):

    def __init__(self, spatial_shape, non_spatial_inputs, hidden_nodes, kernels, actions):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(spatial_shape[0], out_channels=kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding=1)

        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        stream_size += hidden_nodes
        self.linear1 = nn.Linear(stream_size, hidden_nodes)

        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(hidden_nodes, actions)

        self.train()
        self.reset_parameters()

        def forward(self, spatial_input, non_spatial_input):

            x1 = self.conv1(spatial_input)
            x1 = F.relu(x1)
            x1 = self.conv2(x1)
            x1 = F.relu(x1)

            # Concatenate the input streams
            flatten_x1 = x1.flatten(start_dim=1)

            x2 = self.linear0(non_spatial_input)
            x2 = F.relu(x2)

            flatten_x2 = x2.flatten(start_dim=1)
            concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)

            # Fully-connected layers
            x3 = self.linear1(concatenated)
            x3 = F.relu(x3)
            # x2 = self.linear2(x2)
            # x2 = F.relu(x2)

            # Output streams
            value = self.critic(x3)
            actor = self.actor(x3)

            # return value, policy
            return value, actor

        def act(self, spatial_inputs, non_spatial_input, action_mask):
            values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
            actions = action_probs.multinomial(1)
            # In rare cases, multinomial can  sample an action with p=0, so let's avoid that
            for i, action in enumerate(actions):
                correct_action = action
                while not action_mask[i][correct_action]:
                    correct_action = action_probs[i].multinomial(1)
                actions[i] = correct_action
            return values, actions

        def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
            value, policy = self(spatial_inputs, non_spatial_input)
            actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
            policy[~actions_mask] = float('-inf')
            log_probs = F.log_softmax(policy, dim=1)
            probs = F.softmax(policy, dim=1)
            action_log_probs = log_probs.gather(1, actions)
            log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
            dist_entropy = -(log_probs * probs).sum(-1).mean()
            return action_log_probs, value, dist_entropy

        def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
            values, actions = self(spatial_input, non_spatial_input)
            # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
            if action_mask is not None:
                actions[~action_mask] = float('-inf')
            action_probs = F.softmax(actions, dim=1)
            return values, action_probs

