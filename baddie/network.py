from enum import Enum
import numpy as np
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl
from botbowl.ai.env import EnvConf

from reinforced_enemy.reinforced_agent import make_env


class ConfigParams(Enum):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_steps = 12000000
    num_processes = 4 # was 5
    steps_per_update = 8 #10  # 50 # 1000
    batch_size = 8 #32  # 512  # generally the size should be the 2^n, the bigger it is the higher the exploration, but slower learning
    buffer_size = 8000
    multiple_updates = 2
    learning_rate = 5e-6  # 5e-6
    gamma = 0.99
    tau = 0.01 # 0.75 worked well# 0.01 # setting this higher helped talk abt this in paper
    gradient_clip = 2.0 #1.0 #1.5 # was 10
    q_regularization = 0.05 #0.005 #0.0
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.05  # witout it the gradient gets to big and model starts getting worse after the while
    log_interval = 25
    save_interval = 50
    reset_steps = 5000  # The environment is reset after this many steps it gets stuck
    patience = 20
    selfplay_window = 5
    selfplay_save_steps = int(num_steps / 100000)
    selfplay_swap_steps = selfplay_save_steps
    num_hidden_nodes = 1024
    ppcg = True
    env_size = 11  # Options are 1,3,5,7,11
    env_name = f"botbowl-{env_size}"
    env_conf = EnvConf(size=env_size)#, pathfinding=True)  # pathfinndng=False
    selfplay = False  #todo make it work
    exp_id = str(uuid.uuid1())
    model_dir = f"models/{env_name}/"
    model_path = f"models/{env_name}/bc_baseline.nn"  #bc_baseline.nn
    agent_path = f"models/{env_name}/best.nn"  # bc_model.nn"


env = make_env(ConfigParams.env_conf.value)
spat_obs, non_spat_obs, action_mask = env.reset()
spatial_obs_space = spat_obs.shape
non_spatial_obs_space = non_spat_obs.shape[0]
action_space = len(action_mask)


class ResidualBlock(nn.Module):
    def __init__(self, kernels=[128, 128]):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding="same")
        self.conv1 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding="same")
        #self.conv3 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding="same")

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.conv0(x))
        out = F.leaky_relu(self.conv1(out))
        out = F.leaky_relu(self.conv2(out))
        #out = F.leaky_relu(self.conv3(out))
        out += residual
        return out

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv0.weight.data.mul_(relu_gain)
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        #self.conv3.weight.data.mul_(relu_gain)


class ChannelAttention(nn.Module):
    def __init__(self, kernels=[128, 64, 17], spatial_shape=spatial_obs_space):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=kernels[0]+spatial_shape[0],
                               out_channels=kernels[1], kernel_size=3, stride=1, padding='same')
        self.conv1 = nn.Conv2d(in_channels=kernels[1], out_channels=kernels[1], kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=kernels[1], out_channels=kernels[2], kernel_size=3, stride=1, padding='same')
        self.linear0 = nn.Linear(1024, 64)
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(1024, 17)

    def forward(self, x):#, latent_space):
        # out = torch.cat(F.leaky_relu(self.conv0(x[0])), F.sigmoid(F.leaky_relu(self.linear0(latent_space))))
        sigmoid = F.sigmoid(F.leaky_relu(self.linear0(x[1])))
        out = torch.multiply(self.conv0(x[0]), sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1))
        sigmoid = F.sigmoid(F.leaky_relu(self.linear1(x[1])))
        out = torch.multiply(self.conv1(out), sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1))
        sigmoid = F.sigmoid(F.leaky_relu(self.linear2(x[1])))
        out = torch.multiply(self.conv2(out), sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1))
        return out

    def reset_parameters(self, relu_gain):
        sigmoid_gain = nn.init.calculate_gain('sigmoid')
        self.conv0.weight.data.mul_(sigmoid_gain)
        self.conv1.weight.data.mul_(sigmoid_gain)
        self.conv2.weight.data.mul_(sigmoid_gain)
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.linear2.weight.data.mul_(relu_gain)


class CNNPolicy(nn.Module):

    def __init__(self, spatial_shape=spatial_obs_space, non_spatial_inputs=non_spatial_obs_space,
                 hidden_nodes=ConfigParams.num_hidden_nodes.value, kernels=[128, 128], actions=action_space,
                 filename=None):
        super(CNNPolicy, self).__init__()

        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_shape[0], out_channels=kernels[0], kernel_size=4, stride=1, padding="same")
        # Residual block layers
        # self.conv_res = self._make_layer(ResidualBlock, kernels, 4)
        self.conv_res0 = ResidualBlock()
        self.conv_res1 = ResidualBlock()
        self.conv_res2 = ResidualBlock()
        self.conv_res3 = ResidualBlock()
        # Non-spatial input stream
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)
        self.linear1 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear3 = nn.Linear(hidden_nodes, hidden_nodes)
        # Concatenated stream
        self.linear4 = nn.Linear(1024+128*spatial_shape[1]*spatial_shape[2], hidden_nodes)
        self.linear5 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear6 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear7 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear_out = nn.Linear(hidden_nodes, 1)
        # Convolutions with channel attention
        # self.conv_ch = self._make_layer(ChannelAttention, [128, 64, 17], 1)
        self.conv_ch = ChannelAttention([128, 64, 17])
        #self.linear_actor = nn.Linear(1024, 25)
        self.linear_actor = nn.Linear(1024, 24)


        # Linear layers
        # stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        # stream_size += hidden_nodes
        # self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(hidden_nodes, actions)

        self.train()
        self.reset_parameters()

        if filename is not None:
            # self.load_state_dict(torch.load(filename))
            self.torch.load(filename)
            self.eval()

    @staticmethod
    def _make_layer(block, kernels, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(kernels))
        return nn.Sequential(*layers)

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv_res0.reset_parameters()
        self.conv_res1.reset_parameters()
        self.conv_res2.reset_parameters()
        self.conv_res3.reset_parameters()
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.linear2.weight.data.mul_(relu_gain)
        self.linear3.weight.data.mul_(relu_gain)
        self.linear4.weight.data.mul_(relu_gain)
        self.linear5.weight.data.mul_(relu_gain)
        self.linear6.weight.data.mul_(relu_gain)
        self.linear7.weight.data.mul_(relu_gain)
        self.conv_ch.reset_parameters(relu_gain)
        self.actor.weight.data.mul_(relu_gain)
        self.critic.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through two convolutional layers

        x1 = self.conv1(spatial_input)
        # x1 = self.conv_res(x1)  # todo check that, yields too big results
        x1 = self.conv_res0(x1)
        x1 = self.conv_res1(x1)
        x1 = self.conv_res2(x1)
        x1 = self.conv_res3(x1)
        #x1_cat = torch.cat((x1, spatial_input), dim=1)
        x1_cat = torch.cat((spatial_input, x1), dim=1)

        # Concatenate the input streams
        flatten_x1 = x1.flatten(start_dim=1)

        x2 = F.leaky_relu(self.linear0(non_spatial_input))
        x2 = F.leaky_relu(self.linear1(x2))
        x2 = F.leaky_relu(self.linear2(x2))
        x2 = F.leaky_relu(self.linear3(x2))

        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)

        x3 = F.leaky_relu(self.linear4(concatenated))
        x3 = F.leaky_relu(self.linear5(x3))
        x3 = F.leaky_relu(self.linear6(x3))
        x3 = F.leaky_relu(self.linear7(x3))

        # Convolution with channel attention
        # x3 needs to be resized due to the difference in sizes of spatial and non-spatial obs spaces
        x4 = self.conv_ch((x1_cat, x3))
        x4_flatten = x4.flatten(start_dim=1)
        x5 = self.linear_actor(x3)
        x5_flatten = x5.flatten(start_dim=1)
        actor = torch.cat((x4_flatten, x5_flatten), dim=1)

        # Output streams
        value = self.linear_out(x3)
        actor = value + (actor - actor.mean())
         #value = self.critic(x7)
        # actor = self.actor(x6)

        # return value, policy
        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask, prev_actions=None):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        action_probs = torch.nan_to_num(action_probs)
        action_probs = torch.nn.functional.relu(action_probs, inplace=True)
        try:
            actions = action_probs.multinomial(1)
            # In rare cases, multinomial can  sample an action with p=0, to avoid that we use try except
            for i, action in enumerate(actions):
                correct_action = action
                while not action_mask[i][correct_action]:
                    correct_action = action_probs[i].multinomial(1)
                actions[i] = correct_action
        except:
            # if multinomial fails then actions from the previous step are taken instead of calculating new ones
            actions = prev_actions

        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        # actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.).to(next(self.parameters()).device), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # actions = values + (actions - actions.float().mean())  # todo check if this works as duelling dqn
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            if len(action_mask.shape) == 1:
                action_mask = torch.reshape(action_mask, (1, -1))
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs

    def get_action_log_probs(self, spatial_input, non_spatial_input, action_mask=None):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        log_probs = F.log_softmax(actions, dim=1)
        return values, log_probs

