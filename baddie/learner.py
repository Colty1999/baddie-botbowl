''' https://github.com/cyoon1729/distributedRL/blob/master/common/abstract/learner.py '''

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import deque
import numpy as np
import pyarrow as pa
import zmq
from zmq.sugar.stopwatch import Stopwatch

import ray
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm

from reinforced_enemy.reinforced_agent import make_env, ConfigParams  # todo Make specific Conffig Params for DQN


class Learner(ABC):
    def __init__(self, model):
        self.device = ConfigParams.device.value
        self.brain = deepcopy(model)
        self.replay_data_queue = deque(maxlen=1000)

        # unpack communication configs
        self.param_update_interval = self.cfg["param_update_interval"]
        self.repreq_port = 5556
        self.pubsub_port = 5555

        # initialize zmq sockets
        print("[Learner]: initializing sockets..")
        self.initialize_sockets()

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def learning_step(self, data: tuple):
        pass

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return model params for synchronization"""
        pass

    def params_to_numpy(self, model):
        params = []
        new_model = deepcopy(model)
        state_dict = new_model.cpu().state_dict()
        for param in list(state_dict):
            params.append(state_dict[param].numpy())
        return params

    def initialize_sockets(self):
        # For sending new params to workers
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.pubsub_port}")

        # For receiving batch from, sending new priorities to Buffer # write another with PUSH/PULL for non PER version
        context = zmq.Context()
        self.rep_socket = context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://127.0.0.1:{self.repreq_port}")

    def publish_params(self, new_params: np.ndarray):
        new_params_id = pa.serialize(new_params).to_buffer()
        self.pub_socket.send(new_params_id)

    def recv_replay_data_(self):
        replay_data_id = self.rep_socket.recv()
        replay_data = pa.deserialize(replay_data_id)
        self.replay_data_queue.append(replay_data)

    def send_new_priorities(self, idxes: np.ndarray, priorities: np.ndarray):
        new_priors = [idxes, priorities]
        new_priors_id = pa.serialize(new_priors).to_buffer()
        self.rep_socket.send(new_priors_id)

    def run(self):
        tracker = Stopwatch()
        self.update_step = 0
        while True:
            self.recv_replay_data_()
            replay_data = self.replay_data_queue.pop()

            for _ in range(self.cfg["multiple_updates"]):
                step_info, idxes, priorities = self.learning_step(replay_data)

            self.update_step = self.update_step + 1

            self.send_new_priorities(idxes, priorities)

            if self.update_step % self.param_update_interval == 0:
                params = self.get_params()
                self.publish_params(params)


@ray.remote(num_gpus=1)
class DQNLearner(Learner):
    def __init__(self, model):
        super().__init__(model)
        self.num_step = ConfigParams.num_steps.value
        self.gamma = ConfigParams.gamma.value
        self.tau = ConfigParams.tau.value
        self.gradient_clip = ConfigParams.gradient_clip.value
        self.q_regularization = self.ConfigParams.q_regularization.value
        self.network = self.model
        self.network.to(self.device)
        self.target_network = self.model
        self.target_network.to(self.device)
        self.network_optimizer = torch.optim.Adam(
            self.network.parameters(), lr=ConfigParams.learning_rate.value
        )

    def write_log(self):
        pass # Todo include logging (use Tensorboard?)

    def learning_step(self, data: tuple):
        states, actions, rewards, next_states, dones, weights, idxes = data

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        states.cuda(non_blocking=True)
        actions.cuda(non_blocking=True)
        rewards.cuda(non_blocking=True)
        next_states.cuda(non_blocking=True)
        dones.cuda(non_blocking=True)

        curr_q = self.network.forward(states).gather(1, actions.unsqueeze(1))
        bootstrap_q = torch.max(self.target_network.forward(next_states), 1)[0]

        bootstrap_q = bootstrap_q.view(bootstrap_q.size(0), 1)
        target_q = rewards + (1 - dones) * self.gamma ** self.num_step * bootstrap_q
        weights = torch.FloatTensor(weights).to(self.device)
        weights.cuda(non_blocking=True)
        weights = weights.mean()

        q_loss = (
            weights * F.smooth_l1_loss(curr_q, target_q.detach(), reduction="none")
        ).mean()
        dqn_reg = torch.norm(q_loss, 2).mean() * self.q_regularization
        loss = q_loss + dqn_reg

        self.network_optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm(self.network.parameters(), self.gradient_clip) todo Check approporiate gradient clip
        self.network_optimizer.step()

        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        new_priorities = torch.abs(target_q - curr_q).detach().view(-1)
        new_priorities = torch.clamp(new_priorities, min=1e-8)
        new_priorities = new_priorities.cpu().numpy().tolist()

        return loss, idxes, new_priorities

    def get_params(self):
        model = deepcopy(self.network)
        model = model.cpu()

        return self.params_to_numpy(self.network)