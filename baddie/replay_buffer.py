import random
import numpy as np

import torch
import pyarrow as pa
import ray
import zmq

from utils.segtree import MinSegmentTree, SegmentTree, SumSegmentTree
from network import ConfigParams, make_env

#todo deal with code repetitions
env = make_env(ConfigParams.env_conf.value)
spat_obs, non_spat_obs, action_mask = env.reset()
spatial_obs_space = spat_obs.shape
non_spatial_obs_space = non_spat_obs.shape[0]
action_space = len(action_mask)


class ReplayBuffer(object):
    def __init__(self, size=ConfigParams.steps_per_update.value,
                 spatial=spatial_obs_space, non_spatial=(1, non_spatial_obs_space), action_space=8116):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        action_shape = 1
        self.spatial_obs = torch.zeros(ConfigParams.steps_per_update.value + 1,
                                       ConfigParams.num_processes.value, *spatial)
        self.non_spatial_obs = torch.zeros(ConfigParams.steps_per_update.value + 1,
                                           ConfigParams.num_processes.value, *non_spatial)
        self.rewards = torch.zeros(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value, 1)
        self.returns = torch.zeros(ConfigParams.steps_per_update.value + 1, ConfigParams.num_processes.value, 1)
        self.actions = torch.zeros(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(ConfigParams.steps_per_update.value + 1, ConfigParams.num_processes.value,
                                1)  # torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(ConfigParams.steps_per_update.value + 1,
                                        ConfigParams.num_processes.value, action_space, dtype=torch.bool)

    def __len__(self):
        return len(self._storage)

    def to(self, device):
        self._storage = [item.to(device) for item in self._storage]
        self.spatial_obs = self.spatial_obs.to(device)
        self.non_spatial_obs = self.non_spatial_obs.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.action_masks = self.action_masks.to(device)

    def add(self, step, spatial_obs, non_spatial_obs, next_spatial_obs, next_non_spatial_obs, action, reward, mask, action_masks):
        self.spatial_obs[step + 1].copy_(torch.from_numpy(spatial_obs).float())
        self.non_spatial_obs[step + 1].copy_(torch.from_numpy(np.expand_dims(non_spatial_obs, axis=1)).float())
        self.actions[step].copy_(action)
        self.rewards[step].copy_(torch.from_numpy(np.expand_dims(reward, 1)).float())
        self.masks[step].copy_(mask)
        self.action_masks[step + 1].copy_(torch.from_numpy(action_masks))
        # data = (obs_t, action, reward, obs_tp1, done)
        data = (spatial_obs, non_spatial_obs, next_spatial_obs, next_non_spatial_obs, action, reward, mask, action_masks)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        spatial_obs, non_spatial_obs, next_spatial_obs, next_non_spatial_obs, actions, rewards, masks, action_masks = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            spatial, non_spatial, next_spatial, next_non_spatial, action, reward, mask, action_mask = data
            spatial_obs.append(np.array(spatial, copy=False))
            non_spatial_obs.append(np.array(non_spatial, copy=False))
            next_spatial_obs.append(np.array(next_spatial, copy=False))
            next_non_spatial_obs.append(np.array(next_non_spatial, copy=False))
            actions.append(np.array(action.to("cpu"), copy=False))
            rewards.append(reward)
            masks.append(np.array(mask, copy=False))
            action_masks.append(np.array(action_mask, copy=False))
        return (
            np.array(spatial_obs),
            np.array(non_spatial_obs),
            np.array(next_spatial_obs),
            np.array(next_non_spatial_obs),
            np.array(actions),
            np.array(rewards),
            np.array(masks),
            np.array(action_mask)
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size=ConfigParams.steps_per_update.value, alpha=0.6):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


@ray.remote
class PrioritizedReplayBufferHelper(object):
    def __init__(self):

        # todo make adjustable
        self.max_num_updates = 100000
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_end = 1.0
        self.buffer_max_size = 1000000
        self.priority_beta_increment = (
            self.priority_beta_end - self.priority_beta
        ) / self.max_num_updates

        self.batch_size = 512

        self.buffer = PrioritizedReplayBuffer(
            size=self.buffer_max_size, alpha=self.priority_alpha
        )

        # unpack communication configs
        self.repreq_port = 5556
        self.pullpush_port = 5557

        # initialize zmq sockets
        print("[Buffer]: initializing sockets..")
        self.initialize_sockets()

    def initialize_sockets(self):
        # for sending batch to learner and retrieving new priorities
        context = zmq.Context()
        self.rep_socket = context.socket(zmq.REQ)
        self.rep_socket.connect(f"tcp://127.0.0.1:{self.repreq_port}")

        # for receiving replay data from workers
        context = zmq.Context()
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.pullpush_port}")

    def send_batch_recv_priors(self):
        # send batch and request priorities (blocking recv)
        batch = self.buffer.sample(self.batch_size, self.priority_beta)
        batch_id = pa.serialize(batch).to_buffer()
        self.rep_socket.send(batch_id)

        # receive and update priorities
        new_priors_id = self.rep_socket.recv()
        idxes, new_priorities = pa.deserialize(new_priors_id)
        self.buffer.update_priorities(idxes, new_priorities)

    def recv_data(self):
        new_replay_data_id = False
        try:
            new_replay_data_id = self.pull_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass

        if new_replay_data_id:
            new_replay_data = pa.deserialize(new_replay_data_id)
            for replay_data, priorities in new_replay_data:
                self.buffer.add(*replay_data)
                self.buffer.update_priorities(
                    [(self.buffer._next_idx - 1) % self.buffer._maxsize], priorities
                )

    def run(self):
        while True:
            self.recv_data()
            if len(self.buffer) > self.batch_size:
                self.send_batch_recv_priors()
            else:
                pass
