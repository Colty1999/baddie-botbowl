from multiprocessing import Process, Pipe
from typing import Tuple, Iterable

import torch

import botbowl
from botbowl.ai.env import BotBowlWrapper, PPCGWrapper
from botbowl.ai.layers import *

# When using A2CAgent, remember to set exclude_pathfinding_moves = False if you train with pathfinding_enabled = True


class Memory(object):
    def __init__(self, steps_per_update, num_processes, spatial_obs_shape, non_spatial_obs_shape, action_space):
        self.spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, num_processes, 1)
        self.returns = torch.zeros(steps_per_update + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(steps_per_update + 1, num_processes, action_space, dtype=torch.bool)

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.action_masks = self.action_masks.cuda()

    def insert(self, step, spatial_obs, non_spatial_obs, action, reward, mask, action_masks):
        self.spatial_obs[step + 1].copy_(torch.from_numpy(spatial_obs).float())
        self.non_spatial_obs[step + 1].copy_(torch.from_numpy(np.expand_dims(non_spatial_obs, axis=1)).float())
        self.actions[step].copy_(action)
        self.rewards[step].copy_(torch.from_numpy(np.expand_dims(reward, 1)).float())
        self.masks[step].copy_(mask)
        self.action_masks[step + 1].copy_(torch.from_numpy(action_masks))

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.shape[0])):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]


def worker(remote, parent_remote, env: BotBowlWrapper, worker_id):
    parent_remote.close()
    reset_steps = 5000  # The environment is reset after this many steps it gets stuck
    steps = 0
    tds = 0
    tds_opp = 0
    next_opp = botbowl.make_bot('random')

    ppcg_wrapper: Optional[PPCGWrapper] = env.get_wrapper_with_type(PPCGWrapper)

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action, dif = data[0], data[1]
            if ppcg_wrapper is not None:
                ppcg_wrapper.difficulty = dif

            (spatial_obs, non_spatial_obs, action_mask), reward, done, info = env.step(action)

            game = env.game
            tds_scored = game.state.home_team.state.score - tds
            tds_opp_scored = game.state.away_team.state.score - tds_opp
            tds = game.state.home_team.state.score
            tds_opp = game.state.away_team.state.score

            if done or steps >= reset_steps:
                # If we get stuck or something - reset the environment
                if steps >= reset_steps:
                    print("Max. number of steps exceeded! Consider increasing the number.")
                done = True
                env.root_env.away_agent = next_opp
                spatial_obs, non_spatial_obs, action_mask = env.reset()
                steps = 0
                tds = 0
                tds_opp = 0
            remote.send((spatial_obs, non_spatial_obs, action_mask, reward, tds_scored, tds_opp_scored, done))

        elif command == 'reset':
            steps = 0
            tds = 0
            tds_opp = 0
            env.root_env.away_agent = next_opp
            spatial_obs, non_spatial_obs, action_mask = env.reset()
            remote.send((spatial_obs, non_spatial_obs, action_mask, 0.0, 0, 0, False))

        elif command == 'swap':
            next_opp = data
        elif command == 'close':
            break


class VecEnv:
    def __init__(self, envs):
        """
        envs: list of botbowl environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions: Iterable[int], difficulty=1.0) -> Tuple[np.ndarray, ...]:
        """
        Takes one step in each environment, returns the results as stacked numpy arrays
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action, difficulty]))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def reset(self, difficulty=1.0):
        for remote in self.remotes:
            remote.send(('reset', difficulty))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def swap(self, agent):
        for remote in self.remotes:
            remote.send(('swap', agent))

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)


