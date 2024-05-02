import numpy as np
import ray
import torch

import botbowl
from botbowl.ai.env import EnvConf

from reinforced_enemy.reinforced_agent import make_env, ConfigParams  # todo Make specific Conffig Params for DQN
from network import CNNPolicy


@ray.remote
class BaddieBotActor(botbowl.Agent):
    BOT_ID = 'Baddie'

    def __init__(self, name='BaddieBot', env_conf: EnvConf = EnvConf(size=11, pathfinding=True),
                 model_path=ConfigParams.model_path.value):
        super().__init__(name)

        self.action_queue = []
        self.gamma = ConfigParams.gamma.value
        # todo make variables below configurable
        self.worker_buffer_size = 1000
        self.eps_greedy = 0.3
        self.eps_decay = 0.95

        self.env = botbowl.BotBowlEnv(env_conf)
        self.env.reset()

        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_shape = spatial_obs.shape
        num_non_spatial = non_spatial_obs.shape[0]
        num_actions = len(action_mask)

        self.model = CNNPolicy(spatial_shape, num_non_spatial)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # activate evaluation mode
        self.model.to(self.device)

    def new_game(self, game, team):
        '''
        Called when creating a new game.
        '''
        self.own_team = team
        self.opp_team = game.get_opp_team(team)
        self.is_home = team == game.state.home_team

    def sample(self, state):
        self.eps_greedy = self.eps_greedy * self.eps_decay
        if np.random.randn() < self.eps_greedy:
            return self.env.action_space.sample()

        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_obs = torch.from_numpy(np.stack(spatial_obs)[np.newaxis]).float().to(self.device)
        non_spatial_obs = torch.from_numpy(np.stack(non_spatial_obs)[np.newaxis]).float().to(self.device)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        _, actions = self.model.act(spatial_obs, non_spatial_obs, action_mask)

        return actions

    def act(self, game):
        '''
        Called for every game step in order to determine the action to take.
        '''
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_obs = torch.from_numpy(np.stack(spatial_obs)[np.newaxis]).float().to(self.device)
        non_spatial_obs = torch.from_numpy(np.stack(non_spatial_obs)[np.newaxis]).float().to(self.device)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        _, actions = self.model.act(spatial_obs, non_spatial_obs, action_mask)
        # Create action from output
        action_idx = actions[0]

        action_objects = self.env._compute_action(action_idx.cpu().numpy()[0], flip=not self.is_home)

        # Return action to the framework
        self.action_queue = action_objects
        return self.action_queue.pop(0)

    def end_game(self, game):
        '''
        Called when the game ends.
        '''
        winner = game.get_winning_team()
        if winner is None:
            print("It's a draw")
        elif winner == self.own_team:
            print(f'I ({self.name}) won.')
            print(self.own_team.state.score, '-', self.opp_team.state.score)
        else:
            print(f'I ({self.name}) lost.')
            print(self.own_team.state.score, '-', self.opp_team.state.score)

    @staticmethod
    def register_bot(path=ConfigParams.model_path.value):
        """
        Adds the bot to the registered bots if not already done.
        """
        if BaddieBotActor.BOT_ID.lower() not in botbowl.list_bots():
            botbowl.register_bot(BaddieBotActor.BOT_ID, lambda name: BaddieBotActor(name=name, model_path=path))