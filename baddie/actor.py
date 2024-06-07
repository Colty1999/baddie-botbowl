import numpy as np
import ray
import torch
from torch.autograd import Variable
from typing import Callable

import botbowl
from botbowl.ai.env import EnvConf, BotBowlEnv
from botbowl.ai.layers import *
import botbowl.web.server as server

from baddie.network import CNNPolicy, ConfigParams
from Data.scripted_bot import ScriptedBot


# @ray.remote
class BaddieBotActor(botbowl.Agent):
    env: BotBowlEnv
    BOT_ID = 'Baddie'

    def __init__(self, name='BaddieBot', env_conf: EnvConf = EnvConf(size=11, pathfinding=True),
                 scripted_func: Callable[[Game], Optional[Action]] = None,
                 filename=ConfigParams.model_path.value):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)

        self.scripted_func = scripted_func
        self.action_queue = []

        # MODEL
        # self.policy = CNNPolicy()  # For testing games
        # self.policy.load_state_dict(torch.load(filename))
        self.device = ConfigParams.device.value
        self.policy = torch.load(filename)
        self.policy.eval()
        self.policy.to(self.device)
        self.end_setup = False

    def new_game(self, game, team):
        '''
        Called when creating a new game.
        '''
        self.own_team = team
        self.opp_team = game.get_opp_team(team)
        self.is_home = team == game.state.home_team

    @staticmethod
    def _update_obs(array: np.ndarray):
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

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
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        if self.scripted_func is not None:
            scripted_action = self.scripted_func(game)
            if scripted_action is not None:
                return scripted_action

        self.env.game = game

        spatial_obs, non_spatial_obs, action_mask = map(BaddieBotActor._update_obs, self.env.get_state())
        spatial_obs = spatial_obs.to(ConfigParams.device.value)
        non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=0)
        non_spatial_obs = non_spatial_obs.to(ConfigParams.device.value)
        action_mask = action_mask.to(ConfigParams.device.value)

        _, actions = self.policy.act(
            Variable(spatial_obs.float()),
            Variable(non_spatial_obs.float()),
            Variable(action_mask))

        action_idx = actions[0]
        action_objects = self.env._compute_action(action_idx)

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

def main():
    # Register the bot to the framework
    def _make_my_a2c_bot(name, env_size=11):
        return BaddieBotActor(name=name,
                        env_conf=EnvConf(size=env_size),
                        # scripted_func=a2c_scripted_actions,
                        filename=ConfigParams.agent_path.value)
    botbowl.register_bot('my-a2c-bot', _make_my_a2c_bot)
    botbowl.register_bot('scripted', ScriptedBot)

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 10 games
    wins = 0
    draws = 0
    n = 25
    is_home = True
    tds_away = 0
    tds_home = 0
    for i in range(n):
        # host = "127.0.0.1"
        # server.start_server(host=host, debug=True, use_reloader=False, port=1234)
        if is_home:
            away_agent = botbowl.make_bot('random')
            home_agent = botbowl.make_bot('my-a2c-bot')
        else:
            away_agent = botbowl.make_bot('my-a2c-bot')
            home_agent = botbowl.make_bot("random")
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")

        winner = game.get_winner()
        print(winner)
        if winner is None:
            draws += 1
        elif winner == home_agent and is_home:
            wins += 1
        elif winner == away_agent and not is_home:
            wins += 1

        tds_home += game.get_agent_team(home_agent).state.score
        tds_away += game.get_agent_team(away_agent).state.score

    print(f"Home/Draws/Away: {wins}/{draws}/{n-wins-draws}")
    print(f"Home TDs per game: {tds_home/n}")
    print(f"Away TDs per game: {tds_away/n}")


if __name__ == "__main__":
    main()
