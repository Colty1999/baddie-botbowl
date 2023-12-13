from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl
import botbowl.web.server as server
from botbowl.ai.env import EnvConf, BotBowlEnv
from botbowl.ai.layers import *
from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper
from env import A2C_Reward


class ConfigParams(Enum):
    num_steps = 5000000
    num_processes = 8
    steps_per_update = 20
    learning_rate = 0.01
    gamma = 0.99
    entropy_coef = 0.15
    value_loss_coef = 0.3
    max_grad_norm = 0.05
    log_interval = 50
    save_interval = 100
    reset_steps = 5000  # The environment is reset after this many steps it gets stuck
    selfplay_window = 5
    selfplay_save_steps = int(num_steps / 10)
    selfplay_swap_steps = selfplay_save_steps
    num_hidden_nodes = 256
    ppcg = True
    env_size = 11  # Options are 1,3,5,7,11
    env_name = f"botbowl-{env_size}"
    env_conf = EnvConf(size=env_size, pathfinding=False)
    selfplay = True
    exp_id = str(uuid.uuid1())
    model_dir = f"models/{env_name}/"


# Architecture
model_name = 'bd0e2863-8eaf-11ee-ae4f-d45d641e693d'
env_name = f'botbowl-11'
model_filename = f"models/{env_name}/{model_name}.nn"
log_filename = f"logs/{env_name}/{env_name}.dat"


def make_env(env_conf, ppcg=ConfigParams.ppcg.value):
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    # env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env


env = make_env(ConfigParams.env_conf.value)
spat_obs, non_spat_obs, action_mask = env.reset()
spatial_obs_space = spat_obs.shape
non_spatial_obs_space = non_spat_obs.shape[0]
action_space = len(action_mask)
del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down


class CNNPolicy(nn.Module):

    def __init__(self, spatial_shape=spatial_obs_space, non_spatial_inputs=non_spatial_obs_space,
                 hidden_nodes=ConfigParams.num_hidden_nodes.value, kernels=[32, 64], actions=action_space):
        super(CNNPolicy, self).__init__()

        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_shape[0], out_channels=kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding=1)

        # Non-spatial input stream
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # Linear layers
        stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        stream_size += hidden_nodes
        self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(hidden_nodes, actions)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.actor.weight.data.mul_(relu_gain)
        self.critic.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through two convolutional layers

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
        #x2 = self.linear2(x2)
        #x2 = F.relu(x2)

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


class A2CAgent(Agent):
    env: BotBowlEnv

    def __init__(self, name,
                 env_conf: EnvConf,
                 scripted_func: Callable[[Game], Optional[Action]] = None,
                 filename=model_filename):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)

        self.scripted_func = scripted_func
        self.action_queue = []

        # MODEL
        # self.policy = CNNPolicy() # For testing games
        # self.policy.load_state_dict(torch.load(filename))
        self.policy = torch.load(filename)
        self.policy.eval()
        self.end_setup = False

    def new_game(self, game, team):
        pass

    @staticmethod
    def _update_obs(array: np.ndarray):
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def act(self, game):
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        if self.scripted_func is not None:
            scripted_action = self.scripted_func(game)
            if scripted_action is not None:
                return scripted_action

        self.env.game = game

        spatial_obs, non_spatial_obs, action_mask = map(A2CAgent._update_obs, self.env.get_state())
        non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=0)

        _, actions = self.policy.act(
            Variable(spatial_obs.float()),
            Variable(non_spatial_obs.float()),
            Variable(action_mask))

        action_idx = actions[0]
        action_objects = self.env._compute_action(action_idx)

        self.action_queue = action_objects
        return self.action_queue.pop(0)

    def end_game(self, game):
        pass


def main():
    # Register the bot to the framework
    def _make_my_a2c_bot(name, env_size=11):
        return A2CAgent(name=name,
                        env_conf=EnvConf(size=env_size),
                        # scripted_func=a2c_scripted_actions,
                        filename=model_filename)
    botbowl.register_bot('my-a2c-bot', _make_my_a2c_bot)

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
    n = 10
    is_home = True
    tds_away = 0
    tds_home = 0
    for i in range(n):
        host = "127.0.0.1"
        server.start_server(host=host, debug=True, use_reloader=False, port=1234)
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
