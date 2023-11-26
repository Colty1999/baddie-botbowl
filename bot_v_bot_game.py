from enum import Enum
import time as time

from botbowl.examples.scripted_bot_example import *
from reinforced_enemy.reinforced_agent import *  # Please do import all of your bot prerequisites
import botbowl as botbowl


class GameParams(Enum):
    no_games = 10,
    home_bot = "scripted",
    away_bot = "scripted",


def make_a2c_bot(name, env_size, a2c_scripted_actions=None, model_filename=None):
    return A2CAgent(name=name,
                    env_conf=EnvConf(size=env_size),
                    scripted_func=a2c_scripted_actions,
                    filename=model_filename)


def main():
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = True
    config.debug_mode = False
    ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)

    home_agent = botbowl.make_bot(GameParams.home_bot.value)
    home_agent.name = "Bot 1"
    away_agent = botbowl.make_bot(GameParams.away_bot.value)
    away_agent.name = "Bot 2"

    for i in range(GameParams.no_games.value):
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True
        print("Starting game", (i + 1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)


if __name__ == "__main__":
    main()
