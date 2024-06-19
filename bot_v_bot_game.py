from typing import Callable
import itertools
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import botbowl
from botbowl.ai.env import EnvConf, BotBowlEnv
from botbowl.ai.layers import *
from Data.scripted_bot import ScriptedBot
from examples.scripted_bot_example import *
import csv

def run_game(params, game_number, config, ruleset, arena, home, away):
    param1, param2, param3 = params
    is_home = game_number % 2 == 0
    away_agent = botbowl.make_bot('scripted')
    home_agent = botbowl.make_bot('my-script')

    if isinstance(home_agent, ScriptedBot):
        home_agent.set_variables(param1, param2, param3)
    if isinstance(away_agent, ScriptedBot):
        away_agent.set_variables(param1, param2, param3)

    game = botbowl.Game(game_number, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    game.init()

    winner = game.get_winner()
    if winner is None:
        winner_str = 'Draw'
    elif winner == home_agent and is_home:
        winner_str = 'Home'
    elif winner == away_agent and not is_home:
        winner_str = 'Away'
    else:
        winner_str = 'None'

    home_score = game.get_agent_team(home_agent).state.score
    away_score = game.get_agent_team(away_agent).state.score

    return (winner_str, home_score, away_score)

def summarize_results(params, results):
    param1, param2, param3 = params
    wins = draws = 0
    tds_home = tds_away = 0

    for result in results:
        winner_str, home_score, away_score = result
        if winner_str == 'Draw':
            draws += 1
        elif winner_str == 'Home' or winner_str == 'None':
            wins += 1
        elif winner_str == 'Away':
            pass  # Do nothing for now

        tds_home += home_score
        tds_away += away_score

    return (param1, param2, param3, wins, draws, 200-wins-draws, tds_home / 200, tds_away / 200)

def main():
    # Register the bot to the framework
    botbowl.register_bot('my-script', ScriptedBot)

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

    # Open CSV file for writing
    with open('game_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Param1', 'Param2', 'Param3', 'Wins', 'Draws', 'Losses', 'Home TDs per game', 'Away TDs per game'])

        # Generate all possible combinations of parameters
        param_range = [round(x * 0.1, 1) for x in range(11)]  # [0.0, 0.1, ..., 1.0]
        param_combinations = list(itertools.product(param_range, repeat=3))

        # Use multiprocessing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        for params in param_combinations:
            tasks = [(params, i, config, ruleset, arena, home, away) for i in range(200)]
            results = pool.starmap(run_game, tasks)

            # Summarize results after each batch of 1000 games
            summary = summarize_results(params, results)
            writer.writerow(summary)
            file.flush()

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
