import botbowl
from Data.scripted_bot import ScriptedBot
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import random
from joblib import Parallel, delayed
from tqdm import tqdm

scripted_data_path = os.path.join(os.path.dirname(__file__), "scripted_dataset")  # os.path.join(os.getcwd(), "scripted_dataset")


class ScriptedDataGenerator:

    def __init__(self, num_games):
        """
        num_games_random = Number of games played against a random bot
        num_games_scripted = Number of games played against a scripted bot
        """
        self.num_games = num_games

        # Game variables
        self.arena = None
        self.home = None
        self.away = None
        self.game_config = None
        self.out_path = scripted_data_path

        # obtained data
        self.game_data = []

    def _create_game_env(self):
        """
        Load configurations, rules, arena and teams
        """
        ScriptedBot.register_bot()
        self.game_config = botbowl.load_config("bot-bowl")
        self.game_config.competition_mode = False
        self.game_config.pathfinding_enabled = True
        self.ruleset = botbowl.load_rule_set(self.game_config.ruleset, all_rules=False)  # We don't need all the rules
        self.arena = botbowl.load_arena(self.game_config.arena)
        self.home = botbowl.load_team_by_filename("human", self.ruleset)
        self.away = botbowl.load_team_by_filename("human", self.ruleset)

    def _play_game(self, agent_1_id, agent_2_id, i):
        self._create_game_env()
        home_agent = botbowl.make_bot(agent_1_id)
        home_agent.dump = True
        home_agent.out_path = self.out_path
        home_agent.name = f'Agent 1 ({agent_1_id})'
        away_agent = botbowl.make_bot(agent_2_id)
        away_agent.dump = True
        away_agent.out_path = self.out_path
        away_agent.name = f'Agent 2 ({agent_2_id})'
        self.game_config.debug_mode = False

        game = botbowl.Game(i, self.home, self.away, home_agent, away_agent, self.game_config, arena=self.arena,
                            ruleset=self.ruleset)
        game.config.fast_mode = True
        game.init()

    def _do_playouts(self, agent_1_id, agent_2_id, num_games):

        Parallel(n_jobs=-1)(delayed(self._play_game)(agent_1_id, agent_2_id, i) for i in tqdm(range(num_games),desc='Playing games'))

    def generate_training_data(self, bot='scripted'):

        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        # play games
        if bot in ['scripted', 'both']:
            print('Playing games vs scripted bot')
            self._do_playouts(ScriptedBot.BOT_ID, ScriptedBot.BOT_ID, self.num_games)
        if bot in ['random', 'both']:
            print('Playing games vs random bot')
            self._do_playouts(ScriptedBot.BOT_ID, 'random', self.num_games)
        print('Finished playing games')


class ScriptedDataset(Dataset):
    def __init__(self, files, cache_data=False):
        self.files = files
        self.cache_data = cache_data
        self.data = {}
        if cache_data:
            self.load_data()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.data:
            return self.data[idx]

        obs, act = torch.load(self.files[idx])
        pair = [obs['spatial_obs'], obs['non_spatial_obs'], obs['action_mask'], act]

        if self.cache_data:
            self.data[idx] = pair
        return pair

    def load_data(self):
        for i, file in tqdm(enumerate(self.files), desc='Loading data'):
            try:  # Todo: remove later
                obs, act = torch.load(file)
            except:
                print(file)
            pair = [obs['spatial_obs'], obs['non_spatial_obs'], obs['action_mask'], act]
            self.data[i] = pair


def get_scripted_dataset(paths=[scripted_data_path], training_percentage=0.9, cache_data=True):
    if not isinstance(paths, list):
        raise Exception(f'Expected a list of paths but got {paths}')
    tensor_files = []
    for path in paths:
        tensor_files.extend(list(map(lambda file: os.path.join(path, file), filter(lambda file: file.endswith('.pt'), os.listdir(path)))))
    print(f'Found {len(tensor_files)} sample files')
    random.shuffle(tensor_files)
    split_index = int(len(tensor_files) * training_percentage)
    training_files, validation_files = tensor_files[:split_index], tensor_files[split_index:]
    return ScriptedDataset(training_files, cache_data=cache_data), \
           ScriptedDataset(validation_files, cache_data=cache_data)


if __name__ == '__main__':
    generator = ScriptedDataGenerator(num_games=100)
    generator.generate_training_data(bot='random')
