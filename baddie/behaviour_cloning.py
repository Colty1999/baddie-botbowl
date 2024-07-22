import os
import re
import time

import numpy as np
from joblib import Parallel, delayed

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from botbowl import EnvConf, BotBowlEnv, register_bot
import botbowl

from network import CNNPolicy, ConfigParams
from actor import BaddieBotActor

from Data.generator import get_scripted_dataset, scripted_data_path
from Data.scripted_bot import ScriptedBot

import venv


def split_dataset(dataset, train_percentage):
    split_index = int(len(dataset['X_spatial']) * train_percentage)

    def convert_spatial(plain_spatial):
        n_spatial = len(plain_spatial)
        spatial_obs = torch.stack(plain_spatial)
        return torch.reshape(spatial_obs, (n_spatial, 44, 17, 28))  # TODO: dynamically set this based on the env to be usable for smaller boards

    def convert_non_spatial(plain_non_spatial):
        n_non_spatial = len(plain_non_spatial)
        non_spatial_obs = torch.stack(plain_non_spatial)
        return torch.reshape(non_spatial_obs, (n_non_spatial, 1, 115))

    def convert_actions(plain_actions):
        actions = torch.stack(plain_actions)
        return torch.flatten(actions).long()

    X_spatial_train = convert_spatial(dataset['X_spatial'][:split_index])
    X_non_spatial_train = convert_non_spatial(dataset['X_non_spatial'][:split_index])
    Y_train = convert_actions(dataset['Y'][:split_index])
    dataset_train = torch.utils.data.TensorDataset(X_spatial_train, X_non_spatial_train, Y_train)

    X_spatial_test = convert_spatial(dataset['X_spatial'][split_index:])
    X_non_spatial_test = convert_non_spatial(dataset['X_non_spatial'][split_index:])
    Y_test = convert_actions(dataset['Y'][split_index:])
    dataset_test = torch.utils.data.TensorDataset(X_spatial_test, X_non_spatial_test, Y_test)

    return dataset_train, dataset_test


def setup_model(dataset, batch_size=100, num_workers=1, load_model=False, load_path=ConfigParams.model_path.value):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Device ', device)

    env_conf = EnvConf(size=11, pathfinding=True)
    env = BotBowlEnv(env_conf=env_conf)
    env.reset()
    spat_obs, non_spat_obs, action_mask = env.get_state()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]

    if load_model:
        # model = CNNPolicy(spatial_obs_space, non_spatial_obs_space, filename=load_path)
        model = torch.load(load_path)
        model.eval()
    else:
        model = CNNPolicy(spatial_obs_space, non_spatial_obs_space)

    # train on GPU if possible
    model.to(device)

    data_loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
    return model, data_loader, device


def get_dataloader(dataset, batch_size=100, num_workers=1):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers, persistent_workers=True)


def train(model, device, dataloader_train, dataloader_valid=None, criterion=nn.NLLLoss(), n_epochs=30, save_path=None):
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    model = model

    optimizer = optim.RAdam(model.parameters(), lr=0.0001)  # optim.RAdam(model.parameters(), lr=0.0001)  #

    for epoch in range(n_epochs):
        print('Epoch', epoch)
        start_time = time.time()
        train_loss = 0

        num_correct_train = 0
        num_correct_valid = 0

        model.train()

        try:
            if dataloader_train is not None:
                for data in tqdm(dataloader_train):

                    spatial_obs, non_spatial_obs, action_mask, actions = data  # Todo: check if all steps are properly loaded
                    spatial_obs = spatial_obs.to(device)
                    non_spatial_obs = non_spatial_obs.to(device)
                    actions = actions.type(torch.LongTensor)
                    actions = actions.flatten().to(device)
                     # actions.to(torch.float)  # TODO: make sure actions are saved as float instead of int

                    optimizer.zero_grad()
                    _, action_log_probs, = model.get_action_log_probs(spatial_obs, non_spatial_obs)  # , action_mask)

                    loss = criterion(action_log_probs, actions)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    # calculate number of correct predictions
                    predicted_actions = np.argmax(action_log_probs.detach().cpu(), axis=1)
                    num_correct_train += np.sum(predicted_actions.numpy() == actions.detach().cpu().numpy())
        except KeyboardInterrupt:
            print('Training cancelled')
            val = input('Save model? (y/n)')
            if val == 'y':
                break
            else:
                return
        except BrokenPipeError:
            pass
        except Exception as e:
            stop = 1

        # calculate average loss
        train_loss /= len(dataloader_train)
        training_losses.append(train_loss)

        # calculate accuracy
        train_accuracy = num_correct_train / len(dataloader_train.dataset)
        training_accuracies.append(train_accuracy)

        if dataloader_valid is not None:
            valid_loss = 0
            for data in tqdm(dataloader_valid):
                spatial_obs, non_spatial_obs, action_mask, actions = data
                spatial_obs = spatial_obs.to(device)
                non_spatial_obs = non_spatial_obs.to(device)
                actions = actions.type(torch.LongTensor)
                actions = actions.flatten().to(device)

                _, action_log_probs, = model.get_action_log_probs(spatial_obs, non_spatial_obs)#, action_mask)
                valid_loss += criterion(action_log_probs, actions).item()

                # calculate number of correct predictions
                predicted_actions = np.argmax(action_log_probs.detach().cpu(), axis=1)
                num_correct_valid += np.sum(predicted_actions.numpy() == actions.detach().cpu().numpy())

            valid_loss /= len(dataloader_valid)
            validation_losses.append(valid_loss)

            valid_accuracy = num_correct_valid / len(dataloader_valid.dataset)
            validation_accuracies.append(valid_accuracy)

        # calculate time for each epoch
        delta = time.time() - start_time

        # print training / validation metrics
        if dataloader_valid is not None:
            print('Epoch:', epoch, 'took', round(delta, 3), 'secs', '----Training loss:', round(train_loss, 7), '----Validation loss:', round(valid_loss, 7))
            print('----Training accuracy:', round(train_accuracy, 3), '----Validation accuracy:', round(valid_accuracy, 3),
                  '----Predictions train:', num_correct_train, '/', len(dataloader_train.dataset),
                  '----Predictions val:', num_correct_valid, '/', len(dataloader_valid.dataset)
                  )
        else:
            print('Epoch:', epoch, 'took', round(delta, 3), 'secs', '----Training loss:', round(train_loss, 7))
            print('----Training accuracy:', round(train_accuracy, 7))

        # save model after training
        if save_path is not None:
            dir_path = os.path.dirname(save_path)
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            torch.save(model, save_path)
            print('Saved model at', save_path)

    # save loss plot
    plt.plot(training_losses, label='Training loss')
    if dataloader_valid is not None:
        plt.plot(validation_losses, label='Validation loss')
    plt.legend()
    plt.savefig('./losses.png')
    plt.clf()

    # save accuracy plot
    plt.plot(training_accuracies, label='Training accuracy')
    if dataloader_valid is not None:
        plt.plot(validation_accuracies, label='Validation accuracy')
    plt.legend()
    plt.savefig('./accuracies.png')

    if dataloader_valid is not None:
        return training_losses, validation_losses
    else:
        return training_losses


def evaluate(agent_1, agent_2, num_games=1, num_jobs=1):
    wins, draws, tds_p1_mean, tds_p2_mean, _, _ = Parallel(n_jobs=num_jobs)(delayed(evaluation_games())(agent_1, agent_2, num_games//num_jobs) for _ in tqdm(range(num_games), desc='Playing games'))

    wins = np.sum(wins)
    draws = np.sum(draws)
    tds_p1_mean = np.mean(tds_p1_mean)
    tds_p2_mean = np.mean(tds_p2_mean)

    print(f"w/d/l: {wins}/{draws}/{num_games-wins-draws}")
    print(f"TD P1/P2: {tds_p1_mean}/{tds_p2_mean}")


def evaluation_games(agent_path, adversary_agent, num_games):

    def _make_my_a2c_bot(name, env_size=11):
        return BaddieBotActor(name=name,
                        env_conf=EnvConf(size=env_size),
                        filename=agent_path)
    register_bot('a2c-bot', _make_my_a2c_bot)

    if adversary_agent == "scripted":
        ScriptedBot.register_bot()
        enemy_bot = ScriptedBot.BOT_ID
    else:
        enemy_bot = "random"

    game_config = botbowl.load_config("bot-bowl")
    game_config.competition_mode = False
    game_config.pathfinding_enabled = True
    ruleset = botbowl.load_rule_set(game_config.ruleset, all_rules=False)
    arena = botbowl.load_arena(game_config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    wins = 0
    draws = 0
    tds_p1 = []
    tds_p2 = []

    for i in tqdm(range(num_games), desc=f'Playing games vs {enemy_bot}'):
        is_home = i % 2 == 0

        if is_home:
            home_agent = botbowl.make_bot('a2c-bot')
            away_agent = botbowl.make_bot(enemy_bot)
        else:
            home_agent = botbowl.make_bot(enemy_bot)
            away_agent = botbowl.make_bot('a2c-bot')
        game = botbowl.Game(i, home, away, home_agent, away_agent, game_config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        game.init()

        winner = game.get_winner()
        if winner is None:
            draws += 1
        elif winner == home_agent and is_home:
            wins += 1
        elif winner == away_agent and not is_home:
            wins += 1

        if is_home:
            tds_p1.append(game.get_agent_team(home_agent).state.score)
            tds_p2.append(game.get_agent_team(away_agent).state.score)
        else:
            tds_p1.append(game.get_agent_team(away_agent).state.score)
            tds_p2.append(game.get_agent_team(home_agent).state.score)

    tds_p1_mean = np.mean(tds_p1)
    tds_p2_mean = np.mean(tds_p2)

    return wins, draws, tds_p1_mean, tds_p2_mean, tds_p1, tds_p2


if __name__ == '__main__':
    print(torch.cuda.is_available())
    torch.set_num_threads(18)
    print(torch.get_num_threads())
    torch.cuda.empty_cache()


    train_dataset, valid_dataset = get_scripted_dataset(training_percentage=0.7, cache_data=True)
    model, dataloader_train, device = setup_model(train_dataset, batch_size=256, num_workers=2, load_model=False)
    dataloader_valid = get_dataloader(valid_dataset, num_workers=1)
    try:
        training_losses, validation_losses = train(model, device, dataloader_train, dataloader_valid, n_epochs=15, save_path=ConfigParams.model_path.value)
    except Exception as e:
        stop = 1

    # evaluate(ConfigParams.model_path.value, 'random')
