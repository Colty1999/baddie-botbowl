import os
import re
import time

import numpy as np

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from botbowl import EnvConf, BotBowlEnv

from reinforced_agent import CNNPolicy, ConfigParams
# from evaluation import evaluate_bot

from Data.generator import get_scripted_dataset, scripted_data_path


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


def setup_model(dataset, batch_size=100, num_workers=1):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Device ', device)

    env_conf = EnvConf(size=11, pathfinding=True)
    env = BotBowlEnv(env_conf=env_conf)
    env.reset()
    spat_obs, non_spat_obs, action_mask = env.get_state()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]

    model = CNNPolicy(spatial_obs_space, non_spatial_obs_space)

    # train on GPU if possible
    model.to(device)

    data_loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
    return model, data_loader, device


def get_dataloader(dataset, batch_size=100, num_workers=1):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)


def train(model, device, dataloader_train, dataloader_valid=None, criterion=nn.NLLLoss(), n_epochs=10, save_path=None):
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    optimizer = optim.RAdam(model.parameters(), lr=0.0001)

    for epoch in range(n_epochs):
        print('Epoch', epoch)
        start_time = time.time()
        train_loss = 0

        num_correct_train = 0
        num_correct_valid = 0

        model.train()

        try:
            for data in tqdm(dataloader_train):

                spatial_obs, non_spatial_obs, action_mask, actions = data  # Todo: check if all steps are properly loaded
                spatial_obs = spatial_obs.to(device)
                non_spatial_obs = non_spatial_obs.to(device)
                actions = actions.type(torch.LongTensor)
                actions = actions.flatten().to(device)
                 # actions.to(torch.float)  # TODO: make sure actions are saved as float instead of int

                optimizer.zero_grad()
                _, action_log_probs, = model.get_action_probs(spatial_obs, non_spatial_obs, action_mask)

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

        # calculate average loss
        train_loss /= len(dataloader_train)
        training_losses.append(train_loss)

        # calculate accuracy
        train_accuracy = num_correct_train / len(dataloader_train.dataset)
        training_accuracies.append(train_accuracy)

        if dataloader_valid is not None:
            valid_loss = 0
            for data in dataloader_valid:
                spatial_obs, non_spatial_obs, action_mask, actions = data
                spatial_obs = spatial_obs.to(device)
                non_spatial_obs = non_spatial_obs.to(device)
                actions = actions.type(torch.LongTensor)
                actions = actions.flatten().to(device)

                _, action_log_probs, = model.get_action_probs(spatial_obs, non_spatial_obs, action_mask)
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
            print('Epoch:', epoch, 'took', round(delta, 3), 'secs', '----Training loss:', round(train_loss, 5), '----Validation loss:', round(valid_loss, 5))
            print('----Training accuracy:', round(train_accuracy, 5), '----Validation accuracy:', round(valid_accuracy, 5))
        else:
            print('Epoch:', epoch, 'took', round(delta, 3), 'secs', '----Training loss:', round(train_loss, 5))
            print('----Training accuracy:', round(train_accuracy, 5))

    # save model after training
    if save_path is not None:
        dir_path = os.path.dirname(save_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(model.state_dict(), save_path)
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

if __name__ == '__main__':
    train_dataset, valid_dataset = get_scripted_dataset(training_percentage=0.75, cache_data=True)
    model, dataloader_train, device = setup_model(train_dataset, batch_size=200, num_workers=2)
    dataloader_valid = get_dataloader(valid_dataset, num_workers=2)
    training_losses, validation_losses = train(model, device, dataloader_train, dataloader_valid, n_epochs=200, save_path=ConfigParams.model_path.value)

    #evaluate_bot(DEFAULT_MODEL_PATH)
