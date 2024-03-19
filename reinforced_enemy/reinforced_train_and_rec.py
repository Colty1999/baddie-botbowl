from enum import Enum
from functools import partial

import botbowl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import tqdm

from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper
from examples.scripted_bot_example import *

from env import A2C_Reward
from reinforced_agent import CNNPolicy, A2CAgent, ConfigParams
from reinforcement_parallelization import VecEnv, Memory
from Data.generator import get_scripted_dataset, scripted_data_path
from scripted_behaviour_cloning import get_dataloader

# Todo split training and recording functionalities into separate files to add readability

num_cnn_kernels = [32, 64]


# Make directories
def ensure_dirs():
    ensured_dirs = ["logs/", "models/", "plots/",
                    f"logs/{ConfigParams.env_name.value}/", ConfigParams.model_dir.value,
                    f"plots/{ConfigParams.env_name.value}/"]
    for dir in ensured_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def make_env(env_conf, ppcg=ConfigParams.ppcg.value):
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    # env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env


def main(plot=False):
    torch.cuda.empty_cache()
    ensure_dirs()

    envs = VecEnv([make_env(ConfigParams.env_conf.value) for _ in range(ConfigParams.num_processes.value)])
    make_agent_from_model = partial(A2CAgent, env_conf=ConfigParams.env_conf.value, filename=ConfigParams.model_path.value)  # , scripted_func=a2c_scripted_actions)

    env = make_env(ConfigParams.env_conf.value)
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)

    train_dataset, _ = get_scripted_dataset(training_percentage=0.25)#0.25)
    dataloader_train = get_dataloader(train_dataset, batch_size=5, num_workers=2)
    dataloader_train_iter = iter(dataloader_train)

    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down

    # MODEL
    ac_agent = CNNPolicy(spatial_obs_space,
                         non_spatial_obs_space,
                         hidden_nodes=ConfigParams.num_hidden_nodes.value,
                         # kernels=num_cnn_kernels,
                         actions=action_space)
    ac_agent.load_state_dict(torch.load(ConfigParams.model_path.value, map_location=ConfigParams.device.value))
    ac_agent.to(ConfigParams.device.value)
    loss_function = nn.NLLLoss()
    # OPTIMIZER
    optimizer = optim.RAdam(ac_agent.parameters(), ConfigParams.learning_rate.value, weight_decay=0.00001)

    # train_dataset, _ = get_scripted_dataset(training_percentage=1)
    # dataloader_train = get_dataloader(train_dataset, batch_size=5, num_workers=2)
    # dataloader_train_iter = iter(dataloader_train)

    # MEMORY STORE
    memory = Memory(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value,
                    spatial_obs_space, (1, non_spatial_obs_space), action_space)
    memory.to(ConfigParams.device.value)

    # PPCG
    difficulty = 0.0 if ConfigParams.ppcg.value else 0.01
    dif_delta = 0.01

    # Variables for storing stats
    all_updates = 0
    all_episodes = 0
    all_steps = 0
    episodes = 0
    proc_rewards = np.zeros(ConfigParams.num_processes.value)
    proc_tds = np.zeros(ConfigParams.num_processes.value)
    proc_tds_opp = np.zeros(ConfigParams.num_processes.value)
    episode_rewards = []
    episode_tds = []
    episode_tds_opp = []
    wins = []
    losses = []
    draws = []
    value_losses = []
    policy_losses = []
    log_updates = []
    log_episode = []
    log_steps = []
    log_win_rate = []
    log_td_rate = []
    log_td_rate_opp = []
    log_mean_reward = []
    log_difficulty = []

    # self-play
    selfplay_next_save = ConfigParams.selfplay_save_steps.value
    selfplay_next_swap = ConfigParams.selfplay_swap_steps.value
    selfplay_models = 0
    print('playing against random')
    if ConfigParams.selfplay.value:

        model_name = f"{ConfigParams.exp_id.value}_selfplay_0.nn"
        model_path = os.path.join(f"models/{ConfigParams.env_name.value}/", model_name)
        torch.save(ac_agent, model_path)
        self_play_agent = make_agent_from_model(name=model_name, filename=model_path)
        if selfplay_models % 2 == 1:
            envs.swap(self_play_agent)
            print("playing against self")
        else:
            envs.swap(botbowl.make_bot('scripted'))
            print('playing against scripted')
            selfplay_models += 1

    # Reset environments
    spatial_obs, non_spatial_obs, action_masks, _, _, _, _ = map(torch.from_numpy, envs.reset(difficulty))

    spatial_obs = spatial_obs.to(ConfigParams.device.value)
    non_spatial_obs = non_spatial_obs.to(ConfigParams.device.value)

    # Add first obs to memory
    non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=1)
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)
    memory.action_masks[0].copy_(action_masks)
    memory.to(ConfigParams.device.value)
    updates_num = ConfigParams.num_processes.value//ConfigParams.steps_per_update.value
    pbar = tqdm.tqdm(desc="Updates", total=updates_num)

    while all_steps < ConfigParams.num_steps.value:
        torch.cuda.empty_cache()
        for step in range(ConfigParams.steps_per_update.value):

            _, actions = ac_agent.act(memory.spatial_obs[step], memory.non_spatial_obs[step], memory.action_masks[step], memory.actions[step-1])
                #Variable(memory.spatial_obs[step]),
                #Variable(memory.non_spatial_obs[step]),
                #Variable(memory.action_masks[step]))

            action_objects = (action[0] for action in actions.cpu().numpy())

            spatial_obs, non_spatial_obs, action_masks, shaped_reward, tds_scored, tds_opp_scored, done = envs.step(
                action_objects, difficulty=difficulty)

            proc_rewards += shaped_reward
            proc_tds += tds_scored
            proc_tds_opp += tds_opp_scored
            episodes += done.sum()

            # If done then clean the history of observations.
            for i in range(ConfigParams.num_processes.value):
                if done[i]:
                    if proc_tds[i] > proc_tds_opp[i]:  # Win
                        wins.append(1)
                        losses.append(0)
                        draws.append(0)
                        difficulty += dif_delta
                    elif proc_tds[i] < proc_tds_opp[i]:  # Loss
                        wins.append(0)
                        losses.append(1)
                        draws.append(0)
                        difficulty -= dif_delta
                    else:  # Draw
                        wins.append(0)
                        losses.append(0)
                        draws.append(1)
                        difficulty -= dif_delta
                    if ConfigParams.ppcg.value:
                        difficulty = min(1.0, max(0.01, difficulty))
                    else:
                        difficulty = 0
                    episode_rewards.append(proc_rewards[i])
                    episode_tds.append(proc_tds[i])
                    episode_tds_opp.append(proc_tds_opp[i])
                    proc_rewards[i] = 0
                    proc_tds[i] = 0
                    proc_tds_opp[i] = 0

            # insert the step taken into memory
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            memory.insert(step, spatial_obs, non_spatial_obs, actions.data, shaped_reward, masks, action_masks)

        # -- TRAINING -- #

        # bootstrap next value
        next_value = ac_agent(Variable(memory.spatial_obs[-1], requires_grad=False),
                              Variable(memory.non_spatial_obs[-1], requires_grad=False))[0].data

        # Compute returns
        memory.compute_returns(next_value, ConfigParams.gamma.value)

        spatial = Variable(memory.spatial_obs[:-1])
        spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs[:-1])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(torch.LongTensor(memory.actions.cpu().view(-1, 1)))
        actions = actions.to(ConfigParams.device.value)
        actions_mask = Variable(memory.action_masks[:-1])
        actions_mask = actions_mask.view(-1, action_space)
        returns = Variable(memory.returns[:-1])
        returns = returns.view(-1, 1)
        dataloader = DataLoader(TensorDataset(spatial, non_spatial, actions, actions_mask, returns),
                                batch_size=5, shuffle=True)
        for i, data in enumerate(dataloader):
            print('rl model update steps:', i, end='\r')
            eval_spatial, eval_non_spatial, eval_actions, eval_actions_mask, eval_returns = data
            eval_spatial = eval_spatial.to(ConfigParams.device.value)
            eval_non_spatial = eval_non_spatial.to(ConfigParams.device.value)
            eval_actions = eval_actions.to(ConfigParams.device.value)
            eval_actions_mask = eval_actions_mask.to(ConfigParams.device.value)

            # Evaluate the actions taken
            action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(eval_spatial,
                                                                               eval_non_spatial,
                                                                               eval_actions,
                                                                               eval_actions_mask)
            # print(values.shape)
            # print(action_log_probs.shape)
            # print(eval_returns.shape)
            # values = values.view(steps_per_update, num_processes, 1)
            # action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

            advantages = eval_returns - values
            value_loss = advantages.pow(2).mean()
            # value_losses.append(value_loss)

            # Compute loss
            action_loss = -(Variable(advantages.data) * action_log_probs).mean()
            # policy_losses.append(action_loss)

            optimizer.zero_grad()

            total_loss = (value_loss * ConfigParams.value_loss_coef.value + action_loss - dist_entropy * ConfigParams.entropy_coef.value)
            total_loss.backward()

            nn.utils.clip_grad_norm_(ac_agent.parameters(), ConfigParams.max_grad_norm.value)

            optimizer.step()

        # print(returns.shape)

        # # Evaluate the actions taken
        # action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)
        #
        # values = values.view(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value, 1)
        # action_log_probs = action_log_probs.view(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value, 1)
        #
        # advantages = Variable(memory.returns[:-1]) - values
        # value_loss = advantages.pow(2).mean()
        # # value_losses.append(value_loss)
        #
        # # Compute loss
        # action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        # # policy_losses.append(action_loss)
        #
        # optimizer.zero_grad()
        #
        # total_loss = (value_loss * ConfigParams.value_loss_coef.value + action_loss
        #               - dist_entropy * ConfigParams.entropy_coef.value)
        # total_loss.backward()
        #
        # nn.utils.clip_grad_norm_(ac_agent.parameters(), ConfigParams.max_grad_norm.value)
        #
        # optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])
        memory.action_masks[0].copy_(memory.action_masks[-1])

        # Updates
        all_updates += 1
        # Episodes
        all_episodes += episodes
        episodes = 0
        # Steps
        all_steps += ConfigParams.num_processes.value * ConfigParams.steps_per_update.value

        # Self-play save
        if ConfigParams.selfplay.value and all_steps >= selfplay_next_save:
            selfplay_next_save = max(all_steps + 1, selfplay_next_save + ConfigParams.selfplay_save_steps.value)
            model_name = f"{ConfigParams.exp_id.value}_selfplay_{selfplay_models}.nn"
            model_path = os.path.join(ConfigParams.model_dir.value, model_name)
            # print(f"Saving {model_path}")
            pbar.write(f"Saving {model_path}")
            torch.save(ac_agent, model_path)
            selfplay_models += 1

        # Self-play swap
        if ConfigParams.selfplay.value and all_steps >= selfplay_next_swap:
            selfplay_next_swap = max(all_steps + 1, selfplay_next_swap + ConfigParams.selfplay_swap_steps.value)
            lower = max(0, selfplay_models - 1 - (ConfigParams.selfplay_window.value - 1))
            if selfplay_models % 2 == 0:
                i = random.randint(lower, selfplay_models - 1)
                model_name = f"{ConfigParams.exp_id.value}_selfplay_{i}.nn"
                model_path = os.path.join(ConfigParams.model_dir.value, model_name)
                # print(f"Swapping opponent to {model_path}")
                pbar.write(f"Swapping opponent to {model_path}")
                envs.swap(make_agent_from_model(name=model_name, filename=model_path))
            else:
                envs.swap(botbowl.make_bot('scripted'))
                pbar.write("Swapping opponent to scripted")
                # print("Swapping opponent to scripted")
            spatial_obs, non_spatial_obs, action_masks, _, _, _, _ = map(torch.from_numpy, envs.reset(difficulty))

        # --- BC ---
        for _ in range(ConfigParams.steps_per_update.value):
            # get one batch from the dataset
            # Todo remove redundancies below
            is_spatial_zero = True
            while is_spatial_zero:  # Todo fix the bug with all zeroes in spatial obs
                try:
                    spatial_obs, non_spatial_obs, action_mask, actions = next(dataloader_train_iter)
                except StopIteration:  # if the iterator is empty, restart
                    dataloader_train_iter = iter(dataloader_train)
                    spatial_obs, non_spatial_obs, action_mask, actions = next(dataloader_train_iter)
                finally:
                    if torch.count_nonzero(spatial_obs) > 0:
                        is_spatial_zero = False
            # if isinstance(spatial_obs,np.ndarray):
            #     spatial_obs = torch.from_numpy(spatial_obs.astype('float32'))
            # if isinstance(non_spatial_obs,np.ndarray):
            #     non_spatial_obs = torch.from_numpy(non_spatial_obs.astype('float32'))
            # if isinstance(actions,np.ndarray):
            #     actions = torch.from_numpy(actions.astype('float32'))
            spatial_obs = spatial_obs.to(ConfigParams.device.value)
            non_spatial_obs = non_spatial_obs.to(ConfigParams.device.value)
            actions = actions.type(torch.LongTensor)  # Convert to Long to avoid not implemented for Int error
            actions = actions.flatten().to(ConfigParams.device.value)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, action_log_probs, = ac_agent.get_action_log_probs(spatial_obs, non_spatial_obs)

            # actions = actions.to(ConfigParams.device.value)
            loss = loss_function(action_log_probs, actions)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            if train_loss == float('nan'):
                print(f'Train Loss: {train_loss:.3f}')

            all_steps += 5  # Todo make batch size adjustable

        # Logging
        if all_updates % ConfigParams.log_interval.value == 0 \
                and len(episode_rewards) >= ConfigParams.num_processes.value:
            td_rate = np.mean(episode_tds)
            td_rate_opp = np.mean(episode_tds_opp)
            episode_tds.clear()
            episode_tds_opp.clear()
            mean_reward = np.mean(episode_rewards)
            episode_rewards.clear()
            win_rate = np.mean(wins)
            lose_rate = np.mean(losses)
            draw_rate = np.mean(draws)
            wins.clear()
            losses.clear()
            draws.clear()

            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_td_rate_opp.append(td_rate_opp)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)

            log = "Updates: {}, Episodes: {}, Timesteps: {}, Win rate: {:.2f}, Draw rate: {:.2f}, Lose rate: {:.2f}, TD rate: {:.2f}, TD rate opp: {:.2f}, Mean reward: {:.3f}, Difficulty: {:.2f}" \
                .format(all_updates, all_episodes, all_steps, win_rate, draw_rate, lose_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            # Save to files
            log_path = os.path.join(f"logs/{ConfigParams.env_name.value}/", f"{ConfigParams.exp_id.value}.dat")
            # print(f"Save log to {log_path}")
            pbar.write(f"Save log to {log_path}")
            with open(log_path, "a") as myfile:
                myfile.write(log_to_file)

            # print(log)
            pbar.write(log)


            episodes = 0
            value_losses.clear()
            policy_losses.clear()

            # Save model
            model_name = f"{ConfigParams.exp_id.value}.nn"
            model_path = os.path.join(ConfigParams.model_dir.value, model_name)
            torch.save(ac_agent, model_path)

            # plot
            if plot:
                n = 3
                if ConfigParams.ppcg.value:
                    n += 1
                fig, axs = plt.subplots(1, n, figsize=(4 * n, 5))
                axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[0].plot(log_steps, log_mean_reward)
                axs[0].set_title('Reward')
                # axs[0].set_ylim(bottom=0.0)
                axs[0].set_xlim(left=0)
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[1].plot(log_steps, log_td_rate, label="Learner")
                axs[1].set_title('TD/Episode')
                axs[1].set_ylim(bottom=0.0)
                axs[1].set_xlim(left=0)
                if ConfigParams.selfplay.value:
                    axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                    axs[1].plot(log_steps, log_td_rate_opp, color="red", label="Opponent")
                axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[2].plot(log_steps, log_win_rate)
                axs[2].set_title('Win rate')
                axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
                axs[2].set_xlim(left=0)
                if ConfigParams.ppcg.value:
                    axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                    axs[3].plot(log_steps, log_difficulty)
                    axs[3].set_title('Difficulty')
                    axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
                    axs[3].set_xlim(left=0)
                fig.tight_layout()
                plot_name = f"{ConfigParams.exp_id.value}_{'_selfplay' if ConfigParams.selfplay.value else ''}.png"
                plot_path = os.path.join(f"plots/{ConfigParams.env_name.value}/", plot_name)
                fig.savefig(plot_path)
                plt.close('all')
        pbar.update()
    pbar.close()
    model_name = f"{ConfigParams.exp_id.value}.nn"
    dict_name = f"{ConfigParams.exp_id.value}_dict.nn"
    model_path = os.path.join(ConfigParams.model_dir.value, model_name)
    torch.save(ac_agent, model_path)
    torch.save(ac_agent.state_dict(), dict_name)
    envs.close()


if __name__ == "__main__":
    main()
