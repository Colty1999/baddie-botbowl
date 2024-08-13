from enum import Enum
from functools import partial

import botbowl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import tqdm

from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper
#from examples.scripted_bot_example import *

from Data.scripted_bot import ScriptedBot
#Data.scripted_bot2 import ScriptedBot  | to use modify bot



from reinforced_enemy.env import A2C_Reward
from network import CNNPolicy, ConfigParams
from actor import BaddieBotActor
from reinforced_enemy.reinforcement_parallelization import VecEnv, Memory
from Data.generator import get_scripted_dataset, scripted_data_path
from replay_buffer import PrioritizedReplayBuffer
# from test import PrioritizedReplayBuffer
from behaviour_cloning import get_dataloader
# from reinforced_enemy.scripted_behaviour_cloning import get_dataloader
import gc

from botbowl.core import procedure

gc.collect()

torch.cuda.empty_cache()

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
# old scripted bot
def a2c_scripted_actions(game):
    proc_type = type(game.get_procedure())
    scripted_bot = ScriptedBot(name='MyScriptedBot', out_path="./scripted_dataset", dump=False)
    if proc_type is procedure.Block:
        return scripted_bot.block(game=game)
    elif proc_type is procedure.CoinTossFlip:
        return scripted_bot.coin_toss_flip(game=game)
    elif proc_type is procedure.CoinTossKickReceive:
        return scripted_bot.coin_toss_kick_receive(game=game)
    elif proc_type is procedure.Reroll:
        return scripted_bot.reroll(game=game)
    elif proc_type is procedure.PlaceBall:
        return scripted_bot.place_ball(game=game)
    available_action_types = [action_choice.action_type for action_choice in game.get_available_actions()]
    if len(available_action_types) == 1 and len(game.get_available_actions()[0].positions) == 0 and len(game.get_available_actions()[0].players) == 0:
         return botbowl.Action(available_action_types[0])
    if botbowl.ActionType.END_PLAYER_TURN in available_action_types:
        return botbowl.Action(botbowl.ActionType.END_PLAYER_TURN)
    return None

def make_env(env_conf, ppcg=ConfigParams.ppcg.value):
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions) #todo add scripted actions later on
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env


def main(load_model=True, difficulty=0.0, plot=False):
    torch.cuda.empty_cache()
    ensure_dirs()

    envs = VecEnv([make_env(ConfigParams.env_conf.value) for _ in range(ConfigParams.num_processes.value)])

    env = make_env(ConfigParams.env_conf.value)
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)

    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down

    if load_model:
        make_dqn_from_model = partial(BaddieBotActor, env_conf=ConfigParams.env_conf.value,
                                      filename=ConfigParams.model_path.value , scripted_func=a2c_scripted_actions)
        print("Creating model from:", ConfigParams.model_path.value)
        dqn = torch.load(ConfigParams.model_path.value)
        dqn.eval()
        target_dqn = torch.load(ConfigParams.model_path.value)
        target_dqn.eval()
    else:
        make_dqn_from_model = partial(BaddieBotActor, env_conf=ConfigParams.env_conf.value, filename=None)
        dqn = CNNPolicy()
        dqn.to(ConfigParams.device.value)
        target_dqn = CNNPolicy()
        target_dqn.to(ConfigParams.device.value)
        target_dqn.load_state_dict(dqn.state_dict())


    # Todo BC
    train_dataset, _ = get_scripted_dataset(training_percentage=1)
    dataloader_train = get_dataloader(train_dataset, batch_size=16, num_workers=1)  # original batch_size=5
    dataloader_train_iter = iter(dataloader_train)

    loss_function = nn.NLLLoss()
    # OPTIMIZER
    optimizer = optim.RAdam(target_dqn.parameters(), ConfigParams.learning_rate.value, weight_decay=0.00001)

    # PRIORITIZED REPLAY BUFFER INIT
    buffer = PrioritizedReplayBuffer()
    buffer.to(ConfigParams.device.value)

    # todo PPCG
    difficulty = difficulty if ConfigParams.ppcg.value else 1.0  # todo CHANGE DIFFICULTY BACK TO 1.0
    dif_delta = 0 #0.00001

    # Variables for storing stats
    all_updates = 0
    all_episodes = 0
    all_steps = 0
    episodes = 0
    best_reward = -100000
    best_diff = 0.001
    no_improvement = 0
    entropy_coef = ConfigParams.entropy_coef.value
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

    # todo self-play
    selfplay_next_save = ConfigParams.selfplay_save_steps.value
    selfplay_next_swap = ConfigParams.selfplay_swap_steps.value
    selfplay_models = 0
    print('playing against random')
    envs.swap('random')
    if ConfigParams.selfplay.value:

        model_name = f"{ConfigParams.exp_id.value}_selfplay_0.nn"
        model_path = os.path.join(f"models/{ConfigParams.env_name.value}/", model_name)
        torch.save(dqn, model_path)
        self_play_dqn = make_dqn_from_model(name=model_name, filename=model_path)
        if selfplay_models % 2 == 1:
            envs.swap(self_play_dqn)
            print("playing against self")
        else:
            envs.swap(botbowl.make_bot('scripted'))
            print('playing against scripted')
            selfplay_models += 1

    # Reset environments
    spatial_obs, non_spatial_obs, action_masks, _, _, _, _ = map(torch.from_numpy, envs.reset(difficulty))

    spatial_obs = spatial_obs.to(ConfigParams.device.value)
    non_spatial_obs = non_spatial_obs.to(ConfigParams.device.value)

    # Add first obs to buffer
    # todo check if this works as intended
    non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=1)
    buffer.spatial_obs[0].copy_(spatial_obs)
    buffer.next_spatial_obs[0].copy_(spatial_obs)
    buffer.non_spatial_obs[0].copy_(non_spatial_obs)
    buffer.next_non_spatial_obs[0].copy_(non_spatial_obs)
    buffer.action_masks[0].copy_(action_masks)
    buffer.next_action_masks[0].copy_(action_masks)
    buffer.to(ConfigParams.device.value)
    updates_num = ConfigParams.num_processes.value//ConfigParams.steps_per_update.value
    non_spatial_obs = torch.squeeze(non_spatial_obs, dim=1)
    spatial_obs, non_spatial_obs, action_masks = spatial_obs.numpy(force=True), non_spatial_obs.numpy(force=True), action_masks.numpy(force=True)
    pbar = tqdm.tqdm(desc="Updates", total=updates_num)
    test = 0
    step = 0
    epsilon = 0.00
    eps_decay = 0.995
    eps_final = 0.00

    while all_steps < ConfigParams.num_steps.value:
        torch.cuda.empty_cache()
        # epsilon = eps_final + (epsilon - eps_final) * np.exp(-eps_decay * all_steps)
        # todo check if steps per update work well as buffer size
        for _ in range(ConfigParams.steps_per_update.value):
            # todo check if implementation of epsilon greedy will help
            if np.random.rand() < epsilon:
                eps_mask = buffer.action_masks[step].clone().float()
                actions = torch.multinomial(eps_mask, num_samples=1)
            else:
                _, actions = dqn.act(buffer.spatial_obs[step], buffer.non_spatial_obs[step], buffer.action_masks[step], buffer.actions[step-1])
                #todo check the alternative
                # value, advantage = dqn.act(buffer.spatial_obs[step], buffer.non_spatial_obs[step], buffer.action_masks[step],
                #                      buffer.actions[step - 1])
                # actions = value + (advantage - advantage.float().mean())

            action_objects = (action[0] for action in actions.detach().cpu().numpy())

            next_spatial_obs, next_non_spatial_obs, next_action_masks, shaped_reward, tds_scored, tds_opp_scored, done = envs.step(
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
                        # difficulty += dif_delta
                    elif proc_tds[i] < proc_tds_opp[i]:  # Loss
                        wins.append(0)
                        losses.append(1)
                        draws.append(0)
                        # difficulty -= dif_delta/2
                    else:  # Draw
                        wins.append(0)
                        losses.append(0)
                        draws.append(1)
                        # difficulty -= dif_delta/4
                    if ConfigParams.ppcg.value:
                        difficulty = min(1.0, max(0, difficulty))
                    else:
                        difficulty = 1.0
                    episode_rewards.append(proc_rewards[i])
                    episode_tds.append(proc_tds[i])
                    episode_tds_opp.append(proc_tds_opp[i])
                    proc_rewards[i] = 0
                    proc_tds[i] = 0
                    proc_tds_opp[i] = 0

            # insert the step taken into buffer
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            spatial_obs, non_spatial_obs = next_spatial_obs, next_non_spatial_obs  # todo  clean this up, deprecated to be removed
            action_masks = next_action_masks  # todo  clean this up, deprecated to be removed
            # todo remove redundancies, we do not keep next states separetely, we just get idx+1
            buffer.add(
                step, spatial_obs, non_spatial_obs,
                actions.data, shaped_reward, masks, action_masks
            )

            # spatial_obs, non_spatial_obs = next_spatial_obs, next_non_spatial_obs  # ORIGINALLY WAS HERE, MOVED UP
            step = (step + 1) % ConfigParams.buffer_size.value

        # -- TRAINING -- #
        epsilon = max(epsilon*eps_decay, eps_final)
        if step == 0:
            buffer.non_spatial_obs[0].copy_(buffer.non_spatial_obs[-1])
            buffer.spatial_obs[0].copy_(buffer.spatial_obs[-1])
            buffer.action_masks[0].copy_(buffer.action_masks[-1])
        for _ in range(ConfigParams.multiple_updates.value):

            # Sample data
            # todo simplify the process below
            batch = buffer.sample(ConfigParams.batch_size.value)#, beta=0.4)
            batch_spatial_obs, batch_non_spatial_obs, batch_next_spatial_obs, batch_next_non_spatial_obs,\
            batch_actions, batch_shaped_reward, batch_masks, batch_action_masks, batch_next_action_masks, \
            weights, idexes = batch

            # todo check if works as intended/is needed
            spatial = Variable(batch_spatial_obs)
            spatial = spatial.view(-1, *spatial_obs_space)
            non_spatial = Variable(batch_non_spatial_obs)
            non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

            next_spatial = Variable(batch_next_spatial_obs)
            next_spatial = next_spatial.view(-1, *spatial_obs_space)
            next_non_spatial = Variable(batch_next_non_spatial_obs)
            next_non_spatial = next_non_spatial.view(-1, non_spatial.shape[-1])

            batch_actions = Variable(torch.LongTensor(batch_actions.cpu().view(-1, 1)))
            batch_actions = batch_actions.to(ConfigParams.device.value)

            batch_actions_mask = Variable(batch_action_masks)
            batch_actions_mask = batch_actions_mask.view(-1, action_space)
            batch_next_actions_mask = Variable(batch_next_action_masks)
            batch_next_actions_mask = batch_next_actions_mask.view(-1, action_space)

            batch_shaped_reward = Variable(batch_shaped_reward.view(-1, 1))
            batch_masks = Variable(batch_masks.view(-1, 1))
            # todo check if logic below check out with current one
            # if global_step > args.learning_starts:
            #     if global_step % args.train_frequency == 0:
            #         data = rb.sample(args.batch_size)
            #         with torch.no_grad():
            #             target_max, _ = target_network(data.next_observations).max(dim=1)
            #             td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            #         old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            #         loss = F.mse_loss(td_target, old_val)
            #todo check the alternative - remember to also make this change in the gathering phase
            # curr_value, curr_action = dqn.get_action_probs(spatial, non_spatial, batch_actions_mask)
            # curr_q = curr_value + (curr_action - curr_action.float().mean())
            # curr_q = curr_q.gather(1, batch_actions)
            # bootstrap_value, bootstrap_action = target_dqn.get_action_probs(next_spatial, next_non_spatial, batch_next_actions_mask)
            # bootstrap_action.to(torch.int64)
            # bootstrap_q = bootstrap_value + (bootstrap_action - bootstrap_action.float().mean())
            # bootstrap_q = torch.max(bootstrap_q, 1)[0].view(bootstrap_q.size(0), 1)

            _, curr_q = dqn.get_action_probs(spatial, non_spatial, batch_actions_mask)  #todo make more efficient
            curr_q = curr_q.gather(1, batch_actions)
            # _, bootstrap_q = target_dqn(spatial, non_spatial)
            _, bootstrap_q = target_dqn.get_action_probs(next_spatial, next_non_spatial, batch_next_actions_mask)
            bootstrap_q = torch.max(bootstrap_q,1)[0].view(bootstrap_q.size(0), 1)
            # self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]

            target_q = batch_shaped_reward + batch_masks * ConfigParams.gamma.value * bootstrap_q # ConfigParams.gamma.value ** ConfigParams.steps_per_update.value
            weights = torch.FloatTensor(weights).to(ConfigParams.device.value)
            weights.cuda(non_blocking=True)
            weights = weights.mean()

            q_loss = (
                    weights * F.smooth_l1_loss(curr_q, target_q.detach(), reduction="none")
            ).mean()
            dqn_reg = torch.norm(q_loss, 2).mean() * ConfigParams.q_regularization.value
            loss = q_loss + dqn_reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dqn.parameters(), ConfigParams.gradient_clip.value) # Gradient clipping
            optimizer.step()

            for target_param, param in zip(
                    target_dqn.parameters(), dqn.parameters()
            ):
                target_param.data.copy_(ConfigParams.tau.value * param + (1 - ConfigParams.tau.value) * target_param)

            new_priorities = torch.abs(target_q - curr_q).detach().view(-1)
            new_priorities = torch.clamp(new_priorities, min=1e-6)
            new_priorities = new_priorities.cpu().numpy().tolist()

            # for priorities in new_priorities:
            buffer.update_priorities(
                idexes, new_priorities
            )
        buffer.update_beta()
        # buffer.non_spatial_obs[0].copy_(buffer.non_spatial_obs[-1])
        # buffer.spatial_obs[0].copy_(buffer.spatial_obs[-1])
        # buffer.action_masks[0].copy_(buffer.action_masks[-1])

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
            pbar.write(f"Saving {model_path}")
            torch.save(dqn, model_path)
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
                envs.swap(make_dqn_from_model(name=model_name, filename=model_path))
            else:
                envs.swap(botbowl.make_bot('random'))
                pbar.write("Swapping opponent to random")
                # print("Swapping opponent to scripted")
            spatial_obs, non_spatial_obs, action_masks, _, _, _, _ = map(torch.from_numpy, envs.reset())#difficulty))

        # --- BC ---
        bc_spatial_obs, bc_non_spatial_obs, bc_action_mask, bc_actions = spatial_obs, non_spatial_obs, action_masks, actions
        for _ in range(ConfigParams.steps_per_update.value):
            # get one batch from the dataset
            # Todo remove redundancies below
            is_spatial_zero = True
            while is_spatial_zero:  # Todo fix the bug with all zeroes in spatial obs
                try:
                    bc_spatial_obs, bc_non_spatial_obs, bc_action_mask, bc_actions = next(dataloader_train_iter)
                except StopIteration:  # if the iterator is empty, restart
                    dataloader_train_iter = iter(dataloader_train)
                    bc_spatial_obs, bc_non_spatial_obs, bc_action_mask, bc_actions = next(dataloader_train_iter)
                finally:
                    if torch.count_nonzero(bc_spatial_obs) > 0:
                        is_spatial_zero = False
            bc_spatial_obs = bc_spatial_obs.to(ConfigParams.device.value)
            bc_non_spatial_obs = bc_non_spatial_obs.to(ConfigParams.device.value)
            bc_actions = bc_actions.type(torch.LongTensor)  # Convert to Long to avoid not implemented for Int error
            bc_actions = bc_actions.flatten().to(ConfigParams.device.value)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, bc_action_log_probs = target_dqn.get_action_log_probs(bc_spatial_obs, bc_non_spatial_obs)

            # actions = actions.to(ConfigParams.device.value)
            loss = loss_function(bc_action_log_probs, bc_actions)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            if train_loss == float('nan'):
                print(f'Train Loss: {train_loss:.3f}')

        all_steps += ConfigParams.steps_per_update.value  # add BC steps
        if all_steps % 10 == 0:
            dqn.load_state_dict(target_dqn.state_dict())  # sync with learner
            # print("Sync")

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

            log_to_file = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, draw_rate, lose_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            # Save to files
            log_path = os.path.join(f"logs/{ConfigParams.env_name.value}/", f"{ConfigParams.exp_id.value}.dat")
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
            torch.save(target_dqn, model_path)
            # if difficulty >= best_diff:
            if mean_reward*difficulty > best_reward*best_diff:
                best_reward = mean_reward
                best_diff = difficulty
                best_model_name = "best.nn"
                best_model_path = os.path.join(ConfigParams.model_dir.value, best_model_name)
                torch.save(dqn, best_model_path)
                no_improvement = 0
                print("New best found")
            #todo make sure below will work as intended
            # else:
            #     no_improvement += 1
            #     if no_improvement > ConfigParams.patience.value:  # Sanity check, rollback to a better model
            #         best_model_name = "best.nn"
            #         best_model_path = os.path.join(ConfigParams.model_dir.value, best_model_name)
            #         dqn = torch.load(best_model_path)
            #         dqn.eval()
            #         target_dqn = torch.load(best_model_path)
            #         target_dqn.eval()
            #         no_improvement = 0
            #         print("Rollback to best")

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
        test += 1
        pbar.update()
    pbar.close()
    model_name = f"{ConfigParams.exp_id.value}.nn"
    dict_name = f"{ConfigParams.exp_id.value}_dict.nn"
    model_path = os.path.join(ConfigParams.model_dir.value, model_name)
    torch.save(target_dqn, model_path)
    torch.save(target_dqn.state_dict(), dict_name)
    envs.close()


if __name__ == "__main__":
    main(load_model=True, difficulty=1.0)
