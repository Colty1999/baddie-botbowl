from enum import Enum
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import uuid

from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper, PPCGWrapper

from env import A2C_Reward
from reinforced_agent import CNNPolicy, A2CAgent
from reinforcement_parallelization import VecEnv, Memory

# Todo split training and recording functionalities into separate files to add readability


class ConfigParams(Enum):
    num_steps = 1000000
    num_processes = 8
    steps_per_update = 20
    learning_rate = 0.001
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.05
    log_interval = 50
    save_interval = 10
    reset_steps = 5000  # The environment is reset after this many steps it gets stuck
    selfplay_window = 1
    selfplay_save_steps = int(num_steps / 10)
    selfplay_swap_steps = selfplay_save_steps
    num_hidden_nodes = 128
    ppcg = False
    env_size = 1  # Options are 1,3,5,7,11
    env_name = f"botbowl-{env_size}"
    env_conf = EnvConf(size=env_size, pathfinding=False)
    selfplay = False
    exp_id = str(uuid.uuid1())
    model_dir = f"models/{env_name}/"


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


def main():
    ensure_dirs()

    envs = VecEnv([make_env(ConfigParams.env_conf.value) for _ in range(ConfigParams.num_processes.value)])
    make_agent_from_model = partial(A2CAgent, env_conf=ConfigParams.env_conf.value)  # , scripted_func=a2c_scripted_actions)

    env = make_env(ConfigParams.env_conf.value)
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    del env, non_spat_obs, action_mask  # remove from scope to avoid confusion further down

    # MODEL
    ac_agent = CNNPolicy(spatial_obs_space,
                         non_spatial_obs_space,
                         hidden_nodes=ConfigParams.num_hidden_nodes.value,
                         kernels=num_cnn_kernels,
                         actions=action_space)

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), ConfigParams.learning_rate.value)

    # MEMORY STORE
    memory = Memory(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value,
                    spatial_obs_space, (1, non_spatial_obs_space), action_space)

    # PPCG
    difficulty = 0.0 if ConfigParams.ppcg.value else 1.0
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

    if ConfigParams.selfplay.value:
        model_name = f"{ConfigParams.exp_id.value}_selfplay_0.nn"
        model_path = os.path.join(f"models/{ConfigParams.env_name.value}/", model_name)
        torch.save(ac_agent, model_path)
        self_play_agent = make_agent_from_model(name=model_name, filename=model_path)
        envs.swap(self_play_agent)
        selfplay_models += 1

    # Reset environments
    spatial_obs, non_spatial_obs, action_masks, _, _, _, _ = map(torch.from_numpy, envs.reset(difficulty))

    # Add first obs to memory
    non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=1)
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)
    memory.action_masks[0].copy_(action_masks)

    while all_steps < ConfigParams.num_steps.value:
        for step in range(ConfigParams.steps_per_update.value):

            _, actions = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]),
                Variable(memory.action_masks[step]))

            action_objects = (action[0] for action in actions.numpy())

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
                        difficulty += dif_delta
                    elif proc_tds[i] < proc_tds_opp[i]:  # Loss
                        wins.append(0)
                        difficulty -= dif_delta
                    else:  # Draw
                        wins.append(0.5)
                        difficulty -= dif_delta
                    if ConfigParams.ppcg.value:
                        difficulty = min(1.0, max(0, difficulty))
                    else:
                        difficulty = 1
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

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.action_masks[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        values = values.view(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value, 1)
        action_log_probs = action_log_probs.view(ConfigParams.steps_per_update.value, ConfigParams.num_processes.value, 1)

        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()
        # value_losses.append(value_loss)

        # Compute loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        # policy_losses.append(action_loss)

        optimizer.zero_grad()

        total_loss = (value_loss * ConfigParams.value_loss_coef.value + action_loss
                      - dist_entropy * ConfigParams.entropy_coef.value)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), ConfigParams.max_grad_norm.value)

        optimizer.step()

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
            print(f"Saving {model_path}")
            torch.save(ac_agent, model_path)
            selfplay_models += 1

        # Self-play swap
        if ConfigParams.selfplay.value and all_steps >= selfplay_next_swap:
            selfplay_next_swap = max(all_steps + 1, selfplay_next_swap + ConfigParams.selfplay_swap_steps.value)
            lower = max(0, selfplay_models - 1 - (ConfigParams.selfplay_window.value - 1))
            i = random.randint(lower, selfplay_models - 1)
            model_name = f"{ConfigParams.exp_id.value}_selfplay_{i}.nn"
            model_path = os.path.join(ConfigParams.model_dir.value, model_name)
            print(f"Swapping opponent to {model_path}")
            envs.swap(make_agent_from_model(name=model_name, filename=model_path))

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
            wins.clear()

            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_td_rate_opp.append(td_rate_opp)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)

            log = "Updates: {}, Episodes: {}, Timesteps: {}, Win rate: {:.2f}, TD rate: {:.2f}, TD rate opp: {:.2f}, Mean reward: {:.3f}, Difficulty: {:.2f}" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            # Save to files
            log_path = os.path.join(f"logs/{ConfigParams.env_name.value}/", f"{ConfigParams.exp_id.value}.dat")
            print(f"Save log to {log_path}")
            with open(log_path, "a") as myfile:
                myfile.write(log_to_file)

            print(log)

            episodes = 0
            value_losses.clear()
            policy_losses.clear()

            # Save model
            model_name = f"{ConfigParams.exp_id.value}.nn"
            model_path = os.path.join(ConfigParams.model_dir.value, model_name)
            torch.save(ac_agent, model_path)

            # plot
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

    model_name = f"{ConfigParams.exp_id.value}.nn"
    model_path = os.path.join(ConfigParams.model_dir.value, model_name)
    torch.save(ac_agent, model_path)
    envs.close()


if __name__ == "__main__":
    main()
