import torch
import torch.nn.functional as F
from torch import optim
from torch import nn


class ActorCriticAgent:
    def __init__(self, model, learning_rate=0.0005, gamma=0.99, entropy_coeff=0.005):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.transitions = []

    def choose_action(self, list_of_graphs_obs):

        with torch.no_grad():
            mu, log_std, _ = self.model(list_of_graphs_obs)

        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).numpy(), log_prob.squeeze(0)

    def compute_value(self, list_of_graphs_obs):

        with torch.no_grad():
            _, _, value = self.model(list_of_graphs_obs)
        return value.squeeze(0)

    def store_transition(self, obs_list, action, reward, next_obs_list, done):
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        self.transitions.append((obs_list, torch.tensor(
            action, dtype=torch.float32), reward_tensor, next_obs_list, done))

    def update(self):
        if not self.transitions:
            return 0, 0

        batch_obs_lists, batch_actions, batch_rewards, batch_next_obs_lists, batch_dones = zip(
            *self.transitions)
        num_steps = len(self.transitions)

        episode_values_list = []
        episode_log_probs_list = []
        episode_mus_list = []
        episode_log_stds_list = []

        for obs_list, action, _, _, _ in self.transitions:
            mu, log_std, value = self.model(obs_list)
            action_tensor = action

            if action_tensor.device != mu.device:
                action_tensor = action_tensor.to(mu.device)

            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(action_tensor.unsqueeze(0)).sum(dim=-1)

            episode_values_list.append(value)
            episode_log_probs_list.append(log_prob)
            episode_mus_list.append(mu)
            episode_log_stds_list.append(log_std)

        episode_values = torch.cat(episode_values_list, dim=0)
        episode_log_probs = torch.cat(episode_log_probs_list, dim=0)
        episode_mus = torch.cat(episode_mus_list, dim=0)
        episode_log_stds = torch.cat(episode_log_stds_list, dim=0)

        device = episode_values.device

        batch_rewards_tensor = torch.stack(batch_rewards).to(device)

        final_next_value_scalar = 0.0
        if not batch_dones[-1] and batch_next_obs_lists[-1] is not None:
            with torch.no_grad():
                _, _, last_next_value_tensor_model_output = self.model(
                    batch_next_obs_lists[-1])
                final_next_value_scalar = last_next_value_tensor_model_output.squeeze().item()

        td_targets_list = []
        advantages_list = []
        next_value_for_advantage_calc = final_next_value_scalar

        num_steps_in_episode = num_steps
        gamma_float = self.gamma

        for i in reversed(range(num_steps_in_episode)):
            current_step_idx = num_steps_in_episode - 1 - i

            reward_i = batch_rewards_tensor[current_step_idx]

            value_i_tensor = episode_values[current_step_idx]
            value_i_scalar_tensor = value_i_tensor.squeeze(0)

            done_i = batch_dones[current_step_idx]
            done_multiplier_float = float(1 - done_i)

            td_target_i = reward_i + gamma_float * \
                next_value_for_advantage_calc * done_multiplier_float

            advantage_i = td_target_i - value_i_scalar_tensor

            td_targets_list.append(td_target_i)
            advantages_list.append(advantage_i)

            next_value_for_advantage_calc = value_i_scalar_tensor.item()

        td_targets_list = td_targets_list[::-1]
        advantages_list = advantages_list[::-1]

        td_targets = torch.stack(td_targets_list).detach()
        advantages = torch.stack(advantages_list).detach()

        log_probs_expanded = episode_log_probs.unsqueeze(1)
        advantages_expanded = advantages.unsqueeze(1)

        actor_loss = (-log_probs_expanded * advantages_expanded).mean()

        stds = episode_log_stds.exp()
        dists = torch.distributions.Normal(episode_mus, stds)
        entropy = dists.entropy().sum(dim=-1).mean()
        actor_loss = actor_loss - self.entropy_coeff * entropy

        td_targets_expanded = td_targets.unsqueeze(1)

        critic_loss = F.mse_loss(episode_values, td_targets_expanded)

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        self.transitions = []

        return actor_loss.item(), critic_loss.item()
