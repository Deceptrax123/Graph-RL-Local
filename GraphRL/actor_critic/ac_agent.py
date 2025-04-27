import torch
import torch.nn.functional as F
from torch import optim


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

            action_tensor = torch.tensor(
                action, dtype=torch.float32, device=mu.device)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(
                action_tensor.unsqueeze(0)).sum(dim=-1).squeeze(0)

            episode_values_list.append(value.squeeze(0))
            episode_log_probs_list.append(log_prob)
            episode_mus_list.append(mu.squeeze(0))
            episode_log_stds_list.append(log_std.squeeze(0))

        episode_values = torch.stack(episode_values_list, dim=0)
        episode_log_probs = torch.stack(episode_log_probs_list, dim=0)
        episode_mus = torch.stack(episode_mus_list, dim=0)
        episode_log_stds = torch.stack(episode_log_stds_list, dim=0)

        final_next_value_scalar = 0.0
        if not batch_dones[-1]:
            with torch.no_grad():
                _, _, last_next_value_tensor = self.model(
                    batch_next_obs_lists[-1])
                final_next_value_scalar = last_next_value_tensor.squeeze().item()

        td_targets_list = []
        advantages_list = []
        next_value_for_advantage_calc = final_next_value_scalar

        for i in reversed(range(num_steps)):
            original_idx = num_steps - 1 - i

            reward_i = batch_rewards[original_idx]
            value_i = episode_values[original_idx]

            done_i = batch_dones[original_idx]
            done_multiplier = float(1 - done_i)

            gamma_tensor = torch.tensor(
                self.gamma, dtype=torch.float32, device=reward_i.device)
            next_value_tensor = torch.tensor(
                next_value_for_advantage_calc, dtype=torch.float32, device=reward_i.device)
            done_multiplier_tensor = torch.tensor(
                done_multiplier, dtype=torch.float32, device=reward_i.device)

            td_target_i = reward_i + gamma_tensor * \
                next_value_tensor * done_multiplier_tensor

            advantage_i = td_target_i - value_i.squeeze(0)

            td_targets_list.append(td_target_i)
            advantages_list.append(advantage_i)

            next_value_for_advantage_calc = value_i.squeeze(0)

        td_targets = torch.stack(td_targets_list).detach()
        advantages = torch.stack(advantages_list).detach()

        advantages_expanded = advantages.unsqueeze(1)

        actor_loss = (-episode_log_probs.unsqueeze(1)
                      * advantages_expanded).mean()

        stds = episode_log_stds.exp()
        dists = torch.distributions.Normal(episode_mus, stds)
        entropy = dists.entropy().sum(dim=-1).mean()
        actor_loss = actor_loss - self.entropy_coeff * entropy

        td_targets_expanded = td_targets.unsqueeze(1)

        critic_loss = F.mse_loss(episode_values, td_targets_expanded)

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.transitions = []

        return actor_loss.item(), critic_loss.item()
