import torch
import torch.nn.functional as F


class ActorCriticAgent:

    def __init__(self, model, learning_rate=0.0005, gamma=0.99, entropy_coeff=0.005):
        self.model = model
        self.optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.transititons = []  # Store transitions

    def choose_action(self, graph_obs):
        with torch.no_grad():
            mu, std, _ = self.model(graph_obs)

        std = std.exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).numpy(), log_prob.squeeze(0)

    def compute_value(self, graph_obs):
        with torch.no_grad():
            _, _, value = self.model(graph_obs)

    def store_transition(self, graph_obs, action, reward, next_obs_list, done):
        self.transititons.append((graph_obs, torch.tensor(
            action, dtype=torch.float32), reward, next_obs_list, done))

    def update(self):
        if not self.transititons:
            return 0, 0

        batch_obs_list, batch_actions, batch_rewards, batch_next_obs, batch_dones = zip(
            *self.transitions)

        episode_values = []
        episode_log_probs = []
        episode_mus = []
        episode_log_stds = []

        for obs_list, action, _, _, _ in self.transititons:
            mu, log_std, value = self.model(obs_list)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(action).sum(dim=-1)

            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_mus.append(mu)
            episode_log_stds.append(log_std)

        td_targets = []
        advantages = []

        final_next_value = 0

        if not batch_dones[-1]:
            with torch.no_grad():
                _, _, last_next_value = self.model(batch_next_obs[-1])
                final_next_value = last_next_value.squeeze(0)

        next_value = final_next_value
        for i in reversed(range(len(self.transitions))):

            reward = batch_rewards[i]
            value = episode_values[i].squeeze(0)
            done = batch_dones[i]

            td_target = reward + self.gamma * next_value * (1 - done)
            advantage = td_target - value

            td_targets.insert(0, td_target)
            advantages.insert(0, advantage)
            next_value = value

        td_targets = torch.stack(td_targets).detach()
        advantages = torch.stack(advantages).detach()
        episode_values = torch.cat(episode_values)
        episode_log_probs = torch.cat(episode_log_probs)
        episode_mus = torch.cat(episode_mus)
        episode_log_stds = torch.cat(episode_log_stds)

        actor_loss = (-episode_log_probs * advantages).mean()

        stds = episode_log_stds.exp()
        dists = torch.distributions.Normal(episode_mus, stds)
        entropy = dists.entropy().sum(dim=-1).mean()
        actor_loss = actor_loss - self.entropy_coeff * entropy

        critic_loss = F.mse_loss(episode_values.squeeze(1), td_targets)

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.transitions = []

        return actor_loss.item(), critic_loss.item()
