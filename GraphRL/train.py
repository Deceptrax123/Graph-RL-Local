
from environments.graph_environment import GraphEnv
from actor_critic.temporal_ac import SpatioTemporalActorCritic
from actor_critic.ac_agent import ActorCriticAgent
from helpers.matrix import create_data_matrix
from helpers.edge_tensor import edges
import torch
import wandb


def train(env, agent, num_episodes):
    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()

        episode_reward = 0
        done = False
        step_count = 0

        if obs is None:
            print(f"Warning training episode {episode} failed to reset.")
            continue

        while not done:
            action, _ = agent.choose_action(obs)

            next_obs, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs

            episode_reward += reward
            step_count += 1
            if done:
                break

        if step_count > 0:
            actor_loss, critic_loss = agent.update()
        else:
            actor_loss = 0
            critic_loss = 0

        episode_rewards.append(episode_reward)

        if (episode + 1) % 5 == 0:
            episode_total_error = -episode_reward
            avg_total_error_per_step = episode_total_error / \
                step_count if step_count > 0 else 0
            avg_error_per_joint_per_step = avg_total_error_per_step / \
                env.num_markers if step_count > 0 and env.num_markers > 0 else 0

            print(f"\n--- Training Episode {episode+1} ---")
            print(f"Steps: {step_count}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(
                f"Avg Total Error/Step (Training): {avg_total_error_per_step:.4f}")
            print(
                f"Avg Joint Error/Step (Training): {avg_error_per_joint_per_step:.4f}")
            print(
                f"Actor Loss (Training): {actor_loss:.4f}, Critic Loss (Training): {critic_loss:.4f}")

            wandb.log({
                "Training/Episode Reward": episode_reward,
                "Training/Avg Total Error per Step": avg_total_error_per_step,
                "Training/Avg Joint Error per Step": avg_error_per_joint_per_step,
                "Training/Actor Loss": actor_loss,
                "Training/Critic Loss": critic_loss,
                "Training/Steps per Episode": step_count
            }, step=episode+1)

            save_path = f"checkpoints/run_1/model_{episode+1}.pth"
            torch.save(agent.model.state_dict(), save_path)

    return episode_rewards


if __name__ == '__main__':
    window_size = 10
    LEARNING_RATE = 0.00025
    GAMMA = 0.98
    ENTROPY_COEFF = 0.005

    NUM_EPISODES = 2000

    data = create_data_matrix()
    try:
        env = GraphEnv(
            coordinates=data, skeletal_edge_index=edges, emg=None, force=None, window_size=window_size, torque=None
        )

    except ValueError as e:
        print(f"Failed to create environment: {e}")
        exit()

    model = SpatioTemporalActorCritic(
        num_markers=39, node_dim=3, embedding_dim=512, temporal_hidden_dim=1024, window_size=window_size)

    agent = ActorCriticAgent(
        model=model, learning_rate=LEARNING_RATE, gamma=GAMMA, entropy_coeff=ENTROPY_COEFF)

    wandb.init(
        project="Graph RL Pose estimation"
    )

    print("Starting training...")
    rewards_history = train(env, agent, NUM_EPISODES)
    print("Training finished.")
