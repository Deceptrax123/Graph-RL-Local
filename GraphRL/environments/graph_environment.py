import gym
from gym import spaces
import torch
import numpy as np
from helpers.graph_creation import create_graph_data


class GraphEnv(gym.Env):
    def __init__(self, coordinates, skeletal_edge_index, emg, force, window_size, torque):
        super(GraphEnv, self).__init__()
        self.coordinates = coordinates  # (t,markers,3)
        self.edge_index = skeletal_edge_index
        self.emg = emg  # (t,16)
        self.force = force  # (t,num_belts,3)
        self.torque = torque  # (t,num_belts,2)
        self.num_markers = 39  # 39 marker set
        self.window_size = window_size
        self.total_time_steps = self.coordinates.shape[0]

        if self.total_time_steps < (self.window_size+1):
            print("Error: Data too short for window size")

        max_coord = np.max(np.absolute(self.coordinates))

        self.action_space = spaces.Box(
            low=-max_coord*2, high=max_coord*2, shape=(39*3,))

        self.graph_observation_space = spaces.Dict({
            'x': spaces.Box(low=-np.inf, high=np.inf, shape=(39, 3), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=39, shape=(2, 39), dtype=np.int64)
        })
        self.current_time_step = window_size-1

    def step(self, action):
        if self.current_time_step+1 >= self.total_time_steps:
            return self._get_obs(), 0, True, {}

        true_next_pos = self.coordinates[self.current_time_step+1]

        predicted_pos = action.reshape(self.num_markers, 3)

        errors_per_joint = np.linalg.norm(predicted_pos-true_next_pos, axis=1)
        total_error = np.sum(errors_per_joint)

        # predicted_pos_tensor = torch.tensor(predicted_pos, dtype=torch.float32)

        # Add penalty terms once info is attained on the same.
        reward = -total_error
        self.current_time_step += 1
        done = self.current_time_step+1 >= self.total_time_steps

        next_obs = self._get_obs() if not done else None

        info = {
            'total_error': total_error,
            'errors_per_joint': errors_per_joint
        }

        return next_obs, reward, done, info

    def reset(self):
        if self.total_time_steps < self.window_size+1:
            raise ValueError("Reset data is too short for window size")

        self.current_time_step = self.window_size-1
        initial_obs = self._get_obs()
        if initial_obs is None:
            raise RuntimeError(
                "Environment failed to generate initial observation on reset.")

        return initial_obs

    def _get_obs(self):
        window_start_idx = self.current_time_step-self.window_size+1
        window_end_index = self.current_time_step

        if window_start_idx < 0 or window_end_index >= self.total_time_steps:
            return None

        obervation_graphs = []
        for t in range(window_start_idx, window_end_index+1):
            markers_t = self.coordinates[t]

            try:
                graph_t = create_graph_data(markers_t, self.edge_index)
                obervation_graphs.append(graph_t)

            except ValueError as e:
                print(f"Error creating graph at time step {t}: {e}")
                return None

        return obervation_graphs

    def render(self, mode='human'):
        pass

    def close(self):
        pass
