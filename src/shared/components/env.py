import numpy as np
from gym.wrappers import Monitor
import gym
from collections import deque

import sys
sys.path.append('../..')
from shared.utils.noise import generate_noise_variance, add_noise


class Env():
    """
    Environment wrapper for InvertedPendulum-v4 
    """

    def __init__(self, state_stack: int, action_repeat: int, seed: float=0, path_render: str=None, evaluations: int=1, noise=None, done_reward_threshold: float=-0.1, done_reward: float=0):
        self.render_path = path_render is not None
        if not self.render_path:
            self.env = gym.make('InvertedPendulum-v2')
        else:
            self.evaluations = evaluations
            self.idx_val = evaluations // 2
            self.env = Monitor(gym.make('InvertedPendulum-v2'), path_render,
                               video_callable=lambda episode_id: episode_id % evaluations == self.idx_val, force=True)
        self.env.seed(seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.action_repeat = action_repeat
        self.done_reward_threshold = done_reward_threshold
        self.done_reward = done_reward
        #self.env._max_episode_steps = your_value
        self.observation_dims = 4
        self.observation_space = [-3, 3]
        self.action_dims = 1

        self.reward_memory = deque([], maxlen=100)
        self.state_stack = deque([], maxlen=state_stack)

        # Noise in initial observations
        self.use_noise = False
        self.random_noise = 0
        if noise:
            if type(noise) is list:
                if len(noise) == 1:
                    self.set_noise_value(noise[0])
                elif len(noise) >= 2:
                    self.set_noise_range(noise)
            elif type(noise) is float and noise >= 0:
                self.set_noise_value(noise)
    
    def close(self):
        self.env.close()
    
    def set_noise_range(self, noise):
        assert type(noise) is list
        assert len(noise) >= 2
        self.use_noise = True
        self.generate_noise = True
        self.noise_lower, self.noise_upper = noise[0], noise[1]
    
    def set_noise_value(self, noise):
        assert noise >= 0
        self.use_noise = True
        self.generate_noise = False
        self.random_noise = noise

    def reset(self):
        self.reward_memory.clear()
        self.die = False
        state = self.env.reset()

        if self.use_noise:
            if self.generate_noise:
                self.random_noise = generate_noise_variance(
                    self.noise_lower, self.noise_upper)
            state = add_noise(state, self.random_noise)

        for _ in range(self.state_stack.maxlen):
            self.state_stack.append(state)
        return np.array(self.state_stack)

    def step(self, action):
        total_reward = 0
        total_steps = 0
        for _ in range(self.action_repeat):
            state, reward, die, info = self.env.step(action)
            # if no reward recently, end the episode
            done = False
            done_reward = 0
            self.reward_memory.append(reward)
            if np.mean(self.reward_memory) <= self.done_reward_threshold:
                done_reward += self.done_reward
                done = True
            reward += done_reward
            total_steps += 1
            total_reward += reward
            if done or die:
                break
        
        info["steps"] = total_steps
        
        # Add noise in observation
        if self.use_noise:
            state = add_noise(state, self.random_noise)
        self.state_stack.append(state)
        info["noise"] = self.random_noise
        assert len(self.state_stack) == self.state_stack.maxlen
        return np.array(self.state_stack), total_reward, done, die, info

    def render(self, *arg):
        return self.env.render(*arg)


if __name__ == "__main__":
    env = Env(2, 2, path_render='render', evaluations=1)

    state = env.reset()

    for i in range(1000):
        state, reward, done, die, info = env.step([0])
        if done or die:
            break