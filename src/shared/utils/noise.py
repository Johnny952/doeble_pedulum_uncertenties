import numpy as np

def add_noise(state, dev, lower=-1, upper=1):
    if dev == 0:
        return state
    noise = np.random.normal(loc=0, scale=dev, size=state.shape)
    noisy_state = state + noise
    noisy_state[noisy_state > 1] = upper
    noisy_state[noisy_state < -1] = lower
    return noisy_state

def add_random_std_noise(state, upper, lower):
    std = np.random.uniform(lower, upper)
    return add_noise(state, std)

def generate_noise_variance(lower, upper):
    return np.random.uniform(lower, upper)