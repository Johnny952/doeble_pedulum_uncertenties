def adjust_range(action, action_range=[0, 1], target_range=[-1, 1]):
    normalized_action = (action - action_range[0]) / (action_range[1] - action_range[0])
    return normalized_action * (target_range[1] - target_range[0]) + target_range[0]