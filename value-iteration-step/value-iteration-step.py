def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    num_values = len(values)
    new_values=  [0.0] * num_values
    for s in range(num_values):
        max_q = float('-inf')
        for a in range(len(transitions[s])):
            q_val = rewards[s][a]
            expected_future_val = 0.0 
            for s_next in range(num_values):
                expected_future_val += transitions[s][a][s_next] * values[s_next]
            q_val += gamma * expected_future_val
            if q_val > max_q:
                max_q = q_val
        new_values[s] = max_q
    return new_values