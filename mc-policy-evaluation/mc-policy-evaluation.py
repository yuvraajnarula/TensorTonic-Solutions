import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    
    total_returns = np.zeros(n_states)
    visit_counts = np.zeros(n_states)
    for ep in episodes:
        g = 0 
        _ = set()
        firsts = []
        for state, reward in reversed(ep):
            g = reward + gamma * g 
            firsts.append((state, g))

        for state, val in reversed(firsts):
            if state not in _:
                _.add(state)
                total_returns[state] += val 
                visit_counts[state] += 1
    V = np.zeros(n_states)
    nonzero_idx = visit_counts > 0 
    V[nonzero_idx] = total_returns[nonzero_idx] / visit_counts[nonzero_idx]
    return V