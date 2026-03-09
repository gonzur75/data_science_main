import numpy as np


rng = np.random.default_rng(seed=42)


class ThompsonSamplingAgent:
    def __init__(self, n_actions: int):
        if n_actions <= 0:
            raise ValueError("n_actions must be > 0")
        self.actions_alphas = np.ones(n_actions, dtype=np.float64)
        self.actions_betas = np.ones(n_actions, dtype=np.float64)

    def select_action(self):
        samples = np.array(
            [
                rng.beta(self.actions_alphas[i], self.actions_betas[i])
                for i in range(len(self.actions_alphas))
            ]
        )
        return int(np.argmax(samples))

    def update(self, action, reward):
        if reward == 1:
            self.actions_alphas[action] += 1
        else:
            self.actions_betas[action] += 1


class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def pull(self):
        return 1 if rng.random() < self.p else 0


def run_simulation(steps=1000, return_details=False):
    arm_probs = [0.3, 0.3, 0.33]

    arms = [BernoulliArm(p) for p in arm_probs]
    agent = ThompsonSamplingAgent(n_actions=len(arms))
    rewards = []
    action_counts = np.zeros(len(arms), dtype=np.int64)
    success_counts = np.zeros(len(arms), dtype=np.int64)

    for _ in range(steps):
        action = agent.select_action()
        action_counts[action] += 1
        reward = arms[action].pull()
        success_counts[action] += reward
        agent.update(action, reward)
        rewards.append(reward)

    if return_details:
        return rewards, action_counts, success_counts, arm_probs
    return rewards


if __name__ == "__main__":
    rewards, action_counts, success_counts, arm_probs = run_simulation(return_details=True)
    best_machine = int(np.argmax(action_counts))

    print(f"Total reward: {sum(rewards)}")
    print(
        f"Najczesciej wybierana maszyna: {best_machine + 1} "
        f"(wyborów: {int(action_counts[best_machine])})"
    )
    for idx, p_true in enumerate(arm_probs):
        pulls = int(action_counts[idx])
        wins = int(success_counts[idx])
        p_empirical = wins / pulls if pulls > 0 else 0.0
        print(
            f"Maszyna {idx + 1}: true_p={p_true:.2f}, wybory={pulls}, "
            f"trafienia={wins}, empirical_p={p_empirical:.3f}"
        )