import numpy as np

K_VALUES_FILE = "k_values.npy"
SIL_VALUES_FILE = "silhouette_values.npy"

class KEnv:
    def __init__(self):
        self.ks = np.load(K_VALUES_FILE)
        self.sils = np.load(SIL_VALUES_FILE)
        self.k_min = int(self.ks.min())
        self.k_max = int(self.ks.max())
        self.k = None
        self.sil = None

    def reset(self, k_start=None):
        if k_start is None:
            self.k = np.random.randint(self.k_min, self.k_max + 1)
        else:
            self.k = int(k_start)
        self.sil = self._sil_for_k(self.k)
        return self.k

    def step(self, action):
        old_k = self.k
        old_sil = self.sil

        if action == 0:
            self.k = max(self.k_min, self.k - 1)
        elif action == 2:
            self.k = min(self.k_max, self.k + 1)

        self.sil = self._sil_for_k(self.k)
        reward = self.sil - old_sil
        done = False
        info = {"k": self.k, "silhouette": self.sil}
        return self.k, reward, done, info

    def _sil_for_k(self, k):
        idx = np.where(self.ks == k)[0][0]
        return float(self.sils[idx])

class QAgent:
    def __init__(self, k_min, k_max, n_actions=3,
                 alpha=0.3, gamma=0.9, epsilon=1.0, eps_min=0.05, eps_dec=0.995):
        self.k_min = k_min
        self.k_max = k_max
        self.n_states = k_max + 1
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.Q = np.zeros((self.n_states, self.n_actions))

    def choose_action(self, state_k):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_k])

    def learn(self, s, a, r, s_next):
        q_pred = self.Q[s, a]
        q_next = np.max(self.Q[s_next])
        q_target = r + self.gamma * q_next
        self.Q[s, a] = q_pred + self.alpha * (q_target - q_pred)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

def train_agent(n_episodes=200):
    env = KEnv()
    agent = QAgent(k_min=env.k_min, k_max=env.k_max)

    best_ks = []
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0

        # episod scurt, ex. 5 pași
        for t in range(5):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        best_ks.append(state)

        print(f"Episode {ep+1:3d}: final K={state}, total_reward={total_reward:.4f}, epsilon={agent.epsilon:.3f}")

    return env, agent, best_ks

if __name__ == "__main__":
    env, agent, ks_traj = train_agent(n_episodes=200)

    ks_traj = np.array(ks_traj)
    unique, counts = np.unique(ks_traj, return_counts=True)
    print("\n[INFO] Final K distribution over episodes:")
    for k, c in zip(unique, counts):
        print(f"  K={k}: {c} episoade")

    best_k_static = env.ks[np.argmax(env.sils)]
    print(f"\n[INFO] Best K (static Silhouette): {best_k_static}")
