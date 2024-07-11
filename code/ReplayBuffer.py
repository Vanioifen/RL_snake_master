class ReplayBuffer(object):
    def __init__(self, max_size=256):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
            
    def get_size(self):
        return len(self.storage)

    def sample(self, batch_size):
        ind = np.arange(0, len(self.storage))
        #print(ind)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_a_log_probs, batch_dws = [], [], [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done, a_log_prob, dw = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
            batch_a_log_probs.append(np.array(a_log_prob, copy=False))
            batch_dws.append(np.array(dw, copy=False))
            
            
            #print(state)
        self.storage = []
            
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards), np.array(batch_dones), np.array(batch_a_log_probs), np.array(batch_dws)
   