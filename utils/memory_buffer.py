import random
import numpy as np

from .sumtree import SumTree

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Prioritized Experience Replay
    using a Sum Tree
    """
    def __init__(self, buffer_size, n_episodes):
        """ Initialization
        """
        # Prioritized Experience Replay
        self.alpha = 0.6
        self.epsilon = 0.01
        self.beta0 = 0.4
        self.beta_inc = (1 - self.beta0)//n_episodes
        self.beta = self.beta0
        self.max_priority = 1.
        self.tree = SumTree(buffer_size)


        self.tree_size = buffer_size

    def memorize(self, state, action, reward, done, new_state, error=None):
        """ Save an experience to memory
        """

        experience = (state, action, reward, done, new_state)
        p = self.tree.max_p
        if not p:
            p = self.max_priority
        self.tree.add(p, experience)

    def sample_batch(self, batch_size):
        """ Sample a batch
        """
        batch = []
        indices = []
        priorities = []

        T = self.tree.total / batch_size
        if T < 1.0:
            T = 1.0
        for i in range(batch_size):
            while True:
                a = T * i
                b = T * (i + 1)
                s = random.uniform(a, b)

                idx, p, data = self.tree.get(s)

                if isinstance(data, (int, float)):
                    continue

                batch.append(data)
                indices.append(idx)
                priorities.append(p)
                break

        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])

        samp_prob = priorities / self.tree.total

        weights = (self.tree.n_entries*samp_prob)**(-self.beta)
        weights /= weights.max()

        self.betaupdate()

        return s_batch, a_batch, r_batch, d_batch, new_s_batch, indices, weights

    def update(self, idx, TDerror):
        """ Update priority for index
        """
        TDerror += self.epsilon
        TDerror = np.minimum(TDerror, self.max_priority)
        self.tree.update(idx, TDerror**self.alpha)

    def betaupdate(self):
        """Update beta exponent
        """
        if self.beta < 1.0:
            self.beta += self.beta_inc
        else:
            self.beta = 1.0

    def clear(self):
        """ Clear Sum Tree
        """
        self.tree = SumTree(self.tree_size)
