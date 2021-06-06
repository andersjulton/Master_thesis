from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np
from scipy.stats import t as tdist



class Epsilon(ABC):
    """
    Abstract class for the various epsilon-greedy policies.
    """

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_epsilon(self, state):
        pass

    @abstractmethod
    def update_from_experts(self, state, data):
        pass

    @abstractmethod
    def update_end_of_episode(self, episode):
        pass


class Fixed(Epsilon):
    """
    Fixed epsilon value without decay.
    """


    def __init__(self, value):
        self.value = value
        self.expert = False

    def clear(self):
        pass

    def get_epsilon(self, state):
        return self.value

    def update_from_experts(self, state, data):
        pass

    def update_end_of_episode(self, episode):
        pass

    def __str__(self):
        return "Fixed"


class ExpDecay(Epsilon):
    """
    Exponential decay of initial epsilon value.

    Parameters
    ----------
    initial : float
        Initial epsilon value.
    episode_decay : float
        Decay rate for end of epsiode.
    step_decay : type
        Decay rate for each time step (optional).
    """


    def __init__(self, initial, episode_decay, eps_min=0, step_decay=1.0):
        self.initial = initial
        self.episode_decay = episode_decay
        self.eps_min = eps_min
        self.step_decay = step_decay
        self.epsilon = self.initial
        self.expert = False

    def clear(self):
        self.epsilon = self.initial

    def get_epsilon(self, state):
        return self.epsilon

    def update_from_experts(self, state, data):
        self.epsilon *= self.step_decay

    def update_end_of_episode(self, episode):
        if self.epsilon > self.eps_min:
            self.epsilon *= self.episode_decay

    def __str__(self):
        return "ExpDecay"

class StretchedExpDecay(Epsilon):
    """
    StretchedExpDecay Exponential decay of initial epsilon value.

    https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f

    Parameters
    ----------
    initial : float
        Initial epsilon value.
    episode_decay : float
        Decay rate for end of epsiode.
    step_decay : type
        Decay rate for each time step (optional).
    """


    def __init__(self, initial, n_episodes):
        self.initial = initial
        self.epsilon = self.initial
        self.expert = False
        self.n_episodes = n_episodes

    def clear(self):
        self.epsilon = self.initial

    def get_epsilon(self, state):
        return self.epsilon

    def update_from_experts(self, state, data):
        pass

    def update_end_of_episode(self, episode):
        A=0.5
        B=0.1
        C=0.1
        standardized_time = (episode-A*self.n_episodes)/(B*self.n_episodes)
        cosh = np.cosh(np.exp(-standardized_time))
        self.epsilon = self.initial - (1/cosh+(episode*C/self.n_episodes))

    def __str__(self):
        return "StretchedExpDecay"


class PLDecay(Epsilon):
    """
    Power law decay of initial epsilon value.

    ε(episode) = ε_0 / (episode + 1)^x

    Parameters
    ----------
    initial : float
        Initial epsilon value.
    power : float
        Power value for decay at end of episode.
    """

    def __init__(self, initial, power):
        self.initial = initial
        self.power = power
        self.expert = False


    def clear(self):
        self.epsilon = self.initial

    def get_epsilon(self, state):
        return self.epsilon

    def update_from_experts(self, state, data):
        pass

    def update_end_of_episode(self, episode):
        self.epsilon = self.initial / (episode + 1.) ** self.power

    def __str__(self):
        return "PLDecay"


class BMC(Epsilon):

    def __init__(self, alpha=1.0, beta=1.0, sigma_sq=None):
        self.alpha, self.beta = alpha, beta
        self.sigma_sq = sigma_sq
        self.expert = True

    def clear(self):
        self.stat = Average()
        self.post = Beta(self.alpha, self.beta)

    def get_epsilon(self, state):
        post = self.post
        return post.alpha / (post.alpha + post.beta)

    def update_from_experts(self, state, data):
        G_Q, G_U = data[0], data[1]
        epsilon = self.get_epsilon(state)
        G = (1.0 - epsilon) * G_Q + epsilon * G_U
        if self.sigma_sq is None:
            var = self.stat.update(G)
            if var <= 0.0:
                return
        else:
            var = self.sigma_sq
        normalizer = math.log(2.0 * math.pi * var)
        phi_q = math.exp(-0.5 * ((G - G_Q) * (G - G_Q) / var + normalizer))
        phi_u = math.exp(-0.5 * ((G - G_U) * (G - G_U) / var + normalizer))
        self.post.update(phi_u, phi_q)

    def update_end_of_episode(self, episode):
        pass

    def __str__(self):
        return "BMC"


class BMCR(Epsilon):

    def __init__(self, mu, tau, a, b, alpha=1.0, beta=1.0):
        self.mu0, self.tau0, self.a0, self.b0 = mu, tau, a, b
        self.alpha, self.beta = alpha, beta
        self.expert = True

    def clear(self):
        self.stat = Average()
        self.post = Beta(self.alpha, self.beta)

    def get_epsilon(self, state):
        post = self.post
        return post.alpha / (post.alpha + post.beta)

    def update_from_experts(self, state, data):

        # compute return
        G_Q, G_U = data[0], data[1]
        epsilon = self.get_epsilon(state)
        G = (1.0 - epsilon) * G_Q + epsilon * G_U

        # update mu-hat and sigma^2-hat
        self.stat.update(G)
        mu, sigma2, t = self.stat.mean, self.stat.var, self.stat.count

        # update a_t and b_t
        a = self.a0 + t / 2
        b = self.b0 + t / 2 * sigma2 + t / 2 * (self.tau0 / (self.tau0 + t)) * (mu - self.mu0) * (mu - self.mu0)

        # compute e_t
        scale = (b / a) ** 0.5
        e_u = tdist.pdf(G, df=2.0 * a, loc=G_U, scale=scale)
        e_q = tdist.pdf(G, df=2.0 * a, loc=G_Q, scale=scale)

        # update posterior
        self.post.update(e_u, e_q)

    def update_end_of_episode(self, episode):
        pass

    def __str__(self):
        return "BMCR"


class VDBE(Epsilon):
    """
    Value-Difference Based Exploration.
    """

    def __init__(self, initial, delta, sigma):
        self.initial = initial
        self.delta, self.sigma = delta, sigma
        self.expert = True

    def clear(self):
        self.epsilon = defaultdict(lambda: self.initial)

    def get_epsilon(self, state):
        if type(state) == np.ndarray:
            state = tuple(state[0])
        return self.epsilon[state]

    def update_from_experts(self, state, data):
        td_error = data[2]
        coeff = math.exp(-abs(td_error) / self.sigma)
        f = (1.0 - coeff) / (1.0 + coeff)
        if type(state) == np.ndarray:
            state = tuple(state[0])
        self.epsilon[state] = self.delta * f + (1.0 - self.delta) * self.epsilon[state]

    def update_end_of_episode(self, episode):
        pass

    def __str__(self):
        return "VDBE"

class Average:
    """
    Support class for the BMC method.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mean, self.m2, self.var, self.count = 0.0, 0.0, 0.0, 0

    def update(self, point):
        self.count += 1
        count = self.count
        delta = point - self.mean
        self.mean += delta / count
        self.m2 += delta * (point - self.mean)
        if count > 1:
            self.var = self.m2 / (count - 1.0)
        return self.var

class Beta:
    """
    Support class for the BMC method.
    """

    def __init__(self, alpha0, beta0):
        self.alpha = alpha0
        self.beta = beta0

    def update(self, expert1, expert2):
        alpha, beta = self.alpha, self.beta
        mean = expert1 * alpha + expert2 * beta
        if mean <= 0.0:
            return
        m = alpha / (alpha + beta + 1.) * (expert1 * (alpha + 1.) + expert2 * beta) / mean
        s = alpha / (alpha + beta + 1.) * (alpha + 1.) / (alpha + beta + 2.) * \
            (expert1 * (alpha + 2.) + expert2 * beta) / mean
        r = (m - s) / (s - m * m)
        self.alpha, self.beta = m * r, (1. - m) * r
