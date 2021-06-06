import numpy as np
from qutip import *
import random


class Qgate_env:

    def __init__(self, T, npulses, rand_field = False, actions = None):

        self.rand_field = rand_field
        self.max_steps = npulses
        self.dt = T / self.max_steps

        # Default actions are pi-rotations
        if actions == None:
            self.actions = [qeye(2), 1j*sigmax(), 1j*sigmay(), 1j*sigmaz()]
        else:
            err_msg = "Input 'actions' is not a list."
            assert isinstance(actions, list), err_msg
            self.actions = actions

        self.reset()
        self.action_space = len(self.actions)
        self.observation_space = self.state.shape

    def reset(self):

        if self.rand_field:
            chi = np.random.uniform(0, np.pi/self.dt)
            phi = np.random.uniform(0, 2*np.pi)
            theta = np.arccos(1 - 2*np.random.uniform(0, 1))

        else:
            chi = 1/2
            phi = random.choice([0, np.pi/2])
            theta = np.random.uniform(0, np.pi)

        nx = np.sin(theta)*np.cos(phi)
        ny = np.sin(theta)*np.sin(phi)
        nz = np.cos(theta)

        self.U = np.cos(chi)*qeye(2) + 1j*np.sin(chi)*(nx*sigmax() + ny*sigmay() + nz*sigmaz())

        self.steps = 0
        self.cum_reward = 0

        self.sequence = np.zeros(self.max_steps)

        self.q_state = self.U

        state_real = np.real(np.reshape(self.q_state[:], [1,4])[0])
        state_imag = np.imag(np.reshape(self.q_state[:], [1,4])[0])

        self.state = np.append(state_real, state_imag)
        self.state = np.pad(self.state, (0, 1), 'constant')
        self.state = np.concatenate([self.state, self.sequence])

        return self.state.astype('float32')


    def step(self, action):
        self.sequence[self.steps] = action + 1
        self.steps += 1
        reward = 0

        self.q_state = self.actions[action] * self.q_state

        if not self.steps == self.max_steps:
            self.q_state = self.U * self.q_state

        next_state_real = np.real(np.reshape(self.q_state[:], [1, 4])[0])
        next_state_imag = np.imag(np.reshape(self.q_state[:], [1, 4])[0])

        self.state = np.append(next_state_real, next_state_imag)
        self.state = np.pad(self.state, (0, 1), 'constant', constant_values=(0, self.steps))
        self.state = np.concatenate([self.state, self.sequence])

        if self.steps == self.max_steps:
            reward += 100*average_gate_fidelity(self.q_state)
            done = True

        else:
            done = False

        self.cum_reward += reward

        return self.state.astype('float32'), reward, done
