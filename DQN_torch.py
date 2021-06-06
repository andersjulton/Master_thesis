import numpy as np
import random
import torch.nn.functional as F
import torch

from tqdm import tqdm

from utils.memory_buffer import MemoryBuffer
from torchnet import NeuralNet


class DDQN:

    # Initializer for the agent class
    def __init__(self, state_size, action_size, args):
        CUDA = torch.cuda.is_available()

        self.device = torch.device('cuda:0' if CUDA else 'cpu')

        self.action_size = action_size
        self.state_size = state_size

        self.n_episodes = args['n_episodes']
        self.batch_size = args['batch_size']
        self.replace_period = args['replace_period']
        self.n_neurons = args['n_neurons']

        self.memory = MemoryBuffer(150000, self.n_episodes)

        self.gamma = 0.99
        self.learning_rate = 1e-3

        self.model = NeuralNet(state_size, action_size, self.n_neurons).to(self.device)
        self.target_model = NeuralNet(state_size, action_size, self.n_neurons).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss = torch.nn.MSELoss(reduction='mean')

    def train(self, env, epsilon):
        epsilon.clear()

        score = np.zeros(self.n_episodes)
        counter = 0
        step = 0

        sequences = []
        Us = []

        #Start training
        for e in tqdm(range(self.n_episodes)):
            counter += 1

            state = env.reset()
            Us.append(env.U.full())
            state = self.reshape(state)
            sequence = []

            for t in range(env.max_steps):
                step += 1
                epsilon_val = epsilon.get_epsilon(state)

                action = self.act(state, epsilon_val)
                sequence.append(action)
                next_state, reward, done = env.step(action)
                next_state = self.reshape(next_state)

                self.remember(state, action, reward, done, next_state)

                if epsilon.expert:
                    old_nQ = self.predict(next_state)[0]
                    G_Q = reward + self.gamma * np.argmax(old_nQ)
                    G_U = reward + self.gamma * np.mean(old_nQ)

                    old_Q = self.predict(state)[0][action]

                    if step > 5*self.batch_size:
                        self.replay()

                    new_Q = self.predict(state)[0][action]

                    epsilon.update_from_experts(state, data=(G_Q, G_U, new_Q - old_Q))
                else:
                    if step > min(self.n_episodes/10, 200):
                        self.replay()
                    epsilon.update_from_experts(state, data=(0, 0, 0))

                state = next_state
            sequences.append(sequence)
            score[e] = env.cum_reward
            epsilon.update_end_of_episode(e)

            # Update the target model
            if counter == self.replace_period:
                for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(local_param.data)
                counter = 0
        self.clear()
        return score, sequences, Us

    # Performing a random action, or a greedy action if a random number is <= current value of epsilon
    def act(self, state, epsilon_val):

        if np.random.rand() <= epsilon_val:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.predict(state)[0])


    def replay(self):
        """ Take random batch of experiences from memory and update the network
        """
        state, action, reward, done, next_state, idx, weights = self.memory.sample_batch(self.batch_size)

        q = self.predict(state)
        next_q = self.predict(next_state)
        q_targ = self.target_predict(next_state)

        for i in range(self.batch_size):
            old_q = q[i,0,action[i]]
            if done[i]:
                q[i,0,action[i]] = reward[i]
            else:
                next_best_action = np.argmax(next_q[i,0,:])
                q[i,0,action[i]] = reward[i] + self.gamma * q_targ[i,0,next_best_action]

            self.memory.update(idx[i], abs(old_q - q[i,0,action[i]]))

        self.fit(state, q, weights)


    def fit(self, inp, targ, weights):

        """ Perform one epoch of training
        """
        inp = torch.from_numpy(inp).to(self.device)
        targ = torch.from_numpy(targ).to(self.device)
        self.model.train()

        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(inp)

        loss = torch.FloatTensor(weights).to(self.device) * self.loss(pred, targ)
        loss = loss.mean()
        loss.backward()

        self.optimizer.step()


    def predict(self, inp):
        """ Q-Value Prediction
        """

        inp = torch.from_numpy(inp).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(inp)

        return pred.cpu().numpy()

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        inp = torch.from_numpy(inp).to(self.device)

        self.target_model.eval()
        with torch.no_grad():
            pred = self.target_model(inp)
        return pred.cpu().numpy()


    def reshape(self, x):
        if len(x.shape) < 4 and len(self.state_size) > 2: return np.expand_dims(x, axis=0)
        elif len(x.shape) < 3: return np.expand_dims(x, axis=0)
        else: return x

    # Adding experiences to the memory
    def remember(self, state, action, reward, done, next_state):

        q_val = self.predict(state)
        q_val_t = self.target_predict(next_state)
        next_best_action = np.argmax(self.predict(next_state))
        new_val = reward + self.gamma * q_val_t[0, next_best_action]
        td_error = abs(new_val - q_val)[0]

        self.memory.memorize(state, action, reward, done, next_state, td_error)

    # Load neural network parameters
    def load(self, name):
        self.model.load_state_dict(torch.load(name,map_location=torch.device('cpu')))

    # Save neural network parameters
    def save(self,name):
        torch.save(self.model.state_dict(), name)

    # Clear memory
    def clear(self):
        del self.memory
