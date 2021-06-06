import torch.nn as nn
import torch


class NeuralNet(nn.Module):

    def __init__(self, state_size, action_size, n_neurons):
        super(NeuralNet, self).__init__()

        state_size = state_size[0]
        self.Linear1 = nn.Linear(in_features = state_size, out_features = n_neurons, bias=True)
        self.Linear2 = nn.Linear(n_neurons, n_neurons, bias=True)
        self.Linear3 = nn.Linear(n_neurons*2, n_neurons, bias=True)
        self.Linear4 = nn.Linear(n_neurons*2, n_neurons, bias=True)
        self.Linear5 = nn.Linear(n_neurons*2, n_neurons, bias=True)
        self.Linear6 = nn.Linear(n_neurons*2, n_neurons, bias=True)
        self.Linear7 = nn.Linear(n_neurons, action_size, bias=True)


    def forward(self, x):
        x1 = nn.ReLU()(self.Linear1(x))
        x2 = nn.ReLU()(self.Linear2(x1))
        x3 = nn.ReLU()(self.Linear3(torch.cat([x1, x2], dim=-1)))
        x4 = nn.ReLU()(self.Linear4(torch.cat([x2, x3], dim=-1)))
        x5 = nn.ReLU()(self.Linear5(torch.cat([x3, x4], dim=-1)))
        x6 = nn.ReLU()(self.Linear6(torch.cat([x4, x5], dim=-1)))
        x7 = self.Linear7(x6)

        return x7
