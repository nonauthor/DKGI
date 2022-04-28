import torch
import torch.nn as nn
from sklearn import svm

class LogReg(nn.Module):
    def __init__(self, ft_in, hidden , nb_classes):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(ft_in, hidden)
        self.fc2 = nn.Linear(hidden,nb_classes)
        for m in self.modules():
            self.weights_init(m)
        self.softmax = nn.Softmax()
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret1 = self.fc1(seq)
        ret2 = self.fc2(ret1)
        ret3 = self.softmax(ret2)
        return ret3


