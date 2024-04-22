"""
Coordinate prediction on the synthetic dataset, results as mean +- standard variance over 10 runs.
"""
import os
# --------------------------------------------------------------------------------

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ---------------------------
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as scio
import time
from PH_Reg import PH_Reg

class MLP(nn.Module):

    def __init__(self, m=100):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(m, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 100),
                                     nn.ReLU())
        self.regression_layer = nn.Linear(100, 3)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        pred = self.regression_layer(features)
        return pred, features


def main(dataset_path, flag, description="test", num_training=100, lambda_t=100, lambda_d=10):

    lr = 1e-3
    epochs = 5000
    ph_reg = PH_Reg()
    model = MLP().cuda()

    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = dataset.item()

    data = dataset['data'][:3000,:]
    label = dataset['label'][:3000,:]
    X_train = data[:num_training, :]
    y_train = label[:num_training, :]
    X_test = data[num_training:, :]
    y_test = label[num_training:, :]

    X_train = Variable(torch.from_numpy(X_train), requires_grad=True).float().cuda()
    y_train = Variable(torch.from_numpy(y_train), requires_grad=True).float().cuda()
    X_test = Variable(torch.from_numpy(X_test), requires_grad=True).float().cuda()
    y_test = Variable(torch.from_numpy(y_test), requires_grad=True).float().cuda()

    mse_loss = nn.MSELoss().cuda()

    l_train = []
    l_test = []
    for times in range(10):   # run 10 times
        begin = time.time()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        _mse_train = 9999
        _mse_test = 9999
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred, feature = model(X_train)
            loss = mse_loss(pred, y_train)
            if flag == 0:
                loss_oe = loss * 0
            else:
                loss_oe = ph_reg(feature, y_train, min_points=19, point_jump=20, max_points=80, flag=flag)
                loss_oe = lambda_t * loss_oe[0] + lambda_d * loss_oe[1]
            loss_all = loss + loss_oe
            loss_all.backward()
            optimizer.step()
            if epoch % 100 == 0:
                model.eval()
                pred, feature = model(X_test)
                loss_test = mse_loss(pred, y_test)
                print('{0}, Epoch: [{1}]\t'
                      'Loss_train: [{loss:.2e}]\t'
                      'Loss_test: [{loss_test:.2e}]\t'
                      'Loss_entropy: [{loss_e:.2e}]\t'
                      .format(description, epoch, loss=loss.data, loss_test=loss_test.data, loss_e=loss_oe.data))
                if loss_test < _mse_test:
                    _mse_test = loss_test
                    _mse_train = loss
                    print('best model, Loss_test: [{loss_test:.2e}]'.format(
                        loss_test=_mse_test.data))

        l_test.append(_mse_test.cpu().detach().numpy())
        l_train.append(_mse_train.cpu().detach().numpy())
        end = time.time()
        print(end-begin)

    l_train = np.array(l_train)
    l_test = np.array(l_test)
    train_dict = {}
    train_dict['train_mse'] = l_train
    train_dict['test_mse'] = l_test
    path = './'+description + '.mat'
    scio.savemat(path, train_dict)
    print('Mean: \t')
    print(np.mean(l_test))
    print('Std: \t')
    print(np.std(l_test))

if __name__ == "__main__":
    """
    flag ==0: baseline
    flag ==1: only sig_error(sig2, sig1_2)
    flag ==2: only phd dimention (L_d)
    flag ==3: phd dimention (L_d) + sig_error(sig2, sig1_2)
    flag ==4: phd dimention (L_d)+ Topology autoencoder (L_t)
    flag ==5: original ph (L'_d)
    figa ==6: Topology autoencoder (L_t)
    """
    # dataset_path = './mammoth/mammoth.npy'
    dataset_path = './data/swiss_roll.npy'
    # dataset_path = './data/torus.npy'
    # dataset_path = './data/circle.npy'
    flag = 0
    description = 'test'
    num_training = 100
    lambda_t = 100
    lambda_d = 10

    main(dataset_path, flag, description, num_training, lambda_t, lambda_d)