"""
Coordinate prediction on the synthetic dataset, results as mean +- standard variance over 10 runs.

Information Dropout is modified based on its source code:
https://github.com/ucla-vision/information-dropout
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
from ordinal_entropy import ordinal_entropy

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


def sample_lognormal(mean, sigma=None, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backpropagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)
    """
    e = torch.normal(0., 1., size=mean.size()).cuda()
    return torch.exp(mean + sigma * sigma0 * e)


class MLP_InformationDropout(nn.Module):

    def __init__(self, m=100, max_alpha=0.7, sigma0=1.):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(m, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 100),
                                     nn.ReLU())
        self.regression_layer = nn.Linear(100, 3)
        self.alpha_conv = nn.Sequential(nn.Linear(m, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 100),
                                     nn.ReLU())
        self.max_alpha = max_alpha
        self.sigma0 = sigma0
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
        alpha = self.max_alpha * torch.sigmoid(self.alpha_conv(x))
        alpha = 0.001 + self.max_alpha * alpha
        kl = -torch.log(alpha / (self.max_alpha + 0.001))
        if self.training:
            self.sigma0 = 1
        else:
            self.sigma0 = 0
        e = sample_lognormal(mean=torch.zeros_like(features), sigma=alpha, sigma0=self.sigma0)
        features = features * e
        pred = self.regression_layer(features)
        return pred, kl

def main(dataset_path, flag, description="test", num_training=100, num_validation=100, lambda_t=100, lambda_d=10):
    lr = 1e-3
    epochs = 10000
    ph_reg = PH_Reg()
    if flag == 8:
        model = MLP_InformationDropout().cuda()
    else:
        model = MLP().cuda()

    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = dataset.item()

    data = dataset['data'][:3000, :]
    label = dataset['label'][:3000, :]
    X_train = data[:num_training, :]
    y_train = label[:num_training, :]
    X_val = data[num_training:num_training + num_validation, :]
    y_val = label[num_training:num_training + num_validation, :]
    X_test = data[num_training + num_validation:, :]
    y_test = label[num_training + num_validation:, :]

    X_train = Variable(torch.from_numpy(X_train), requires_grad=True).float().cuda()
    y_train = Variable(torch.from_numpy(y_train), requires_grad=True).float().cuda()
    X_val = Variable(torch.from_numpy(X_val), requires_grad=True).float().cuda()
    y_val = Variable(torch.from_numpy(y_val), requires_grad=True).float().cuda()
    X_test = Variable(torch.from_numpy(X_test), requires_grad=True).float().cuda()
    y_test = Variable(torch.from_numpy(y_test), requires_grad=True).float().cuda()

    mse_loss = nn.MSELoss().cuda()

    l_train = []
    l_test = []
    for times in range(10):  # run 10 times
        begin = time.time()
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        _mse_train = 9999
        _mse_val = 9999
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred, feature = model(X_train)
            loss = mse_loss(pred, y_train)
            if flag == 0:   # baseline
                loss_oe = loss * 0
            elif flag ==7:   # Ordinal entropy
                loss_oe = ordinal_entropy(feature, y_train) * 0.1   # 0.1 generally has the best performance within the set {0.01, 0.1, 1, 10}
            elif flag ==8:  # infromation Dropout
                loss_oe = torch.sum(feature) /num_training * 0.01   # 0.01 generally has the best performance within the set {0.01, 0.1, 1, 10}
            else:
                loss_oe = ph_reg(feature, y_train, min_points=19, point_jump=20, max_points=80, flag=flag)
                loss_oe = lambda_t * loss_oe[0] + lambda_d * loss_oe[1]
            loss_all = loss + loss_oe
            loss_all.backward()
            optimizer.step()
            if epoch % 100 == 0:
                model.eval()
                pred, feature = model(X_val)
                loss_val = mse_loss(pred, y_val)
                print('{0}, Epoch: [{1}]\t'
                      'Loss_train: [{loss:.2e}]\t'
                      'Loss_val: [{loss_test:.2e}]\t'
                      'Loss_entropy: [{loss_e:.2e}]\t'
                      .format(description, epoch, loss=loss.data, loss_test=loss_val.data, loss_e=loss_oe.data))
                if loss_val < _mse_val:
                    _mse_val = loss_val
                    _mse_train = loss
                    torch.save(model.state_dict(),
                               './%s.pth' % (description))
                    print('best model, Loss_val: [{loss_val:.2e}]'.format(
                        loss_val=_mse_val.data))

        model.load_state_dict(torch.load('./%s.pth' % (description)))
        print("success load the best model")
        model.eval()
        pred, feature = model(X_test)
        loss_test = mse_loss(pred, y_test)
        print('Loss_test: [{loss_test:.2e}]'
              .format(loss_test=loss_test.data))

        l_test.append(loss_test.cpu().detach().numpy())
        l_train.append(_mse_train.cpu().detach().numpy())
        end = time.time()
        print(end - begin)

    l_train = np.array(l_train)
    l_test = np.array(l_test)
    train_dict = {}
    train_dict['train_mse'] = l_train
    train_dict['test_mse'] = l_test
    path = './' + description + '.mat'
    scio.savemat(path, train_dict)
    print("{0}: Test over 10 runs: \t".format(description))
    print(l_test)
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
    flag ==5: original phd dimention (L'_d)
    flga ==6: Topology autoencoder (L_t)
    figa ==7: Ordinal Entropy
    flag ==8: Infromation Dropout
    """
    # task = 'mammoth'
    # task = 'swiss_roll'
    # task = 'torus'
    task = 'circle'

    dataset_path = './data/' + task + '.npy'
    flag = 4
    description = task + str(flag)
    num_training = 100
    num_validation =100
    lambda_t = 100
    lambda_d = 10

    if task == 'mammoth':
        lambda_t = 10000
        lambda_d = 1
    elif task =='torus':
        lambda_d =1
    elif task == 'circle':
        lambda_d =1

    main(dataset_path, flag, description, num_training, num_validation, lambda_t, lambda_d)
