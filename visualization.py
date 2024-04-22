"""
Visualization
"""
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PH_Reg import PH_Reg
from sklearn import manifold
from sklearn.decomposition import PCA

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
    epochs = 3000

    ph_reg = PH_Reg()

    model = MLP().cuda()

    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = dataset.item()
    data = dataset['data'][:3000,:]
    label = dataset['label'][:3000,:]
    if 'mammoth' in dataset_path:
        color = label[:3000,2]
    elif 'torus' in dataset_path:
        color = label[:3000, 2]
    else:
        color = dataset['color'][:3000]

    X_train = data[:num_training, :]
    y_train = label[:num_training, :]
    X_test = data[num_training:, :]
    y_test = label[num_training:, :]

    X_train = Variable(torch.from_numpy(X_train), requires_grad=True).float().cuda()
    y_train = Variable(torch.from_numpy(y_train), requires_grad=True).float().cuda()
    X_test = Variable(torch.from_numpy(X_test), requires_grad=True).float().cuda()
    y_test = Variable(torch.from_numpy(y_test), requires_grad=True).float().cuda()

    mse_loss = nn.MSELoss().cuda()


    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    _mse_test=999
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

    model = model.eval()
    pred, feature = model(X_train)
    pred2, feature2 = model(X_test)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.scatter(
        label[:, 0], label[:, 1], label[:, 2],c=color, vmin=np.min(color), vmax=np.max(color), cmap='brg', s=5, alpha=0.5
    )
    ax.set_title("Visualization of the target space")
    ax.view_init(azim=-66, elev=12)
    _ = ax.text2D(0.8, 0.05, s="dataset", transform=ax.transAxes)



    pred = pred.cpu().detach().numpy()
    pred2 = pred2.cpu().detach().numpy()
    feature = feature.cpu().detach().numpy()
    feature2 = feature2.cpu().detach().numpy()

    feature = np.concatenate([feature,feature2],axis=0)

    sr_tsne = manifold.TSNE(n_components=3, perplexity=30).fit_transform(feature)

    # sr_tsne, sr_err = manifold.locally_linear_embedding(
    #     feature, n_neighbors=12, n_components=3
    # )

    # pca = PCA(n_components=3, svd_solver='arpack')
    # pca.fit(feature)
    # sr_tsne =pca.transform(feature)

    fig2 = plt.figure(figsize=(8, 6))
    ax = fig2.add_subplot(111, projection="3d")
    fig2.add_axes(ax)
    ax.scatter(
        pred[:, 0], pred[:, 1], pred[:, 2],c=color[:num_training], vmin=np.min(color), vmax=np.max(color), cmap='brg', s=5, alpha=0.5
    )
    ax.set_title("Predicted Y from the training set")
    ax.view_init(azim=-66, elev=12)
    _ = ax.text2D(0.8, 0.05, s="training", transform=ax.transAxes)

    fig3 = plt.figure(figsize=(8, 6))
    ax = fig3.add_subplot(111, projection="3d")
    fig3.add_axes(ax)
    ax.scatter(
        pred2[:, 0], pred2[:, 1], pred2[:, 2],c=color[num_training:], vmin=np.min(color), vmax=np.max(color), cmap='brg', s=5, alpha=0.5
    )
    ax.set_title("Predicted Y from the test set")
    ax.view_init(azim=-66, elev=12)
    _ = ax.text2D(0.8, 0.05, s="test", transform=ax.transAxes)


    fig4 = plt.figure(figsize=(8, 6))
    ax = fig4.add_subplot(111, projection="3d")
    fig4.add_axes(ax)
    ax.scatter(
        sr_tsne[:num_training, 0], sr_tsne[:num_training, 1], sr_tsne[:num_training, 2], c=color[:num_training], vmin=np.min(color), vmax=np.max(color), cmap='brg', s=5, alpha=0.5
    )
    ax.set_title("Visualization of the feature space (training set)")
    ax.view_init(azim=-66, elev=12)
    _ = ax.text2D(0.8, 0.05, s="feature_training", transform=ax.transAxes)


    fig5 = plt.figure(figsize=(8, 6))
    ax = fig5.add_subplot(111, projection="3d")
    fig5.add_axes(ax)
    ax.scatter(
        sr_tsne[num_training:, 0], sr_tsne[num_training:, 1], sr_tsne[num_training:, 2], c=color[num_training:], vmin=np.min(color), vmax=np.max(color), cmap='brg', s=5, alpha=0.5
    )
    ax.set_title("Visualization of the feature space (test set)")
    ax.view_init(azim=-66, elev=12)
    _ = ax.text2D(0.8, 0.05, s="feature_test", transform=ax.transAxes)

    plt.show()




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
    # dataset_path = './data/mammoth.npy'
    # dataset_path = './data/swiss_roll.npy'
    # dataset_path = './data/torus.npy'
    dataset_path = './data/circle.npy'
    flag = 5
    description = 'test'
    num_training = 100
    lambda_t = 100
    # lambda_d = 10
    lambda_d = 1

    main(dataset_path, flag, description, num_training, lambda_t, lambda_d)
