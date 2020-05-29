import numpy as np
import random
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet50
import copy
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import pylab
import matplotlib.pyplot as plt
from torchvision import datasets

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
batch_size = 128
device = 'cuda'

class MNISTDataset(Dataset):
    
    def __init__(self, transform=None):
        self.mnist = fetch_openml('mnist_784', version=1,)
        self.data = self.mnist.data.reshape(-1, 28, 28).astype('uint8')
        self.target = self.mnist.target.astype(int)
        self.indices = range(len(self))
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #idx2 = random.choice(self.indices)
        data1 = self.data[idx]
        #data2 = self.data[idx2]
        target1 = torch.from_numpy(np.array(self.target[idx]))
        #target2 = torch.from_numpy(np.array(self.target[idx2]))
        if self.transform:
            data1 = self.transform(data1)
            target1 = torch.from_numpy(np.array(self.target[idx])) #torch.from_numpy()でTensorに変換
                
        #sample = (data1, data2, target1, target2)
        sample = data1
        return data1, target1

transform = transforms.ToTensor()
train_data = MNISTDataset(transform=ToTensor())

n_samples = len(train_data) # n_samples is 60000
train_size = int(len(train_data) * 0.8) # train_size is 48000
val_size = n_samples - train_size # val_size is 48000

# shuffleしてから分割してくれる.
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
print(len(train_dataset)) # 48000
print(len(val_dataset)) # 12000

train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = True)

valid_loader = DataLoader(val_dataset,  
                          batch_size = batch_size,
                          shuffle = True)


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.dense_enc1 = nn.Linear(28*28, 200)
        self.dense_enc2 = nn.Linear(200, 100)
        self.dense_encmean = nn.Linear(100, z_dim)
        self.dense_encvar = nn.Linear(100, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 100)
        self.dense_dec2 = nn.Linear(100, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)
 
    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        var = self.dense_encvar(x)
        #var = F.softplus(self.dense_encvar(x))
        return mean, var
 
    def _sample_z(self, mean, var): #普通にやると誤差逆伝搬ができないのでReparameterization Trickを活用
        epsilon = torch.randn(mean.shape).to(device)
        #return mean + torch.sqrt(var) * epsilon #平均 + episilonは正規分布に従う乱数, torc.sqrtは分散とみなす？平均のルート
        return mean + epsilon * torch.exp(0.5*var)
        # イメージとしては正規分布の中からランダムにデータを取り出している
        #入力に対して潜在空間上で類似したデータを復元できるように学習, 潜在変数を変化させると類似したデータを生成
        #Autoencoderは決定論的入力と同じものを復元しようとする
 
 
    def _decoder(self,z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = F.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, z
    
    def loss(self, x): #lossは交差エントロピーを採用している, MSEの事例もある
        #https://tips-memo.com/vae-pytorch#i-7, http://aidiary.hatenablog.com/entry/20180228/1519828344のlossを参考 
        mean, var = self._encoder(x)
        #KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var)) #オリジナル, mean意味わからんけど, あんまり値が変わらないか>ら
        #上手くいくんじゃないか
        #KL = 0.5 * torch.sum(torch.exp(var) + mean**2 - 1. - var)
        KL = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp()) 
        # sumを行っているのは各次元ごとに算出しているため
        #print("KL: " + str(KL))
        z = self._sample_z(mean, var)
        y = self._decoder(z)
        #delta = 1e-8
        #reconstruction = torch.mean(torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta)))
        #reconstruction = F.binary_cross_entropy(y, x.view(-1, 784), size_average=False)
        reconstruction = F.binary_cross_entropy(y, x, size_average=False)
        #交差エントロピー誤差を利用して, 対数尤度の最大化を行っている, 2つのみ=(1-x), (1-y)で算出可能
        #http://aidiary.hatenablog.com/entry/20180228/1519828344(参考記事)
        #print("reconstruction: " + str(reconstruction))
        #lower_bound = [-KL, reconstruction]
        #両方とも小さくしたい, クロスエントロピーは本来マイナス, KLは小さくしたいからプラスに変換
        #returnで恐らくわかりやすくするために, 目的関数から誤差関数への変換をしている
        #return -sum(lower_bound)
        return KL + reconstruction



def train(model, optimizer, i):
        losses = []
        model.train()
        for x, t in train_loader: #data, label
            x = x.view(x.shape[0], -1)  
            x = x.to(device)
            optimizer.zero_grad() #batchごとに勾配の更新
            y = model(x)
            loss = model.loss(x) / batch_size
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        print("Epoch: {} train_loss: {}".format(i, np.average(losses)))
 
def test(model, optimizer, i):
    losses = []
    model.eval()
    with torch.no_grad():
        for x, t in valid_loader: #data, label
            x = x.view(x.shape[0], -1)  
            x = x.to(device)                       
            y = model(x)                           
            loss = model.loss(x) / batch_size          
            losses.append(loss.cpu().detach().numpy())                 
    print("Epoch: {} test_loss: {}".format(i, np.average(losses)))
 
def main():
    model = VAE(2).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    #model.train()i
    for i in range(50): #num epochs
        train(model, optimizer, i)
        test(model, optimizer, i)
    torch.save(model.state_dict(), "model1/model0529_2_128.pth")

 
if __name__ == "__main__":
    main()                

