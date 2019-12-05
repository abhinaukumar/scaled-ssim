from torch import nn
import torch
import generators
import progressbar
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
    
    def forward(self, *inputs):
        pass

    def train(self, generator, num_train, num_epochs, lr = 1e-4):

        opt = torch.optim.RMSprop(self.parameters(),lr = lr)

        widgets=[
            progressbar.ETA(),
            progressbar.Bar(),
            ' ',progressbar.DynamicMessage('Loss'),
            ' ',progressbar.DynamicMessage('Epoch')
        ]
        with progressbar.ProgressBar(max_value = num_epochs, widgets=widgets) as bar:
            for n in range(num_epochs):
                tot_loss = 0
    
                for i in range(num_train):

                    [x,y] = next(generator)

                    opt.zero_grad()
                    pred = self.forward(x)
                    loss = torch.mean((pred - y)**2)
                    loss.backward()
                    opt.step()
                    tot_loss += torch.mean((torch.mean(pred) - torch.mean(y))**2).cpu().detach().item()
                
                bar.update(n,Loss = tot_loss/num_train, Epoch = str(n+1)+'/'+str(num_epochs))
                
    
class DistConvNet(Net):
    def __init__(self, ksize, *inputs):
        super(DistConvNet, self).__init__()
        self.ksize = ksize
        self.layer1 = nn.Conv2d(1,15,(self.ksize,self.ksize),padding=(int((self.ksize-1)/2),int((self.ksize-1)/2)))
        self.layer2 = nn.Conv2d(15,15,(self.ksize,self.ksize),padding=(int((self.ksize-1)/2),int((self.ksize-1)/2)))
        self.layer3 = nn.Conv2d(15,1,(self.ksize,self.ksize),padding=(int((self.ksize-1)/2),int((self.ksize-1)/2)))
        self.logistic_params = nn.Parameter(torch.zeros((5,1)).uniform_(0,1).float())

    def logistic(self, x):
        y = self.logistic_params[0] * (0.5 - 1.0/(1 + torch.exp(self.logistic_params[1]*(x - self.logistic_params[2]))) + self.logistic_params[3] * x + self.logistic_params[4])
        return y

    def forward(self, x):
        h = nn.ReLU()(self.layer1.forward(x))
        h = nn.ReLU()(self.layer2.forward(h))
        y = nn.Sigmoid()(self.layer3.forward(h))
        return y*x

class FCNet(Net):
    def __init__(self, num_feats, hsize, *inputs):
        super(FCNet, self).__init__()
        self.num_feats = num_feats
        self.hsize = hsize
        self.layer1 = nn.Linear(self.num_feats,self.hsize)
        self.layer2 = nn.Linear(self.hsize,1)

    def forward(self,x):
        h = nn.ReLU()(self.layer1.forward(x))
        y = nn.Sigmoid()(self.layer2.forward(h))
        return y

class TwoFeatProdNet(Net):

    def __init__(self, *inputs):

        super(TwoFeatProdNet, self).__init__()
        self.exponents = nn.Parameter(torch.zeros((1,1)).uniform_(0,1).float())

    def forward(self, x, mode = 'train'):

        y = (self.exponents[0]*torch.log(x[:,0]) + (1 - self.exponents[0])*torch.log(x[:,1])).unsqueeze(-1)

        return y if mode == 'train' else torch.exp(y)

model_class = {"DistConvNet": DistConvNet, "FCNet": FCNet, "TwoFeatProdNet": TwoFeatProdNet}
data_generator = {"DistConvNet": generators.DistMapGen, "FCNet": generators.SSIMDataGen, "TwoFeatProdNet": generators.LogSSIMDataGen}