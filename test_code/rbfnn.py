import torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

class RBFN(nn.Module):
    def __init__(self, centers, n_out=10):
        super(RBFN_TS, self).__init__()
        self.n_out = n_out
        # self.n_in = centers.size(1)
        self.num_centers = centers.size(0)

        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1,self.num_centers), requires_grad = True)
        # self.linear = nn.Linear(self.num_centers + self.n_in, self.n_out, bias=True)
        self.linear = nn.Linear(self.num_centers, self.n_out, bias=True)
        self.initialize_weights()

    def kernel_fun(self, batches):
        n_input = batches.size(0) # number of inputs
        A = self.centers.view(self.num_centers,-1).repeat(n_input,1,1)
        B = batches.view(n_input,-1).unsqueeze(1).repeat(1,self.num_centers,1)
        C = torch.exp(-self.beta.mul((A-B).pow(2).sum(2,keepdim=False) ) )
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score
    
    def initialize_weights(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
    
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)

class RBFN_TS(object):
    def __init__(self, args):
        self.max_epoch = args.epoch
        self.trainset = args.dataset[0]
        self.testset = args.dataset[1]
        self.model_name = args.model_name
        self.lr = args.lr
        self.n_in = args.n_in
        self.n_out = args.n_out
        self.num_centers = args.num_centers
        #  self.center_id = np.random.choice(len(self.trainset[0]),self.num_centers,replace=False)
        #  self.centers = torch.from_numpy(self.trainset[0][self.center_id]).float()
        self.centers = torch.rand(self.num_centers,self.n_in)

        self.model = RBFN(self.centers, n_out=self.n_out)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fun = nn.MSELoss()

    def train(self, epoch=1):
        self.model.train()
        for epoch in range(min(epoch,self.max_epoch)):
            avg_cost = 0

            X = torch.from_numpy(self.trainset[0]).float()
            Y = torch.from_numpy(self.trainset[1]).float()        # label is not one-hot encoded

            self.optimizer.zero_grad()             
            Y_prediction = self.model(X)         
            cost = self.loss_fun(Y_prediction, Y) 
            cost.backward()                   
            self.optimizer.step()                  

            print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, cost.item()))
        print(" [*] Training finished!")

    def test(self):
        self.model.eval()
        X = torch.from_numpy(self.testset[0]).float()
        Y = torch.from_numpy(self.testset[1]).float()        # label is not one-hot encoded

        with torch.no_grad():             # Zero Gradient Container
            Y_prediction = self.model(X)         # Forward Propagation
            cost = self.loss_fun(Y_prediction, Y[:,:3])

            print('Accuracy of the network on test data: %f' % cost.item())
            print(" [*] Testing finished!")

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


args = Dict(
    lr = 0.01,
    epoch = 1000,
    n_in = 3*n,
    n_out = 3,
    num_centers = 128,  # 100
    save_dir = 'ckpoints',
    result_dir = 'outs',
    dataset = [(x_train.T, y_train.T), (x_test.T, y_test.T)],
    model_name='RBFN',
    cuda=False
)

rbfn = RBFN_TS(args)
rbfn.train(1000)
rbfn.test()

Y_true = np.vstack([select_samples(data, test_start + n + i, num_test) for i in range(horizon)])
print(Y_true.shape)  # (60, 2000)

horizon = 30  # 多步预测，相前预测30步

with torch.no_grad():
    for i in np.random.choice(len(x_test.T),10,replace=False):
        x = torch.from_numpy(x_test.T[i:i+1]).float()
        pred = []
        for _ in range(Y_true.shape[0]//3):
            y = rbfn.model(x)
            x = torch.from_numpy(np.hstack([x[:,3:],y]))
            pred.append(y.numpy()[0])
        pred = np.hstack(pred)
        
        A = pred
        B = Y_true[:,i:i+1]
        X = np.vstack((x_test[:,i:i+1],B[:3]))
        plt.figure()
        ylabel = ['x','y','z']
        for j in range(3):
            plt.subplot(3,1,j+1)
            plt.plot(range(n+1),X[j::3], 'b', label='input')
            plt.plot(range(n, n+horizon),B[j::3], 'k',label='ground truth')
            plt.plot(range(n, n+horizon),A[j::3], 'r--',label='prediction')
            plt.ylabel(ylabel[j])
            plt.legend(loc='upper right')
        plt.xlabel('horizon')
        plt.show()
