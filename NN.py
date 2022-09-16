import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DatasetFromListOfDicts(torch.utils.data.Dataset):
    LABEL_NUMBER_DICT={"walk":0,"bike":1,"idle":2,"car":3}
    def __init__(self, data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = self.data[index]["X"].T.astype(np.float32).values
            
        return torch.tensor([X]), self.LABEL_NUMBER_DICT[self.data[index]["y"]]

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv1d(6, 12, (1,3))
        ## original length of accuracy data = 100 - 4 for both convolution layers
        self.fc1 = nn.Linear(12*96, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = (self.fc4(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
def train(dataset, n_epochs = 10):
    net=Net()
    load_data=DatasetFromListOfDicts(dataset)
    dataloader= torch.utils.data.DataLoader(load_data, batch_size=10,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    LR=0.001
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f"epoch: {epoch + 1}/{n_epochs}")
        if epoch and epoch%5==0:
            LR=.1*LR
            optimizer=optim.Adam(net.parameters(),lr=LR)
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, torch.tensor(labels))
            loss.backward()
            optimizer.step()
    
    
    print('Finished Training')
    return net
    
def evaluate(net,dataset):
    y_pred=[0]*len(dataset)
    y_true=[0]*len(dataset)
    load_data=DatasetFromListOfDicts(dataset)
    dataloader= torch.utils.data.DataLoader(load_data)
    for i, data in enumerate(dataloader):
        inp, label = data
        outputs = net(inp)
        y_pred[i]=outputs[0].tolist().index(outputs[0].max())
        y_true[i]=int(label)
    return y_pred,y_true