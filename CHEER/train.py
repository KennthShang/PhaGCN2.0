import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from model import Wcnn
import argparse


"""
===============================================================
                        Input Params
===============================================================
"""

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=int, default = 2)
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--kmers', type=str, default='3,7,11,15')
parser.add_argument('--lr', type=str, default=0.001)
parser.add_argument('--epoch', type=int, default=7)
parser.add_argument('--embed', type=str, default='embed.pkl')
parser.add_argument('--weight', type=str, default='1,1,1,1,1')
args = parser.parse_args()

kmers = args.kmers
kmers = kmers.split(',')
kmers = list(map(int, kmers))

weight = args.weight
weight = weight.split(',')
weight = list(map(int, weight))



# Accuarcy function
def accuracy(pred, label):
    correct_pred = np.equal(np.argmax(pred, 1), label)
    acc = np.mean(correct_pred)
    return acc

"""
===============================================================
                        Data Part
===============================================================
"""

torch_embeds = nn.Embedding(65, 100)
torch_embeds.weight.requires_grad=False



train = np.genfromtxt('dataset/train.csv', delimiter=',')
train_label = train[:, -1]
train_feature = train[:, :-1]

train_feature = torch.from_numpy(train_feature).long()
train_label = torch.from_numpy(train_label).float()
train_dataset = Data.TensorDataset(train_feature, train_label)
training_loader = Data.DataLoader(
    dataset=train_dataset,    # torch TensorDataset format
    batch_size=200,           # mini batch size
    shuffle=True,               
    num_workers=0,              
)


val = np.genfromtxt('dataset/val.csv', delimiter=',')
val_label = val[:, -1]
val_feature = val[:, :-1]

val_feature = torch.from_numpy(val_feature).long()
val_label = torch.from_numpy(val_label).float()
val_dataset = Data.TensorDataset(val_feature, val_label)
validation_loader = Data.DataLoader(
    dataset=val_dataset,      # torch TensorDataset format
    batch_size=200,           # mini batch size
    shuffle=False,               
    num_workers=0,              
)


"""
===============================================================
                        Model Part
===============================================================
"""
# CrossEntropyLoss
torch.cuda.set_device(args.gpus)
net = Wcnn.WCNN(num_token=100,num_class=args.n, kernel_sizes=kmers, kernel_nums=[256, 256, 256, 256])
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
loss_func = torch.nn.CrossEntropyLoss(torch.tensor(weight).float().cuda())
net = net.cuda()

max_acc_val = -1
max_acc_train = -1
flag = 0
# Training
for epoch in range(args.epoch):
    net = net.train()
    acc = []
    for step, (batch_x, batch_y) in enumerate(training_loader): 
        batch_x = torch_embeds(batch_x)
        batch_x = batch_x.reshape(len(batch_x), 1, 1698, 100)
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # Predict
        prediction = net(batch_x)
        loss = loss_func(prediction, batch_y.long())
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    print("\n\nEpoch no.: " +str(epoch))
    net = net.eval()
    with torch.no_grad():
        acc = []
        for step, (batch_x, batch_y) in enumerate(training_loader): 
            batch_x = torch_embeds(batch_x)
            batch_x = batch_x.reshape(len(batch_x), 1,1698 , 100)
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            prediction = net(batch_x)
            tmp = accuracy(prediction.cpu().detach().numpy(), batch_y.cpu().numpy())
            acc.append(tmp)
    train_acc = np.mean(acc)
    print("Training loss: " + str(loss.cpu().detach().numpy())[:4] + "\nTraining acc: " +str(train_acc)[:4])
    if train_acc > max_acc_train:
        max_acc_train = train_acc
        flag = 1
    # Validation
    with torch.no_grad():
        acc = []
        for step, (batch_x, batch_y) in enumerate(validation_loader):
            batch_x = torch_embeds(batch_x)
            batch_x = batch_x.reshape(len(batch_x), 1, 1698, 100)
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            prediction = net(batch_x)
            tmp = accuracy(prediction.cpu().detach().numpy(), batch_y.cpu().numpy())
            acc.append(tmp)
    acc = np.mean(acc)
    print("Validation acc: " +str(acc)[:4])
    if acc > max_acc_val:
        max_acc_val = acc
        torch.save(net.state_dict(), 'Params.pkl')
        print("params stored!")
    elif acc == max_acc_val:
        if flag == 1:
            torch.save(net.state_dict(), 'Params.pkl')
            print("params stored!")
            flag = 0
torch.save(torch_embeds.state_dict(), 'Embed.pkl')
