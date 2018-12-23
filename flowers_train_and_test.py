from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import sys

batch_size = 5000
threshold_compare = -35.0

test_on_test_dataset = True
if test_on_test_dataset:
    test_dataset_string = "test"
else:
    test_dataset_string = "val"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flowers zero shot learning experiment')
parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

my_temperature = 50


if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

dataset = 'flowers_data'
use_both_proba_and_target = False

#loading training image data

proba = np.load('%s/train_new_proba_10.npy' % dataset).astype(float)
proba = torch.from_numpy(proba).float()

data = torch.from_numpy(np.load('%s/train_classes_data.npy' % dataset)).float()

train_batch_size = 735

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, proba),
    batch_size=train_batch_size, shuffle=True, **kwargs)


#loading test (or validation) image data

test_labels = np.load('%s/%s_classes_onehot.npy' % (dataset, test_dataset_string)).astype(float)
test_labels = torch.from_numpy(test_labels).float()

test_data = torch.from_numpy(np.load('%s/%s_classes_data.npy' % (dataset, test_dataset_string))).float()

test_batch_size = 100
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_labels),
    batch_size=test_batch_size, shuffle=True, **kwargs)

# loading attribute information

test_centers = torch.from_numpy(np.load('%s/%s_classes_centroids_l2normalized.npy' % (dataset, test_dataset_string))).float().cuda()
test_centers = Variable(test_centers)

train_centers = torch.from_numpy(np.load('%s/train_classes_centroids_l2normalized.npy' % dataset)).float().cuda()
train_centers = Variable(train_centers).cuda()

nb_test_categories = 50

#################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1024, 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.out   = nn.Linear(1024,512)

    def forward(self, x):
        x = F.tanh(self.hidden(x))
        x = F.tanh(self.hidden2(x))
        x = self.out(x)
        return x

model = Net()
model2 = Net()
if args.cuda:
    model.cuda()
    model2.cuda()


optimizer = optim.Adam(list(model.parameters()) + list(model2.parameters()), lr=0.00001,  weight_decay=0)



def softmax_class(x_query, x_proto, weights=None, temperature=100):
    n_example = x_query.size(0)
    n_query = n_example
    n_class = x_proto.size(0)
    d = x_query.size(1)
    assert d == x_proto.size(1)
    
    y = torch.pow(x_proto.unsqueeze(0).expand(n_query, n_class, d) - x_query.unsqueeze(1).expand(n_query, n_class, d), 2).sum(2).squeeze()
    y = -y / temperature
    
    a = y.size()
    if len(a) > 1:
        [ymax,ymax_indices] = torch.max(y,1)
    else:
        [ymax,ymax_indices] = torch.max(y,0)
    thres_indices = ((torch.le(ymax,threshold_compare)).data).cpu().numpy() #threshold_vector
    nb_zero_indices = np.sum(thres_indices)
    cpt_index = -1
    if nb_zero_indices:
        ymax_indices = (ymax_indices.data).cpu().numpy()
        for i in range(n_example):
            cpt_index += 1
            if thres_indices[i]:
                qqqqq = ymax_indices[i]
                for j in range(n_class):
                    if j == qqqqq:
                        y[cpt_index,j] = 0
                    else:
                        y[cpt_index,j] = -10
    
    y = torch.exp(y)
    if weights is not None:
        y = y * weights.unsqueeze(0).expand_as(y)
    y = y / y.sum(1, keepdim=True).expand_as(y)
    return y

def kldivergence(y_target, y_pred):
    return (y_target * torch.log(((y_target) / (y_pred)))).sum()

def zeroshot_train(epoch):
    model.train()
    model2.train()
    enum_train = enumerate(train_loader)
    for batch_idx, (data, target) in enum_train:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        z = model(data)
        pi = None
        mu = model2(train_centers)
        y_hat = softmax_class(z, mu, weights=pi,temperature=my_temperature)
        loss = kldivergence(target,y_hat)
        loss.backward()
        optimizer.step()


def my_test():
    model.eval()
    correct = 0
    accuracy_dict = {}
    category_dict = {}
    l2_normalize_data = False
    l2_normalize_centroids = False
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        z = model(data)
        if l2_normalize_data:
            qn = torch.norm(z, p=2, dim=1).detach()
            z = z.div(qn.expand_as(z))
        test_c = model2(test_centers)
        if l2_normalize_centroids:
            qn = torch.norm(test_c, p=2, dim=1).detach()
            test_c = test_c.div(qn.expand_as(test_c))
        y_hat = softmax_class(z, test_c)
        [a1, a2] = torch.max(y_hat,1)
        [b1, b2] = torch.max(target,1)

        a3 = a2.size()
        for uu in range(a3[0]):
            true_category = b2[uu].cpu().data.numpy().astype(int)[0]
            predicted_category = a2[uu].cpu().data.numpy().astype(int)[0]
            correct_prediction = (predicted_category == true_category)
            correct += (correct_prediction)
            if true_category in category_dict:
                category_dict[true_category] += 1
                accuracy_dict[true_category] += int(correct_prediction)
            else:
                category_dict[true_category] = 1
                accuracy_dict[true_category] = int(correct_prediction)
    mean_accuracy = 0
    cpt = 0
    for (k,v) in category_dict.items():
        cpt += 1
        mean_accuracy += float(accuracy_dict[k]) / v
    mean_accuracy = mean_accuracy / cpt
    #print("test mean accuracy: %f percent" % (100.0 * mean_accuracy))
    return (100.0 * mean_accuracy)



nb_epochs = 1000

score_list = []
epoch_list = []

for epoch in range(1, nb_epochs+1):
    zeroshot_train(epoch)    
    if not(epoch % 3000):
        my_temperature = my_temperature * 0.9
    if not(epoch % 100): 
        score = my_test() 
        print("test mean accuracy %f : current epoch %d" % (score,epoch))
        score_list.append(score)
        epoch_list.append(epoch)


scores_file = open('%s_flowers_scores.txt' % test_dataset_string,'w')
for iscore in range(len(score_list)):
    scores_file.write("\nepoch %f : %d" % (score_list[iscore],epoch_list[iscore]))
scores_file.close()
