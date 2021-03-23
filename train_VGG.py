import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import time
import argparse

from LISNN_VGG import LISNN

parser = argparse.ArgumentParser(description='train_VGG.py')

parser.add_argument('-gpu', type = int, default = 0)
parser.add_argument('-seed', type = int, default = 3154)
parser.add_argument('-epoch', type = int, default = 100)
parser.add_argument('-batch_size', type = int, default = 100)
parser.add_argument('-learning_rate', type = float, default = 1e-3)
parser.add_argument('-dts', type = str, default = 'CIFAR10')
parser.add_argument('-if_lateral', type = bool, default = True)
parser.add_argument('-loss', type = str, default = 'MSE')
parser.add_argument('-weight_decay', type = float, default = 0)
parser.add_argument('-time_window',type = int, default = 20)

opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True

test_scores = []
train_scores = []
save_path = './results/' + 'LISNN_VGG' + '_' + opt.dts + '_' + str(opt.seed)
file_name = "/result" + '_' + time.strftime("%Y%m%d%H%M%S", time.localtime()) +'.txt'
if not os.path.exists(save_path):
    os.mkdir(save_path)


if opt.dts == 'CIFAR10':
    train_dataset = dsets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    test_dataset = dsets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
elif opt.dts == 'MNIST':
    train_dataset = dsets.MNIST(root = './data/mnist/', train = True, transform = transforms.ToTensor(), download = True)
    test_dataset = dsets.MNIST(root = './data/mnist/', train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = opt.batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = opt.batch_size, shuffle = False)

model = LISNN(opt)
model.cuda()
if opt.loss == 'MSE':
    loss_function = nn.MSELoss()
elif opt.loss == 'CE':
    loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = opt.learning_rate, weight_decay = opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 80, gamma = 0.1)

def train(epoch):
    model.train()
    start_time = time.time()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = Variable(images.cuda())
        
        if opt.loss == 'MSE':
            one_hot = torch.zeros(opt.batch_size, model.fc[-1]).scatter(1, labels.unsqueeze(1), 1)
            labels = Variable(one_hot.cuda())
        if opt.loss == 'CE':
            labels=torch.tensor(labels, dtype=torch.long).cuda()

        outputs = model(images) 
        
        loss = loss_function(outputs, labels)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()

        if (i + 1) % (len(train_dataset) // (opt.batch_size * 5)) == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Time: %.2f' % (epoch + 1, opt.epoch, i + 1, len(train_dataset) // opt.batch_size, total_loss, time.time() - start_time))
            f = open(save_path + file_name ,'a')
            f.write('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Time: %.2f' % (epoch + 1, opt.epoch, i + 1, len(train_dataset) // opt.batch_size, total_loss, time.time() - start_time))
            f.write('\n')
            start_time = time.time()
            total_loss = 0
            
    scheduler.step()

def eval(epoch, if_test):
    model.eval()
    correct = 0
    total = 0
    if if_test:
        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            with torch.no_grad():
                outputs= model(images)
            pred = outputs.max(1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum()

        acc = 100.0 * correct.item() / total
        print('Test correct: %d Accuracy: %.2f%%' % (correct, acc))
        f = open(save_path + file_name ,'a')
        f.write('Test correct: %d Accuracy: %.2f%%' % (correct, acc))
        f.write('\n')
        test_scores.append(acc)
        if acc > max(test_scores):
            save_file = str(epoch) + '.pt'
            torch.save(model, os.path.join(save_path, save_file))
    else:
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            with torch.no_grad():
                outputs= model(images)
            pred = outputs.max(1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum()

        acc = 100.0 * correct.item() / total
        print('Train correct: %d Accuracy: %.2f%%' % (correct, acc))
        f = open(save_path + file_name ,'a')
        f.write('Train correct: %d Accuracy: %.2f%%' % (correct, acc))
        f.write('\n')
        train_scores.append(acc)

def main():
    torch.cuda.empty_cache()
    
    f = open(save_path + file_name ,'a')
    f.write('Loss_function: %s, Learning_rate: %f, Weight_decay: %f \n' % (opt.loss, opt.learning_rate, opt.weight_decay))
    f.write('Time_window: %d\n' % (opt.time_window))
    
    for epoch in range(opt.epoch):
        train(epoch)
        if (epoch + 1) % 2 == 0:
            eval(epoch, if_test = True)
        if (epoch + 1) % 20 == 0:
            eval(epoch, if_test = False)
        if (epoch + 1) % 20 == 0:
            print('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
            print('Best Train Accuracy in %d: %.2f%%' % (epoch + 1, max(train_scores)))
            f = open(save_path + file_name ,'a')
            f.write('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
            f.write('Best Train Accuracy in %d: %.2f%%' % (epoch + 1, max(train_scores)))
            f.write('\n')

if __name__ == '__main__':
    main()