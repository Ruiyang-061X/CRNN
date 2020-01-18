import argparse
import os
import torch
from dataset import LMDBDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from crnn import CRNN
from torch.optim.adadelta import Adadelta
from torch.nn import CTCLoss
from utils import Converter
from torch.backends import cudnn


parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', required=True, help='path to trainset')
parser.add_argument('--validationset_path', required=True, help='path to validationset')
parser.add_argument('--alphabet', default='0123456789abcdefghijklmnopqrstuvwxyz*')
parser.add_argument('--image_h', default=32, help='the height of the imput image to the network')
parser.add_argument('--image_w', default=100, help='the width of the input image to the network')
parser.add_argument('--nh', default=256, help='size of the lstm hidden state')
parser.add_argument('--batch_size', default=4, help='batch size')
parser.add_argument('--nepoch', default=100, help='number of epoch')
option = parser.parse_args()

if not os.path.exists('model'):
    os.mkdir('model')

cudnn.benchmark = True

trainset = LMDBDataset(option.trainset_path, transform=transforms.Compose([transforms.Resize((option.image_h, option.image_w)), transforms.ToTensor()]))
trainset_dataloader = DataLoader(trainset, batch_size=option.batch_size, shuffle=True)
validationset = LMDBDataset(option.validationset_path, transform=transforms.Compose([transforms.Resize((option.image_h, option.image_w)), transforms.ToTensor()]))
validationset_dataloader = DataLoader(validationset, batch_size=option.batch_size, shuffle=True)

nc = 1
nclass = len(option.alphabet) + 1
crnn = CRNN(nc, nclass, option.nh)
crnn = crnn.cuda()
def weight_init(module):
    class_name = module.__class__.__name__
    if class_name.find('Conv') != -1:
        module.weight.data.normal_(0, 0.02)
    if class_name.find('BatchNorm') != -1:
        module.weight.data.normal_(1, 0.02)
        module.bias.data.fill_(0)
crnn.apply(weight_init)

loss_function = CTCLoss(zero_infinity=True)
loss_function = loss_function.cuda()
optimizer = Adadelta(crnn.parameters())
converter = Converter(option.alphabet)
print_every = 100
total_loss = 0.0

def validation():
    print('start validation...')
    crnn.eval()
    total_loss = 0.0
    n_correct = 0
    for i, (input, label) in enumerate(validationset_dataloader):
        if i == len(validationset_dataloader) - 1:
            continue
        if i == 9:
            break
        label_tmp = label
        label, length = converter.encode(label)
        input = input.cuda()
        predicted_label = crnn(input)
        predicted_length = [predicted_label.size(0)] * option.batch_size
        label = torch.tensor(label, dtype=torch.long)
        label = label.cuda()
        predicted_length = torch.tensor(predicted_length, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        loss = loss_function(predicted_label, label, predicted_length, length)

        total_loss += loss
        _, predicted_label = predicted_label.max(2)
        predicted_label = predicted_label.transpose(1, 0).contiguous().view(-1)
        predicted_label = converter.decode(predicted_label, predicted_length, raw=False)
        for j, k in zip(predicted_label, label_tmp):
            if j == k.lower():
                n_correct += 1

        if i == 0:
            for j, k in zip(predicted_label, label_tmp):
                print(k.lower())
                print(j)

    accuarcy = n_correct / float(10 * option.batch_size)
    print('loss: %.4f accuracy: %.4f' % (total_loss / 10, accuarcy))
    crnn.train()

    return accuarcy

for i in range(option.nepoch):
    for j, (input, label) in enumerate(trainset_dataloader):
        if j == len(trainset_dataloader) - 1:
            continue
        crnn.zero_grad()
        label, length = converter.encode(label)
        input = input.cuda()
        predicted_label = crnn(input)
        predicted_length = [predicted_label.size(0)] * option.batch_size
        label = torch.tensor(label, dtype=torch.long)
        label = label.cuda()
        predicted_length = torch.tensor(predicted_length, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        loss = loss_function(predicted_label, label, predicted_length, length)
        loss.backward()
        optimizer.step()

        total_loss += loss
        if j % print_every == 0:
            print('[%d / %d] [%d / %d] loss: %.4f' % (i, option.nepoch, j, len(trainset_dataloader), total_loss / print_every))
            total_loss = 0

    accuracy = validation()
    print('save model...')
    torch.save(crnn.state_dict(), 'model/crnn_%d_%.4f.pth' % (i, accuracy))