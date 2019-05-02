import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sb
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
train_dir = '/data/flowers/train'
valid_dir = '/data/flowers/valid'
test_dir = '/data/flowers/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(255),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
class Train():

    def train(self, epoch=1, path='checkpoint_test.pth'):

        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 2048)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(2048, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier
        # 将模型转入gpu
        model.cuda()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)

        def train(model, loader):
            steps = 0
            running_loss = 0
            print_every = 40
            model.train()
            for inputs, labels in loader:
                steps += 1
                optimizer.zero_grad()

                # 将数据移入GPU内存并训练
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 查看loss情况
                running_loss += loss.item()

                if steps % print_every == 0:
                    with torch.no_grad():
                        model.eval()
                        test_loss = 0
                        accuracy = 0

                        output = model.forward(inputs)
                        test_loss += criterion(output, labels).item()

                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                        # 打印
                        print("Epoch: {}/{}.. ".format(e, epoch - 1),
                              "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                              "Valid Loss: {:.3f}.. ".format(test_loss),
                              "Train Accuracy: {:.3f}".format(accuracy))

                        running_loss = 0
                        # 回到训练
                        model.train()

        def test(model, loader):
            # 打开验证模式并取消dropout
            model.eval()
            with torch.no_grad():
                for inputs, labels in loader:
                    test_loss = 0
                    accuracy = 0
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                    outputs = model.forward(inputs)
                    test_loss += criterion(outputs, labels).item()

                    ps = torch.exp(outputs)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()

                    print("Test Loss: {:.3f}.. ".format(test_loss),
                          "Test Accuracy: {:.3f}".format(accuracy))
                    
        for e in range(1, epoch):
            train(model, train_loader)
            test(model, valid_loader)
        test(model, test_loader)
        
        image_datasets = {'train': train_datasets,
                          'valid': valid_datasets,
                          'test': test_datasets}
        checkpoint = {'train': image_datasets['train'].class_to_idx,
                      'model': model}
        torch.save(checkpoint, path)
        print('Model has saved to: ' + path)

