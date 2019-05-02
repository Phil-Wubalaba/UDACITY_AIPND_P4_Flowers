import matplotlib.pyplot as plt
import numpy as np
import time
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
# TODO: 使用datasets定义dataloaders
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)

class Predict():
    def predict(self, model_path='checkpoint_test.pth', pic=test_dir+"/101/image_07949.jpg"):

        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

        def load_checkpoint(filepath):
            checkpoint = torch.load(filepath)
            model_ = checkpoint['model']
            image_dict = checkpoint['train']
            #将字典的键和值交换
            new_image_dict = {k: v for v, k in image_dict.items()}
            return model_, new_image_dict

        model, class_to_idx = load_checkpoint(model_path)


        def process_image(image):
            ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
                returns an Numpy array
            '''
            im = Image.open(image)
            w = im.size[0]
            h = im.size[1]
            # 判断宽和高哪个更大，并截取
            if w > h:
                w = round(w / h * 256)
                h = 256
            else:
                w = 256
                h = round(h / w * 256)
            im = im.resize((w, h))

            # 剧中截取
            left = w / 2 - 112
            up = h / 2 - 112
            right = w / 2 + 112
            low = h / 2 + 112
            im = im.crop((left, up, right, low))
            im = np.array(im)

            # 归一化，并平均标准化
            im = im / 225
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            im = (im - mean) / std

            im = im.transpose((2, 0, 1))

            im_tensor = torch.from_numpy(im)
            return im_tensor


        def imshow(image, ax=None, title=None):
            """Imshow for Tensor."""
            if ax is None:
                fig, ax = plt.subplots()

            # PyTorch tensors assume the color channel is the first dimension
            # but matplotlib assumes is the third dimension
            image = image.numpy().transpose((1, 2, 0))

            # Undo preprocessing
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean

            # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
            image = np.clip(image, 0, 1)
            # 这里进行一下修改，以便能够显示title
            if title is not None:
                plt.title(title[0], fontsize=30, color='y')

            ax.imshow(image)

            return ax

        def predict(path, model, topk=5):
            ''' Predict the class (or classes) of an image using a trained deep learning model.
            '''
            model.eval()
            img_tensor = process_image(path)
            img_tensor.unsqueeze_(0)
            img_tensor = img_tensor.type(torch.cuda.FloatTensor)
            #Variable已经无效，所以改成with torch.no_grad()
            with torch.no_grad():
                output = model(Variable(img_tensor.cuda()))
            ps = torch.exp(output)
            probs, index = ps.topk(topk)
            probs = probs.cpu().detach().numpy().tolist()[0]
            index = index.cpu().detach().numpy().tolist()[0]
            index = [class_to_idx[i] for i in index]
            return probs, index


        def show_result(path, model, topk=5):
            # 得出概率和花朵名字并显示第一幅图
            probs, classes = predict(path, model, topk)
            names = list(cat_to_name[i] for i in classes)

            plt.figure(figsize=(8, 14))
            imshow(image=process_image(path), ax=plt.subplot(2, 1, 1), title=names)

            # 赋值一个dataframe，并借助seaborn画第二张图
            data = {'probs': pd.Series(probs),
                    'names': pd.Series(names)}
            df = pd.DataFrame(data)

            plt.subplot(2, 1, 2)
            base_color = sb.color_palette()[0]
            sb.barplot(data=df, x='probs', y='names', color=base_color)

        show_result(pic, model, top=5)