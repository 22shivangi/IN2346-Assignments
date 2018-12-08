import os
import torch.utils.model_zoo as model_zoo

__al__ = ['VGG','vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# here (https://github.com/pytorch/vision/tree/master/torchvision/models) to find the download link of pretrained models

root = '/Users/shivangi/Desktop/Masters/Sem2/DL/P/i2dl/exercise_4/pretrained'
res101_path = os.path.join(root, 'ResNet', 'resnet101-5d3b4d8f.pth')
res152_path = os.path.join(root, 'ResNet', 'resnet152-b121ed2d.pth')
inception_v3_path = os.path.join(root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
vgg19_bn_path = os.path.join(root, 'VggNet', 'vgg19_bn-c79401a0.pth')
vgg16_path = os.path.join(root, 'VggNet', 'vgg16-397923af.pth')
dense201_path = os.path.join(root, 'DenseNet', 'densenet201-4c113574.pth')

'''
vgg16 trained using caffe
visit this (https://github.com/jcjohnson/pytorch-vgg) to download the converted vgg16
'''
vgg16_caffe_path = os.path.join(root, 'VggNet', 'vgg16-caffe.pth')



