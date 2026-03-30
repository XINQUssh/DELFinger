# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test, ResNet18
from efficientnet_pytorch import EfficientNet
from ghostnet import ghost_net
from ecn import ECN

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--efficientnet', default='b2', type=str, help="efficientnet-b2")

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 32

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        #transforms.Resize((256,128), interpolation=3),
        #transforms.Grayscale(1),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    print('Use the modle %s'%(save_path))
    network.load_state_dict(torch.load(save_path))
    return network
   
######################################################################
# PCA
#--------------------------- 
def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    #H = H.cuda()
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    #print(len(dataloaders))
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        ff = torch.FloatTensor(n,960).zero_().cuda()#tochange
    
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                feature1,feature2 = model(input_img)
                ff +=feature2
        # norm feature
        
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
        
    return features

def get_id(img_path):   
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-2]
        #print(filename)
        label = filename
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
    return labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_label = get_id(gallery_path)
query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    #model_structure = ResNet18(opt.nclasses)
    model_structure = ghost_net()

model = load_network(model_structure)
'''
# Remove the final fc layer and classifier layer
if opt.PCB:
    #if opt.fp16:
    #    model = PCB_test(model[1])
    #else:
        model = PCB_test(model)
else:
    #if opt.fp16:
        #model[1].model.fc = nn.Sequential()
        #model[1].classifier = nn.Sequential()
    #else:
        model.classifier.classifier = nn.Sequential()
'''
# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])

    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'query_f':query_feature.numpy(),'query_label':query_label}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.name)
result = './model/%s/result.txt'%opt.name
os.system('python evaluate_rerank_new_ecn.py | tee -a %s'%result)

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label}
    scipy.io.savemat('multi_query.mat',result)
