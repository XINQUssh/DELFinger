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
import glob
import shutil
import matplotlib.pyplot as plt
#from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test, ResNet18
#from efficientnet_pytorch import EfficientNet
from delf_global import Delf_V1
from re_ranking import re_ranking
from sklearn.decomposition import PCA
#from ghostnet import ghost_net
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
parser.add_argument('--test_dir',default='~/Data/xiaomi_test_mix_all/normal',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--efficientnet', default='b1', type=str, help="efficientnet-b1")

opt = parser.parse_args()

###load config###
# load the training config
opt.nclasses = 786

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir
#print(test_dir)

#print('-------read the order-----------')

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
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./repo/res18_mix_debase_1/finetune/ckpt/bestshot.pth.tar')
    network.load_state_dict(torch.load(save_path))
    return network


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
        ff = torch.FloatTensor(n,512).zero_().cuda()#tochange
    
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                feature1,feature2 = model(input_img)
                ff += feature2
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

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
#
def evaluate(score,ql,gl):
    score = (score-np.min(score))/(np.max(score)-np.min(score))
    sec_score = []
    detect_score_good = np.where(gl == ql)
    detect_score_bad = np.where(gl != ql)
    good_labels = gl[detect_score_good[0]]
    bad_labels = gl[detect_score_bad[0]]
    score_good = score[detect_score_good[0]]
    index_good = np.argsort(score_good)
    #print(index_good[0:5])
    #score_good = normalization(score_good)
    score_bad = score[detect_score_bad[0]]
    index_bad = np.argsort(score_bad)
    #score_bad = normalization(score_bad)
    only_label = np.unique(bad_labels)
    for i in only_label:
        tem_finger = np.where(gl == i)
        tem_score = score[tem_finger[0]]
        tem_index = np.argsort(tem_score)
        sec_score.append(tem_score[tem_index[1]])
    FRR_tag = np.zeros(1001)
    FAR_tag = np.zeros(1001)
    for k in range(0,1001):
        if score_good[index_good[1]] > k/1000:
            FRR_tag[k] = 1
        for j in range(len(sec_score)):
            if sec_score[j] <= k/1000:
                FAR_tag[k] = FAR_tag[k] + 1
    
    return FRR_tag,FAR_tag

######################################################################
# Load Collected data Trained model
print('-------test-----------')
#model_structure = ResNet18(opt.nclasses)
#model_structure = EfficientNet.from_pretrained('efficientnet-%s' % opt.efficientnet, num_classes = 1000)
#model_structure = EfficientNet.from_pretrained('efficientnet-%s' % opt.efficientnet, num_classes = opt.nclasses)
model_structure = Delf_V1(
    ncls = opt.nclasses,
    load_from = './bestshot.pth.tar',
    arch = 'resnet18',
    stage = 'finetune',
    target_layer = 'layer3')
    
#model = load_network(model_structure)
model = model_structure
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#

data_transforms = transforms.Compose([transforms.Resize(224),
        transforms.ToTensor(),
])
use_gpu = torch.cuda.is_available()
###############################get the envirement

envirement = test_dir.split('/')[-1]
print('-------data_load-----------')
image_datasets = {x: datasets.ImageFolder( os.path.join(test_dir,x) ,data_transforms) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                         shuffle=False, num_workers=16) for x in ['gallery','query']}
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_label = get_id(gallery_path)
query_label = get_id(query_path)
# Change to test mode

model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])

#features=np.append(gallery_feature,query_feature,axis=0)
#pca = PCA(n_components=0.97,whiten=False)
#fit = pca.fit(features)
#features = pca.fit_transform(features)
#features = torch.from_numpy(features)
#fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
#features = features.div(fnorm.expand_as(features))
#features = features[:,2:features.shape[1]-8]
#gallery_feature,query_feature = np.split(features,[gallery_feature.shape[0]])
#m,n=query_feature.shape
#print ('x:%d,y:%d'%(m,n))
#m,n=gallery_feature.shape
#p,q=query_feature.shape
#print ('x:%d,y:%d,x1:%d,y1:%d'%(m,n,p,q))
#print('test_old')
#print(query_feature.shape)
# Save to Matlab for check

result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'query_f':query_feature.numpy(),'query_label':query_label}
#result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'query_f':query_feature.numpy(),'query_label':query_label}
output_name = './xiaomi_new_model_test/pytorch_result_' + envirement + '.mat'
scipy.io.savemat(output_name,result)
## continue to change
#re-ranking
result = scipy.io.loadmat(output_name)
query_feature = result['query_f']
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_label = result['gallery_label'][0]
print('test')
print(query_feature.shape)
'''
print('calculate initial distance')
q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
q_q_dist = np.dot(query_feature, np.transpose(query_feature))
g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
since = time.time()
#re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
#print(re_rank.shape)
'''
distmat = ECN(query_feature, gallery_feature).transpose()


FRR = np.zeros(1001)
FAR = np.zeros(1001)
#FRR = 0
#FAR = 0
print(len(query_label))
for i in range(len(query_label)):
    FRR_tem,FAR_tem=evaluate(distmat[i,:],query_label[i],gallery_label)
    FRR = FRR_tem + FRR
    FAR = FAR_tem + FAR

x = np.arange(0, 1.001, 0.001)
x1 = np.arange(0,1,0.1)
FRR = FRR / len(query_label)
FAR = FAR / (19*len(query_label))
total = np.zeros((2,1001))
total[0,:] = FRR
total[1,:] = FAR
total=np.around(total,5)
np.savetxt(envirement + "_FAR_FRR.txt", total, fmt = '%.6e')
plt.plot(x, FRR, 'b-', label='FRR')
plt.plot(x, FAR, 'r-', label='FAR')
plt.xticks(x1)
plt.xlabel('score')
plt.legend()
plt.title(envirement + '_pca_norm_Rank2'+ '_FAR_FRR')
plt.savefig('./'+ envirement + '_pca_norm_Rank2_FAR_FRR.jpg')
#plt.show()
