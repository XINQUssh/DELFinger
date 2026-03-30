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
from rank import rankflit
import os, sys, time, random
sys.path.append('../')
sys.path.append('../train')

from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from train_res.delf import Delf_V1
from helper.feeder import Feeder
from helper import matcher
import numpy as np
from matplotlib.pyplot import imshow
from feature.match import feature_extract

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
parser.add_argument('--test_dir',default='~/Data/xiaomi_test_mix_all_ru/class_one/normal',type=str, help='./test_data')
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
    return np.array(labels)

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
    
##############################
##############################
#local feature
feeder_config = {
    'GPU_ID': 0,
    'IOU_THRES': 0.98,
    'ATTN_THRES': 0.37,
    'TARGET_LAYER': 'layer3',
    'TOP_K': 1000,
    #'PCA_PARAMETERS_PATH':'./output/pca/delf_real/pca.h5',
    #'PCA_PARAMETERS_PATH':'../extract/output/pca/mix_debase_ae/pca.h5',
    'PCA_PARAMETERS_PATH':'./extract/output/pca/res18_mix_debase_2/pca.h5',
    'PCA_DIMS':40,
    'USE_PCA': True,
    'SCALE_LIST': [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0],
    
    #'LOAD_FROM': '../train/repo/delf_real_clean/keypoint/ckpt/fix.pth.tar',
    #'LOAD_FROM': '../train/repo/mix_debase_ae/keypoint/ckpt/fix.pth.tar',
    'LOAD_FROM': './train_res/res18_mix_debase_1/keypoint/ckpt/fix.pth.tar',
    'ARCH': 'resnet18',
    'EXPR': 'dummy',
    'TARGET_LAYER': 'layer3',
}
myfeeder = Feeder(feeder_config)


def resize_image(image, target_size=800):
    def calc_by_ratio(a, b):
        return int(a * target_size / float(b))

    size = image.size
    if size[0] < size[1]:
        w = calc_by_ratio(size[0], size[1])
        h = target_size
    else:
        w = target_size
        h = calc_by_ratio(size[1], size[0])

    image = image.resize((w, h), Image.BILINEAR)
    return image


def get_and_cache_image(image_path, basewidth=224):
    image = Image.open(image_path)
    if basewidth is not None:
        image = resize_image(image, basewidth)
    imgByteArr = BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return image, imgByteArr


def get_result(feeder, query, dirs):
    pil_image = []
    byte_image = []
    for _, v in enumerate(query):
        pil, byte = get_and_cache_image(v)
        pil_image.append(pil)
        byte_image.append(byte)

    # feed and get output.
    outputs = feeder.feed_to_compare(query, pil_image)
    
    att1 = matcher.get_attention_image_byte(outputs[0]['attention_np_list'])
    att2 = matcher.get_attention_image_byte(outputs[1]['attention_np_list'])

    yuantu, side_by_side_comp_img_byte, score, local_1, local_2 = matcher.get_ransac_image_byte(
        byte_image[0],
        outputs[0]['location_np_list'],
        outputs[0]['descriptor_np_list'],
        byte_image[1],
        outputs[1]['location_np_list'],
        outputs[1]['descriptor_np_list'],
        dirs)
    print('matching inliner num:', score)
    return side_by_side_comp_img_byte, att1, att2, local_1, local_2

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
    
FAR = [] 
FRR = []
FAR,FRR = rankflit(query_feature, gallery_feature, query_label, gallery_label)
print(len(FAR))
print(len(FRR))

far = []
for i in range(len(FAR)):
    for j in range(len(FAR[i])):
        pair = []
        pair.append(query_path[i][0])
        pair.append(gallery_path[FAR[i][j]][0])
        far.append(pair)

frr = []
for i in range(len(FRR)):
    for j in range(len(FRR[i])):
        pair = []
        pair.append(query_path[i][0])
        pair.append(gallery_path[FRR[i][j]][0])
        frr.append(pair)
        
print('frr:',len(frr))
print('far:',len(far))

our_feature_1 = []

for i in range(len(far)):
     dirs = './'
     result_image_byte, att1, att2, local_1, local_2 = get_result(myfeeder, far[i], dirs)
     result_image = Image.open(BytesIO(result_image_byte))
     ###print our feature
     list1= feature_extract(local_1, local_2, far[i][0], far[i][1])
     list1.append(int(0))
     print(list1)
     our_feature_1.append(list1)

for i in range(len(frr)):
     dirs = './'
     result_image_byte, att1, att2, local_1, local_2 = get_result(myfeeder, frr[i], dirs)
     result_image = Image.open(BytesIO(result_image_byte))
     ###print our feature
     list1 = feature_extract(local_1, local_2, frr[i][0], frr[i][1])
     list1.append(int(1))
     print(list1)
     our_feature_1.append(list1)


our_feature_array_1 = np.array(our_feature_1)
np.savetxt('normal_feature_1.txt',our_feature_array_1)








