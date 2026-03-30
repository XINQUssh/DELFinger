# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
from datetime import datetime
import os
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, ResNet18
from random_erasing import RandomErasing
import yaml
import math
from shutil import copyfile
import re


from efficientnet_pytorch import EfficientNet
from ghostnet import ghost_net

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--weight_cent', type=float, default=0.0005, help="weight for center loss")
parser.add_argument('--lr_cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--pretrainedmodel', default='', type=str, help="load pretrainedmodel file (./model/.../net_59.pth)")
parser.add_argument('--efficientnet', default='b1', type=str, help="efficientnet-b1")
opt = parser.parse_args()

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        #transforms.Resize((256,128), interpolation=3),
        #transforms.Pad(10),
        #transforms.Grayscale(1),
        #transforms.CenterCrop((144, 144)),
        transforms.Resize(size=(160,160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        #transforms.Grayscale(1),
        #transforms.CenterCrop((144, 144)),
        transforms.Resize(size=(160,160)),#,interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}

image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
'''
print(image_datasets)
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
    
train_path = image_datasets['train'].imgs
val_path = image_datasets['val'].imgs
train_label = get_id(train_path)
val_label = get_id(val_path)
print(train_label)
'''    
    
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train', 'val']}
'''
for data in dataloaders['train']:
    inputs, label_scene = data
    print(label_scene)
'''
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion_cel, optimizer, scheduler, pre_epochs, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    num_epochs = pre_epochs + num_epochs
    for epoch in range(pre_epochs, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), '( lr = %.4f' % optimizer.state_dict()['param_groups'][0]['lr'] , ')')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                #if fp16:
                #    inputs = inputs.half()
 
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, features = model(inputs)
                        #outputs = model(inputs)
                else:
                    outputs, features = model(inputs)
                    #outputs = model(inputs)

                
                _, preds = torch.max(outputs.data, 1)
                #loss_cent = criterion_cent(features, labels) * opt.weight_cent
                loss = criterion_cel(outputs, labels) #+ loss_cent
                #loss_scenes = criterion_cel(out_scenes, scenes) * 0.8 + loss_fingers * 0.2
                

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                        #loss_scenes.backward()
                    optimizer.step()
                    '''
                    # by doing so, weight_cent would not impact on the learning of centers
                    for param in criterion_cent.parameters():
                        param.grad.data *= (1. / opt.weight_cent)
                    optimizer_centloss.step()
                    '''
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name, 'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#


    #model = ft_net(len(class_names), opt.droprate, opt.stride)
    #model = ResNet18(len(class_names))
    #model = EfficientNet.from_pretrained('efficientnet-%s' % opt.efficientnet, num_classes = len(class_names))
model = ghost_net(num_classes = len(class_names))
opt.nclasses = len(class_names)
print("num_classes:", len(class_names))
    
if opt.PCB:
    model = PCB(len(class_names))

opt.nclasses = len(class_names)
print("num_classes:", len(class_names))

if opt.pretrainedmodel:
    model.load_state_dict(torch.load('%s' % opt.pretrainedmodel))
    pre_epochs = int(re.findall(r"\d*\d", opt.pretrainedmodel)[-1]) + 1
else:
    pre_epochs = 0

print(model)

criterion_cel = nn.CrossEntropyLoss()

if not opt.PCB:
    '''
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    '''
    optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum=0.9, weight_decay=1e-4)
else:
    ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params += (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.classifier3.parameters(), 'lr': opt.lr},
             {'params': model.classifier4.parameters(), 'lr': opt.lr},
             {'params': model.classifier5.parameters(), 'lr': opt.lr},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./ghostnet.py', dir_name+'/ghostnet.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
if fp16:
    #model = network_to_half(model)
    #optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
    model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")



model = train_model(model, criterion_cel, optimizer, exp_lr_scheduler, pre_epochs, num_epochs=90)

