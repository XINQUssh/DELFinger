import os
import pandas as pd
import torch

class LoadConfig(object):
    def __init__(self, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        self.numcls = 209
        
        self.use_dcl = False
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False
        
        self.weighted_sample = False
        self.cls_2 = False
        self.cls_2xmul = True





