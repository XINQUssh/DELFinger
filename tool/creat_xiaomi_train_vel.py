# -*- coding: utf-8 -*-
import os
import glob
import argparse
import shutil
import numpy as np
import random


def get_parser():
    parser = argparse.ArgumentParser(description="Get the Path")
    parser.add_argument('--inpath', default='error', type=str,help='input file path')
    #parser.add_argument('--old_file_path', default='error', type=str,help='input file path')
    #parser.add_argument('--outpath', default='error', type=str, help='the output flie to put')
    return parser

parser = get_parser()
args = parser.parse_args()
inpath = args.inpath + '/train'
tem_environment = glob.glob(inpath + '/*')
val_dir = args.inpath + '/val'

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

for i in tem_environment:
    tem_finger_pic = glob.glob(i + '/*')
    all_member = []
    for j in tem_finger_pic:
        all_member.append(j.split('/')[-1])
    print(len(all_member))
    select_finger = random.sample(all_member,15)
    target_dir = val_dir + '/' + i.split('/')[-1]
    os.makedirs(target_dir)
    for j in select_finger:
        tag_pic = i + '/' + j
        shutil.move(tag_pic,target_dir)
    print('finish  %s'%(i.split('/')[-1]))
    