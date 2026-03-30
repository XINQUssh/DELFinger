# -*- coding: utf-8 -*-
import os
import glob
import argparse
import shutil
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Get the Path")
    parser.add_argument('--inpath', default='error', type=str,help='input file path')
    #parser.add_argument('--old_file_path', default='error', type=str,help='input file path')
    #parser.add_argument('--outpath', default='error', type=str, help='the output flie to put')
    return parser

parser = get_parser()
args = parser.parse_args()
tem_environment = glob.glob(args.inpath + '/*')
all_dir = []
query = '/Samples'
gallery = '/Template'
environment = []
for i in tem_environment: #找到所有的debase文件并且计算总个数
    k = i.split('/')[-1]
    #print(k)
    environment.append(i.split('/')[-1])
    tem_query = i + query
    tem_gallery = i + gallery
    query_dir = glob.glob(tem_query + '/*')
    gallery_dir = glob.glob(tem_gallery + '/*')
    for j in query_dir:
        tem_last_dir = j.split('/')[-1]
        all_dir.append(tem_last_dir)
    for j in gallery_dir:
        tem_last_dir = j.split('/')[-1]
        all_dir.append(tem_last_dir)
all_dir = np.unique(all_dir)
tag = all_dir + 1
'''
tag = 1

print(len(all_dir))
if not os.path.exists(args.inpath + '/train'):
    os.makedirs(args.inpath + '/train')

for j in all_dir: 
    os.makedirs(args.inpath + '/train/' + str(tag))
    for i in tem_environment:
        tem_query = i + query
        tem_gallery = i + gallery
        tem_dir = tem_query + '/' + j
        #print(tem_dir)
        if os.path.exists(tem_dir):#找到该环境下存在的手指，移动到train文件夹中
            #print(tem_dir)
            for k in glob.glob(tem_dir + '/*'):
                shutil.copy(k,args.inpath + '/train/' + str(tag))
        tem_dir = tem_gallery + '/' + j
        if os.path.exists(tem_dir):
            #print(tem_dir)
            for k in glob.glob(tem_dir + '/*'):
                shutil.copy(k,args.inpath + '/train/' + str(tag))
    tag = tag + 1 

print('finish')
'''
old_file_path = '/home/fingerprint/Data/xiaomi_debase_train/train/*'
old_train_files = glob.glob(old_file_path)
for i in old_train_files:
    print(i)
    shutil.copytree(i,args.inpath + '/train/' + str(tag))
    tag = tag + 1







