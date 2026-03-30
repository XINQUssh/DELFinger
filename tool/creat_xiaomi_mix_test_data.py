# -*- coding: utf-8 -*-
import os
import glob
import argparse
import shutil
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Get the Path")
    parser.add_argument('--inpath', default='error', type=str,help='input file path')
    parser.add_argument('--outpath', default='error', type=str, help='the output flie to put')
    return parser

parser = get_parser()
args = parser.parse_args()
tem_environment = glob.glob(args.inpath + '/*')
all_dir = []
query = '/Samples'
gallery = '/Template'
environment = []
for i in tem_environment: # 取出各个环境下的图片
    k = i.split('/')[-1]
    #print(k)
    target_dir = args.outpath + '/' + k
    if not os.path.exists(target_dir) :
        os.makedirs(target_dir)
    tem_query = i + query
    tem_gallery = i + gallery
    query_dir = glob.glob(tem_query + '/*')
    gallery_dir = glob.glob(tem_gallery + '/*')
    for i in range(0,15): #取出15个手指的作为测试集
        tem = query_dir[i]
        shutil.move(tem,target_dir + '/query/' + str(i + 1))
    for i in range(0,15):
        tem = gallery_dir[i]
        shutil.move(tem,target_dir + '/gallery/' + str(i + 1))