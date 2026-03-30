# -*- coding: utf-8 -*-
import os
import glob
import argparse
import shutil

def get_parser():
    parser = argparse.ArgumentParser(description="Get the Path")
    parser.add_argument('--inpath', default='error', type=str,help='input file path')
    #parser.add_argument('--outpath', default='error', type=str, help='the output flie to put')
    return parser

parser = get_parser()
args = parser.parse_args()
query = '/query'
gallery = '/gallery'
query_path = args.inpath + query
gallery_path = args.inpath + gallery
person_name = glob.glob(query_path + '/*')
i = 1
for old_name_query in person_name:
    tem = old_name_query.split('/')[-1]
    new_query_name = query_path + '/' + str(i)
    new_gallery_name = gallery_path + '/' + str(i)
    old_name_gallery = gallery_path + '/' + tem
    os.rename(old_name_query , new_query_name)
    os.rename(old_name_gallery , new_gallery_name)
    i = i + 1
    

