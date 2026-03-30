import os, sys, time, random
sys.path.append('../')
sys.path.append('../train')

from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from train.delf import Delf_V1
from helper.feeder import Feeder
from helper import matcher
import numpy as np
from matplotlib.pyplot import imshow
from feature.match import feature_extract

feeder_config = {
    'GPU_ID': 0,
    'IOU_THRES': 0.98,
    'ATTN_THRES': 0.37,
    'TARGET_LAYER': 'layer3',
    'TOP_K': 1000,
    #'PCA_PARAMETERS_PATH':'./output/pca/delf_real/pca.h5',
    #'PCA_PARAMETERS_PATH':'../extract/output/pca/mix_debase_ae/pca.h5',
    'PCA_PARAMETERS_PATH':'../extract/output/pca/res18_mix_debase_2/pca.h5',
    'PCA_DIMS':40,
    'USE_PCA': True,
    'SCALE_LIST': [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0],
    
    #'LOAD_FROM': '../train/repo/delf_real_clean/keypoint/ckpt/fix.pth.tar',
    #'LOAD_FROM': '../train/repo/mix_debase_ae/keypoint/ckpt/fix.pth.tar',
    'LOAD_FROM': '../train/repo/res18_mix_debase_1/keypoint/ckpt/fix.pth.tar',
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


'''   
with open('test/list.txt') as input_file:
    lines = input_file.readlines()

    imgpath_set = []
    print(len(lines))
    for i in range(len(lines)):
        imgpath_set.append(lines[i].strip())
    random.shuffle(imgpath_set)
    #print(imgpath_set)
    
    imgpath_pair = []
    judge = []
    pairs_num = int(len(lines)/2)
    for i in range(pairs_num):
        pair = []
        pair.append(imgpath_set[i])
        pair.append(imgpath_set[i + pairs_num])
        fingerid_1 = imgpath_set[i].strip().split('/')[-2]
        fingerid_2 = imgpath_set[i + pairs_num].strip().split('/')[-2]
        if(fingerid_1 == fingerid_2):
            judge.append(1)
        else:
            judge.append(0)
        imgpath_pair.append(pair)

for i in range(len(imgpath_pair)):
    os.makedirs('match' + '/{}'.format(str(i)))
    dirs = os.path.join('match/' + '{}/'.format(str(i)))
    print(dirs)
    result_image_byte, att1, att2 = get_result(myfeeder, imgpath_pair[i], dirs)
    result_image = Image.open(BytesIO(result_image_byte))
    if(judge[i] == 1):
        result_image.save(dirs + '1.png')
    else:
        result_image.save(dirs + '0.png')
    for idx in range(2):
        image = Image.open(imgpath_pair[i][idx])
        image = resize_image(image, 224)
        image.save(dirs + '{}.bmp'.format(str(idx)))
'''

with open('frr.txt') as input_file:
    lines = input_file.readlines()
    
    imgpath_pair = []
    for i in range(len(lines)):
        img1 = lines[i].strip().split(',')[0]
        img2 = lines[i].strip().split(',')[1]
        pair = []
        pair.append(img1)
        pair.append(img2)
        imgpath_pair.append(pair)
        
        dirs = './'
        result_image_byte, att1, att2, local_1, local_2 = get_result(myfeeder, pair, dirs)
        result_image = Image.open(BytesIO(result_image_byte))
        ###print our feature
        list1 = []
        list2 = []
        list3 = []
        list1, list2, list3 = feature_extract(local_1, local_2, img1, img2)
        print(list1)
        
        
