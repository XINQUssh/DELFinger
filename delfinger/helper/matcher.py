'''matcher.py
Matches two images using their DELF features.
The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.
The DELF features can be extracted using the extract_features.py script.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, time

import numpy as np
from PIL import Image
import io
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
np.set_printoptions(threshold=np.inf)  

import cv2

#_DISTANCE_THRESHOLD = 3.4           # Adjust this value depending on your dataset. 
                                    # This value needs to be engineered for optimized result.
                                    
_DISTANCE_THRESHOLD = 6.6
IMAGE_SIZE = (16, 12)

def LPM(X, Y, K, tau, lamda):
    treeX = KDTree(X)
    treeY = KDTree(Y)
    dist, indX = treeX.query(X, k=K) 
    dist, indY = treeY.query(Y, k=K) 

    #print(indX)
    indX = indX[:,1:]
    #print(indX)
    indY = indY[:,1:]

    sindX = np.sort(indX)
    sindY = np.sort(indY)

    #print(sindX - sindY)
    temp = ((sindX - sindY)==0)

    c1 = K - temp.sum(axis=1)
    #print(c1)

    vec = X-Y + 0.00001
    #print(vec)

    d2 = np.square(vec).sum(axis=1)
    #print(d2)

    vx = vec[:,0]
    vy = vec[:,1]
    #print(vx)
    #print(vy)

    index = np.array([sindX, sindY])

    d2i = d2[index]
    vxi = vx[index]
    vyi = vy[index]

    #print(d2i)
    #print(vxi)
    #print(vyi)

    vxr = np.repeat(vx[:,np.newaxis],K-1,axis=1)
    vyr = np.repeat(vy[:,np.newaxis],K-1,axis=1)
    d2r = np.repeat(d2[:,np.newaxis],K-1,axis=1)
    #print(vxr)

    cos_sita = (vxi * vxr + vyi * vyr) / np.sqrt(d2i * d2r);
    #print(np.sqrt(d2i * d2r))

    ratio = np.minimum(d2i, d2r) / np.maximum(d2i, d2r);
    #print(ratio)
    
    c2i = (cos_sita * ratio) < tau
    c2i0 = c2i[0] * temp
    c2 = c2i0.sum(axis=1)

    C = 0
    C += (c1+c2)/K
    #print(C)
    P = (C <= lamda)
    
    Xi = X[np.where(P==True)]
    Yi = Y[np.where(P==True)]
    return Xi, Yi

def load_image_into_numpy_array(image):
    if image.mode == "P": # PNG palette mode
        image = image.convert('RGBA')
        # image.palette = None # PIL Bug Workaround

    (im_width, im_height) = image.size
    imgarray = np.asarray(image).reshape(
        (im_height, im_width, -1)).astype(np.uint8)

    return imgarray[:, :, :3] # truncate alpha channel if exists. 

def read_image(image_path):
    with open(image_path, 'rb') as image_fp:
        image = Image.open(image_fp)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
    return image_np

def get_inliers(locations_1, descriptors_1, locations_2, descriptors_2):  

    num_features_1 = locations_1.shape[0]
    num_features_2 = locations_2.shape[0]

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    distances, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    # Perform geometric verification using RANSAC.
    model_robust, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    return inliers, locations_1_to_use, locations_2_to_use


def get_attention_image_byte(att_score):
    print('attn_score shape: {}'.format(att_score.shape))
    attention_np = np.squeeze(att_score, (0, 1)).astype(np.uint8)

    im = Image.fromarray(np.dstack((attention_np, attention_np, attention_np)))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return buf.getvalue()

    
def get_ransac_image_byte(img_1, locations_1, descriptors_1, img_2, locations_2, descriptors_2, dirs = './', save_path=None, use_opencv_match_vis=True):
    """
    Args:
        img_1: image bytes. JPEG, PNG
        img_2: image bytes. JPEG, PNG
    Return:
        ransac result PNG image as byte
        score: number of matching inlier
    """

    # Convert image byte to 3 channel numpy array
    with Image.open(BytesIO(img_1)) as img:
        img_1 = load_image_into_numpy_array(img)
    with Image.open(BytesIO(img_2)) as img:
        img_2 = load_image_into_numpy_array(img)

    inliers, locations_1_to_use, locations_2_to_use = get_inliers(
        locations_1,
        descriptors_1,
        locations_2,
        descriptors_2)


    # Visualize correspondences, and save to file.
    #fig, ax = plt.subplots(figsize=IMAGE_SIZE)
    inlier_idxs = np.nonzero(inliers)[0]
    score = sum(inliers)
    if score is None:
        score = 0
    
    #print(len(inlier_idxs))
    
    #pick up indices of keypoint    
    local_1 = locations_1_to_use[inlier_idxs]
    local_2 = locations_2_to_use[inlier_idxs]
    
    #transfer to int
    #local_1 = [[round(j) for j in local_1[i]] for i in range(len(local_1))]
    #local_2 = [[round(j) for j in local_2[i]] for i in range(len(local_2))]

    #save keypoint location to txt            
    np.savetxt(dirs + 'locations_1_to_use.txt',local_1,delimiter=',')
    np.savetxt(dirs + 'locations_2_to_use.txt',local_2,delimiter=',')
    #np.savetxt('inlier_idxs.csv',inlier_idxs,delimiter=',')

    
    if use_opencv_match_vis:
        inlier_matches = []
        for idx in inlier_idxs:
            inlier_matches.append(cv2.DMatch(idx, idx, 0))
        
        kp1 =[]
        for point in locations_1_to_use:
            kp = cv2.KeyPoint(point[1], point[0], _size=1)
            kp1.append(kp)

        kp2 =[]
        for point in locations_2_to_use:
            kp = cv2.KeyPoint(point[1], point[0], _size=1)
            kp2.append(kp)

        ransac_img_o = cv2.drawMatches(img_1, kp1, img_2, kp2, inlier_matches, None, flags=0)
        ransac_img_o = cv2.cvtColor(ransac_img_o, cv2.COLOR_BGR2RGB)    
        image_byte_o = cv2.imencode('.png', ransac_img_o)[1].tostring()
    
    locations_1_to_use, locations_2_to_use = LPM(local_1, local_2, K = 10,  tau = 0.05, lamda = 0.9)
    print("local_1")
    print(len(local_1))
    print("local_1")
    print(len(locations_1_to_use))
        
    inlier_idxs = np.asarray(range(len(locations_1_to_use)))

    
    
    #print(inlier_idxs)
    #print(locations_1_to_use.shape)
    print(locations_2_to_use.shape)
    
    if use_opencv_match_vis:
        inlier_matches = []
        for idx in inlier_idxs:
            inlier_matches.append(cv2.DMatch(idx, idx, 0))
            #print(cv2.DMatch(idx, idx, 0).queryIdx)
        
        kp1 =[]
        for point in locations_1_to_use:
            kp = cv2.KeyPoint(point[1], point[0], _size=1)
            #print(kp.pt)
            kp1.append(kp)
            #pts = np.float([kp1[idx].pt for idx in len(kp1)]).reshape(-1, 1, 2)
            #print(pts)

        kp2 =[]
        for point in locations_2_to_use:
            kp = cv2.KeyPoint(point[1], point[0], _size=1)
            kp2.append(kp)
        '''
        for i, (m1, m2) in enumerate(inlier_matches):
            #matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx
            pt2 = kp2[m1.trainIdx].pt  # queryIdx
            # print(kpts1)
            print(i, pt1, pt2)
        '''       
        ransac_img = cv2.drawMatches(img_1, kp1, img_2, kp2, inlier_matches, None, flags=0)
        ransac_img = cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB)    
        image_byte = cv2.imencode('.png', ransac_img)[1].tostring()

    else:
        plot_matches(
            ax,
            img_1,
            img_2,
            locations_1_to_use,
            locations_2_to_use,
            np.column_stack((inlier_idxs, inlier_idxs)),
            matches_color='b')
        ax.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())      
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches=extent, format='png')
        plt.close('all') # close resources. 
        image_byte = buf.getvalue()
    
    return image_byte_o, image_byte, score, locations_1_to_use, locations_2_to_use
    



