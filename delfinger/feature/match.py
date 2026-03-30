
# author : zhangru                    time : 2020-08-04
# description : calculate the score of two match images

import os
import cv2
import random
import numpy as np
import scipy as sp
import scipy.linalg
#from tools import *
#from area_Overlap import *
#from areamask import *

import feature.tools as tools
import feature.area_Overlap as area_Overlap
import feature.areamask as areamask
	
# calculate the score of source image and match image
# in terms of the number of match points
def socreFeatureMatchNumber(featureSrc, featureMatch):

    # the threshold may be not reliable, change later
	minThre = 10
	maxThre = 500
	number = len(featureSrc)
    
    # make sure the value is valid
	if len(featureSrc) != len(featureMatch):
		raise ValueError("The feature number of srcImg and matchImg is wrong")
	if number < minThre:
		raise ValueError("Match number don't meet the request")
    
	score = (number-minThre) / (maxThre-minThre) * 100
	return score


	
# calculate the score of source image and match image 
# in terms of the area of match points
def scoreFeatureMatchArea(featureSrc, featureMatch):

    # the threshold may be not reliable, change later
	minThre = 2*9*9
	maxThre = 2*144*144   # imageSize = (144,144)
	listSrcX, listSrcY = tools.listFeature(featureSrc)
	listMatchX, listMatchY = tools.listFeature(featureMatch)
	
	# list the X and Y of featureSrc and featureMatch
	srcMinX,_ = min(enumerate(listSrcX))
	srcMinY,_ = min(enumerate(listSrcY))
	srcMaxX,_ = max(enumerate(listSrcX))
	srcMaxY,_ = max(enumerate(listSrcY))
	matchMinX,_ = min(enumerate(listMatchX))
	matchMinY,_ = min(enumerate(listMatchY))
	matchMaxX,_ = max(enumerate(listMatchX))
	matchMaxY,_ = max(enumerate(listMatchY))
	
	# calculate maximum circumscribed rectangle as area
	areaSrc = (srcMaxX-srcMinX) * (srcMaxY-srcMinY)
	areaMatch = (matchMaxX-matchMinX) * (matchMaxY-matchMinY)
	
	# make sure the value is valid
	if areaSrc<minThre/2 or areaMatch<minThre/2:
		raise ValueError("Match area don't meet the request")
		
	if areaSrc/areaMatch>4 or areaMatch/areaSrc>4:
		raise ValueError("Match area scale don't meet the request")
	
	score = (areaSrc+areaMatch-minThre) / (maxThre-minThre) * 100
	return score
	
	
	
# calculate the score of source image and match image 
# in terms of the density of match points
def scoreFeatureMatchDensity(featureSrc, featureMatch):
	
	# You can think it like socreFeatureMatchNumber and scoreFeatureMatchArea
	# Certainly, the code is also like their's
	
	# the threshold may be not reliable, change later
	minThre = 0.00001
	maxThre = 0.1
	number = len(featureSrc)
	listSrcX, listSrcY = tools.listFeature(featureSrc)
	listMatchX, listMatchY = tools.listFeature(featureMatch)
	
	# list the X and Y of featureSrc and featureMatch
	srcMinX,_ = min(enumerate(listSrcX))
	srcMinY,_ = min(enumerate(listSrcY))
	srcMaxX,_ = max(enumerate(listSrcX))
	srcMaxY,_ = max(enumerate(listSrcY))
	matchMinX,_ = min(enumerate(listMatchX))
	matchMinY,_ = min(enumerate(listMatchY))
	matchMaxX,_ = max(enumerate(listMatchX))
	matchMaxY,_ = max(enumerate(listMatchY))
	
	# calculate maximum circumscribed rectangle as area
	areaSrc = (srcMaxX-srcMinX) * (srcMaxY-srcMinY)
	areaMatch = (matchMaxX-matchMinX) * (matchMaxY-matchMinY)
	
	# calculate density by number and area of match feature
	density = number * 2 / (areaSrc+areaMatch)
	
	# make sure the value is valid
	if (density < minThre):
		raise ValueError("Match density don't meet the request") 
	
	# perhaps, the score should be higner in terms of change the 'maxThre'
	score = (density-minThre) / (maxThre-minThre) * 2 * 100
	return score


	
# calculate the transformMat of imageSrc and imageMatch
# in terms of the srcPoints and matchPoints
def transformMat(srcPoints, matchPoints):
	
	# get transformMat from 3 or 4 couple points 
	if (len(srcPoints) == 3):
		# cv2.warpAffine(img, Mat, (w,h))
		transformMat = cv2.getAffineTransform(srcPoints, matchPoints)
	if (len(srcPoints) == 4):
		# cv2.warpPerspective(img, Mat, (w,h))
		transformMat = cv2.getPerspectiveTransform(srcPoints, matchPoints)
	
	return transformMat
		
	

# calculate the finalTransformMat of imageSrc and imageMatch
# calculate per transformMat firstly and average per parameter
def finalTransforMatMeans(featureSrc, featureMatch, iter, coupleNum):

	# make sure the value is valid
	if coupleNum!=3 and coupleNum!=4:
		raise ValueError("CoupleNum don't meet the request")
	
	allData, meansFitMat = [], []
	
	# get couple points from featureSrc and featureMatch randomly
	for i in range(iter):
		srcPoints,matchPoints = tools.randomPoints(featureSrc, featureMatch, coupleNum)
		curTransformMat = transformMat(srcPoints, matchPoints)
		curTransformMat = list(np.array(curTransformMat).flatten())
		allData.append(curTransformMat)
	
	# when coupleNum is 3(or 4), the size is 2*3(or 3*3).   
	for i in range(3*(coupleNum-1)):
		tempData = [allData[j][i] for j in range(len(allData))]
		meansFitMat.append(tools.calMeans(tempData))
		
	# if coupleNum is 3, add [0,0,1]
	if len(meansFitMat) == 6:
		meansFitMat += [0,0,1]	
	
	# change shape to 3*3
	meansFitMat = np.array(meansFitMat).reshape(3, 3)
	
	return meansFitMat
	
	

# calculate the finalTransformMat of imageSrc and imageMatch
# fit the model by ransac and the ransacFitMat is 3*3
def finalTransforMatRansac(featureSrc, featureMatch):
	
	# add third dimension for featureSrc and featureMatch
	thirdDimen = [[1] for i in range(len(featureSrc))]
	srcThreeDimen = np.hstack((featureSrc, thirdDimen))
	matchThreeDimen = np.hstack((featureMatch, thirdDimen))
	allData = np.hstack((srcThreeDimen, matchThreeDimen))
	
	# fit the model by ransac
	inputCols = [i for i in range(3)]
	outputCols = [i+3 for i in range(3)]
	model = tools.LinearLeastSquareModel(inputCols, outputCols)
	ransacFitMat, ransacData = tools._ransac(allData, model, len(featureSrc)//2, 100, 250, len(featureSrc)//4)
	
	return ransacFitMat

	

# main function to calculate the feature
# text_path: the path of Pairing point text
#pic1_path , pic2_path: the path of the pair pic 
def match(text_path,pic1_path,pic2_path):
	
	# read data from txt
	featureSrc, featureMatch = tools.readTxt('1151')
	
	# calculate score of every judging criteria
	scoreNumber = socreFeatureMatchNumber(featureSrc, featureMatch)
	scoreArea = scoreFeatureMatchArea(featureSrc, featureMatch)
	scoreDensity = scoreFeatureMatchDensity(featureSrc, featureMatch)
	score = scoreNumber + scoreArea + scoreDensity
	
	# calculate the transforMatRansac by two way
	transforMatRansac = finalTransforMatRansac(featureSrc, featureMatch).T
	transforMatMeans = finalTransforMatMeans(featureSrc, featureMatch, 1000, 3)
	finalTransfromMat = (transforMatRansac + transforMatMeans) / 2


	#read the pic
	# pic_one = cv2.imread("/home/fingerprint/finger_ru/finger_ru/1151/0.bmp",0)
	# pic_two = cv2.imread("/home/fingerprint/finger_ru/finger_ru/1151/1.bmp",0)

	pic_one = cv2.imread(pic1_path,0)
	pic_two = cv2.imread(pic2_path,0)

	#calculate the overlap area
	area_overlap_Ransac = area_Overlap(transforMatRansac,pic_one,pic_two)
	area_overlap_Means = area_Overlap(transforMatMeans,pic_one,pic_two)
	area_overlap_finalTrans = area_Overlap(finalTransfromMat,pic_one,pic_two)

	#calulate the Coincident area after affine transformation
	area_mask_Ransac = areamask(transforMatRansac,pic_one,pic_two)
	area_mask_Means = areamask(transforMatMeans,pic_one,pic_two)
	area_mask_finalTrans = areamask(finalTransfromMat,pic_one,pic_two)

	list1 = [scoreNumber,  scoreArea,  scoreDensity,  area_overlap_Ransac,      area_mask_Ransac]
	list2 = [scoreNumber,  scoreArea,  scoreDensity,  area_overlap_Means,       area_mask_Means]
	list3 = [scoreNumber,  scoreArea,  scoreDensity,  area_overlap_finalTrans,  area_mask_finalTrans]

	return list1,list2,list3



	# print('\n',transforMatRansac)
	# print('\n',transforMatMeans)
	# print('\n',finalTransfromMat)
	# print('\n',scoreNumber, scoreArea, scoreDensity)
	# print("\n The total score of srcImg and matchImg is ", score)
	# print("I think you should input some data, bye!")
	# print(area_overlap_Ransac)
	# print(area_overlap_Means)
	# print(area_overlap_finalTrans)
	# print("the next is areamask")
	# print(area_mask_Ransac)
	# print(area_mask_Means)
	# print(area_mask_finalTrans)
 
def feature_extract(featureSrc, featureMatch, pic1_path, pic2_path):
	# calculate score of every judging criteria
	scoreNumber = socreFeatureMatchNumber(featureSrc, featureMatch)
	scoreArea = scoreFeatureMatchArea(featureSrc, featureMatch)
	scoreDensity = scoreFeatureMatchDensity(featureSrc, featureMatch)
	score = scoreNumber + scoreArea + scoreDensity
	
	# calculate the transforMatRansac by two way
	#transforMatRansac = finalTransforMatRansac(featureSrc, featureMatch).T
	transforMatMeans = finalTransforMatMeans(featureSrc, featureMatch, 1000, 3)
	#finalTransfromMat = (transforMatRansac + transforMatMeans) / 2


	#read the pic
	# pic_one = cv2.imread("/home/fingerprint/finger_ru/finger_ru/1151/0.bmp",0)
	# pic_two = cv2.imread("/home/fingerprint/finger_ru/finger_ru/1151/1.bmp",0)

	pic_one = cv2.imread(pic1_path,0)
	pic_two = cv2.imread(pic2_path,0)

	#calculate the overlap area
	#area_overlap_Ransac = area_Overlap(transforMatRansac,pic_one,pic_two)
	area_overlap_Means = area_Overlap(transforMatMeans,pic_one,pic_two)
	#area_overlap_finalTrans = area_Overlap(finalTransfromMat,pic_one,pic_two)

	#calulate the Coincident area after affine transformation
	#area_mask_Ransac = areamask(transforMatRansac,pic_one,pic_two)
	area_mask_Means = areamask(transforMatMeans,pic_one,pic_two)
	#area_mask_finalTrans = areamask(finalTransfromMat,pic_one,pic_two)

	#list1 = [scoreNumber,  scoreArea,  scoreDensity,  area_overlap_Ransac,      area_mask_Ransac]
	list2 = [scoreNumber,  scoreArea,  scoreDensity,  area_overlap_Means,       area_mask_Means]
	#list3 = [scoreNumber,  scoreArea,  scoreDensity,  area_overlap_finalTrans,  area_mask_finalTrans]

	return list2
    
	

	
	
	
	
	
