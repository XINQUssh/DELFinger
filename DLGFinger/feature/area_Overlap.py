import numpy as np
import cv2 as cv
import os

#trans is the Affine matrix
#binary_Graph1 is input pic one
#binary_Graph2 is input pic two
#modle 0  use the normal image else use the Binary image
#pic_kind :pic is debase or enhance.     0 : enchance
#the function is used to get the similarity of the pic

def area_Overlap(trans, binary_Graph1, binary_Graph2, modle = 0, pic_kind = 0):
#################################
    
    [m,n] = binary_Graph1.shape
    #print(m)
    match_Num = 0
    similar_Num = 0
    if(pic_kind == 0):
        binary_Graph1 =  cv.adaptiveThreshold(binary_Graph1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
        binary_Graph2 =  cv.adaptiveThreshold(binary_Graph2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
        #print(binary_Graph1)
        #print(binary_Graph2)
        for i in range(0,m):
            for j in range(0,n):
                x_trans = int(i * trans[0][0] + j * trans[0][1] + trans[0][2])
                y_trans = int(i * trans[1][0] + j * trans[1][1] + trans[1][2])
                if( 0 <= x_trans and x_trans < m and 0 < y_trans and y_trans < n):
                    if(binary_Graph1[i][j] >0):
                        match_Num += 1
                        #print(match_Num)
                        if(binary_Graph2[x_trans][y_trans] > 0):
                            similar_Num += 1

    else:
        if(modle == 0):
            for i in range(0,m - 1):
                for j in range(0,n - 1):
                    x_trans = int(i * trans[0][0] + j * trans[0][1] + trans[0][2])
                    y_trans = int(i * trans[1][0] + j * trans[1][1] + trans[1][2])
                    if( 0 <= x_trans and x_trans < m and 0 <= y_trans and y_trans < n):
                        match_Num += 1
                        if(abs(binary_Graph1[i][j] - binary_Graph2[x_trans][y_trans]) <= 10):
                            similar_Num += 1
        else :
            binary_Graph1 =  cv.adaptiveThreshold(binary_Graph1, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
            binary_Graph2 =  cv.adaptiveThreshold(binary_Graph2, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
            for i in range(0,m - 1):
                for j in range(0,n - 1):
                    x_trans = int(i * trans[0][0] + j * trans[0][1] + trans[0][2])
                    y_trans = int(i * trans[1][0] + j * trans[1][1] + trans[1][2])
                    if( 0 <= x_trans and x_trans < m and 0 < y_trans and y_trans < n):
                        if(binary_Graph1[i][j] >0):
                            match_Num += 1
                            if(binary_Graph2[x_trans][y_trans] > 0):
                                similar_Num += 1
    
    return similar_Num / match_Num

        
            
        

