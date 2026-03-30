import numpy as np
import cv2

def areamask(trans, im1, im2):
    h = im1.shape[0]
    w = im1.shape[1]
    dst = np.array([[0, w-1, w-1, 0],[0, 0, h-1, h-1],[1, 1, 1, 1]], dtype = "float32")
    res1 = np.dot(trans, dst)
    v1 = np.array([res1[0,:]/res1[2,:], res1[1,:]/res1[2,:]],dtype=np.int)
    v1 = v1.T
    print(v1)
    
    inv_trans = np.linalg.inv(trans)
    res2 = np.dot(inv_trans, dst)
    v2 = np.array([res2[0,:]/res2[2,:], res2[1,:]/res2[2,:]],dtype=np.int)
    v2 = v2.T
    print(v2)
    
    im3 = np.zeros(im1.shape[:2], dtype="uint8")
    print(im3.shape)
    polygon_mask3 = cv2.fillPoly(im3, [v1], 255) 
    area3 = np.sum(np.greater(polygon_mask3, 0))
    
    im4 = np.zeros(im2.shape[:2], dtype="uint8")
    polygon_mask4 = cv2.fillPoly(im4, [v2], 255) 
    area4 = np.sum(np.greater(polygon_mask4, 0))    
    
    return (area3 + area4)/(h*w*2)
    
if __name__ == '__main__':
    trans = [[2,0,0],[0,2,0],[0,0,1]]
    
    im1 = cv2.imread("A00EBIN.gray.png")
    im2 = cv2.imread("A00ABIN.gray.png")
    print(areamask(trans, im1, im2))
    