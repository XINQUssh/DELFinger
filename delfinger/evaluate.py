import scipy.io
import torch
import numpy as np
#import time
import os
from sklearn import decomposition

#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl):
    
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    
    #index = index[0:2000]
    # good index
    good_index = np.argwhere(gl==ql)
    np.savetxt("good_index.txt", good_index, delimiter=',')
    
    #camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    #junk_index1 = np.argwhere(gl==-1)
    #junk_index2 = np.intersect1d(query_index, camera_index)
    #junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    #CMC_tmp = compute_mAP(index, good_index, junk_index)
    ap_tmp, CMC_tmp = compute_mAP(index, good_index)
    return ap_tmp, CMC_tmp

#def compute_mAP(index, good_index, junk_index):
def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    #mask = np.in1d(index, junk_index, invert=True)
    #index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    
    
    rows_good = np.argwhere(mask==True)
    
    np.savetxt("rows_good.txt", rows_good, delimiter=',')
    
    rows_good = rows_good.flatten()
    
    np.savetxt("rows_good2.txt", rows_good, delimiter=',')
    
    cmc[rows_good] = 1
    
    cmc2 = torch.IntTensor(len(index)).zero_()
    cmc2[rows_good[0]:] = 1;
    
    np.savetxt("cmc.txt", cmc, delimiter=',')
    
    for i in range(0,ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc2

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = result['query_f']

#pca = decomposition.PCA(n_components=)
#query_feature = pca.fit_transform(query_feature)

query_label = result['query_label'][0]
gallery_feature = result['gallery_f']

#pca = decomposition.PCA(n_components=0.99)
#gallery_feature = pca.fit_transform(gallery_feature)

gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = m_result['mquery_f']

    mquery_label = m_result['mquery_label'][0]
    
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    if i%100==0:
        print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@2:%f Rank@3:%f Rank@4:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[1],CMC[2],CMC[3],CMC[4],CMC[9],ap/len(query_label)))
np.savetxt("dets.txt", CMC, delimiter=',')

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
        mq = np.mean(mquery_feature[mquery_index,:], axis=0)
        ap_tmp, CMC_tmp = evaluate(mq,query_label[i],gallery_feature,gallery_label)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
