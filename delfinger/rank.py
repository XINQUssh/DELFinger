from re_ranking import re_ranking
import numpy as np
import scipy.io

def rankflit(query_feature, gallery_feature, ql, gl, thresh = 0.5):
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    FRR = []
    FAR = []
    c1 = 0
    c2 = 0
    for i in range(len(ql)):     
        score = re_rank[i,:]
        score = (score-np.min(score))/(np.max(score)-np.min(score))
        sec_score = []
        detect_score_good = np.where(gl == ql[i])
        #print(detect_score_good)
        detect_score_bad = np.where(gl != ql[i])
        good_labels = gl[detect_score_good[0]]
        bad_labels = gl[detect_score_bad[0]]
        score_good = score[detect_score_good[0]]
        index_good = np.argsort(score_good)
        #print(index_good[0:5])
        #score_good = normalization(score_good)
        score_bad = score[detect_score_bad[0]]
        index_bad = np.argsort(score_bad)
        #score_bad = normalization(score_bad)
        tmp = []
        for k in range(len(index_good)):
            #print(score_good[index_good[k]])
            if(score_good[index_good[k]] > thresh):
                tmp.append(index_good[k])
                c1 += 1
        FRR.append(tmp)
        
        tmp1 = []
        for k in range(len(index_bad)):
            if(score_bad[index_bad[k]] <= thresh):
                tmp1.append(index_bad[k])
                c2 += 1
        FAR.append(tmp1)        
    #FAR = np.array(FAR)
    #FRR = np.array(FRR)
    #print(c1)
    #print(c2)
    return FAR, FRR
    
if __name__ == '__main__':    
    result = scipy.io.loadmat("./xiaomi_new_model_test/pytorch_result_normal.mat")
    query_feature = result['query_f']
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_label = result['gallery_label'][0]
    print('test')
    print(query_feature.shape)

    FAR, FRR = rankflit(query_feature, gallery_feature,query_label, gallery_label, 0.15)
    #print(FAR)
    #print(FRR)