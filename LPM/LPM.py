import numpy as np
from sklearn.neighbors import KDTree


def LPM(X,Y,K,tau,lamda):
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

    vec = X-Y
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
    #print(cos_sita)

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
    
if __name__ == '__main__':
    rng = np.random.RandomState(80)
    X = rng.random_sample((20, 2))  # 10 points in 2 dimensions
    Y = rng.random_sample((20, 2))
    K = 6
    tau = 0.2
    lamda = 0.8
    print(LPM(X,Y,K,tau,lamda))