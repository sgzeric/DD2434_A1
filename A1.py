import pandas as pd
import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import floyd_warshall
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap

data = pd.read_csv('zoo.data', sep=',', names = ['animal name', 'hair', 'feathers', 'eggs', 'milk','airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type'])
data = pd.concat([data,pd.get_dummies(data['legs'], prefix='legs')], axis=1)
data_type = data['type']
animal = data['animal name']
data = data.drop(columns=['legs'])
data = data.drop(columns=['type'])
data_arr = data[data.columns[1:data.columns.size - 1]].to_numpy()

# Isomap implementation
def IsoMap(K, array):
    a=np.zeros((array.shape[0],array.shape[0]))
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            a[i,j] = np.sqrt(np.sum(np.power((array[i]-array[j]),2)))
    dist_mat = np.ones([a.shape[0], a.shape[0]])*1000
    for i in range(a.shape[0]):
        temp = np.argpartition(a[i], K)[:K + 1]
        dist_mat[i][temp] = a[i][temp]
    #dist_mat = []
    for k in range(dist_mat.shape[0]):
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[0]):
                dist_mat[i][j] = min(dist_mat[i][j], dist_mat[i][k]+dist_mat[k][j])
    '''
    dist_mat = np.zeros((array.shape[0], array.shape[0]))
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            dist_mat[i,j]= np.sqrt(np.sum((array[i]-array[j])**2))    
    index = dist_mat.argsort()
    neighbours = index[:, :K+1]
    inf_mat = np.ones((array.shape[0], array.shape[0]), dtype='float')*np.inf
    for i in range(array.shape[0]):
        inf_mat[i, neighbours[i, :]] = dist_mat[i, neighbours[i, :]]
    '''
    #dist_mat = floyd_warshall(dist_mat)**2

    id_mat = np.mat(np.ones(dist_mat.shape[0]))
    temp1 = dist_mat*id_mat.T*id_mat/dist_mat.shape[0]
    temp2 = np.array(id_mat.T*id_mat*dist_mat/dist_mat.shape[0])
    temp = id_mat.T*id_mat*dist_mat*id_mat.T*id_mat/np.power(dist_mat.shape[0], 2)
    B = -1/2*(dist_mat-temp1-temp2+temp)
    e_value, e_vector = np.linalg.eigh(B)
    index = np.argsort(-e_value)
    e_vector = e_vector[:,index]
    e_value = e_value[index]
    #X = np.eye(N=2, M=data_arr.shape[0])@np.sqrt(np.diag(np.abs(e_value)))@np.transpose(e_vector)
    
    e_vector_k = e_vector[:, [0,1]]
    e_value_k = np.diag(e_value[[0,1]])
    X = np.dot(e_vector_k, np.sqrt(e_value_k))
    
    return X

# MDS implementation
def Mds(array):
    dist_mat = np.zeros((array.shape[0], array.shape[0]))
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            dist_mat[i,j]= np.sqrt(np.sum((array[i]-array[j])**2))
    id_mat = np.mat(np.ones(dist_mat.shape[0]))
    temp1 = dist_mat*id_mat.T*id_mat/dist_mat.shape[0]
    temp2 = np.array(id_mat.T*id_mat*dist_mat/dist_mat.shape[0])
    temp = id_mat.T*id_mat*dist_mat*id_mat.T*id_mat/np.power(dist_mat.shape[0], 2)
    B = -1/2*(dist_mat-temp1-temp2+temp)

    e_value, e_vector = np.linalg.eig(B)
    index = np.argsort(-e_value)
    e_value = e_value[index]
    e_vector = e_vector[:,index]
    X = np.eye(N=2, M=data_arr.shape[0])@np.sqrt(np.diag(np.abs(e_value)))@np.transpose(e_vector)
    '''
    e_vector_k = e_vector[:, [0,1]]
    e_value_k = np.diag(e_value[[0,1]])
    X = np.dot(e_vector_k, np.sqrt(e_value_k))
    reconstruct_error = 1 - np.sum(e_value_k)/np.sum(np.diag(e_value[:]))
    '''
    return X

def plot_data(array, data, name):
    plt.figure()
    for i in range(1, 8):
        plt.scatter(array[data == i, 0].tolist(), array[data == i, 1].tolist(), alpha=.8, lw=0.1, label=i)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(name + ' of zoo dataset')
    plt.show()
'''
def plot_data(array, type, name):
    plt.figure()
    p = []
    temp = []
    colors = ['navy', 'turquoise', 'darkorange', 'crimson', 'lime', 'yellow', 'fuchsia']
    for i in range(1,20):
        scatter = plt.scatter(array[i,0], array[i,1], color=colors[type[i]], alpha=0.8, lw=0.1)
        if colors[type[i]] not in temp:
            temp.append(colors[type[i]])
            p.append(scatter)
    plt.legend(p,('1','2','3','4','5','6','7'), scatterpoints=1, title='Type')
    plt.title(name +' of Animals')
'''
if __name__=='__main__':
    K = 100
    pca = PCA(n_components=2)
    pca.fit(data_arr)
    pca_arr = pca.transform(data_arr)
    '''
    mds = MDS(n_components=2)
    sk_mds_arr = mds.fit_transform(data_arr)
    isomap = Isomap(n_neighbors=30, n_components=2)
    sk_isomap_arr = isomap.fit_transform(data_arr)
    '''
    #factor = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    mds_arr = Mds(data_arr).transpose()
    isomap_arr = IsoMap(K, data_arr)
    
    #plot_data(pca_arr, data_type, 'PCA')
    #plot_data(mds_arr, data_type, 'MDS')
    #plot_data(sk_mds_arr, data_type, 'MDS_sklearn')
    plot_data(isomap_arr, data_type, 'Isomap K='+ str(K))
    
    #plot_data(sk_isomap_arr, data_type, 'Isomap')
