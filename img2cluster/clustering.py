 
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
import numpy as np

def clusterImagesKmeans(embeddings,nClusters:int=10):

    model = KMeans(n_clusters=nClusters)
    labels = model.fit_predict(embeddings)
    
    np.save("labels.npy",labels)
    return labels

def clusterImagesHDBSCAN(embeddings,min_cluster_size:int=300):

    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(embeddings)
    
    np.save("labels.npy",labels)
    return labels
