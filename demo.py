from img2cluster import clusterImagesKmeans,clusterImagesHDBSCAN,save_embeddings,reduceDimensions
import numpy as np
import glob

yourBasePath=""
paths=glob.glob(yourBasePath+"/*")

embeddings=save_embeddings(imagePaths=paths,batchSize=32,nDataLoaderWorkers=4)
reducedEmbeddings=reduceDimensions(embeddings=embeddings,min_dist= 0.01, n_neighbors= 40, n_components= 20)
labels=clusterImagesHDBSCAN(embeddings=reducedEmbeddings,min_cluster_size = 300)
##or you can use kmeans for shorter run time and heterogeneous data
# labels=clusterImages(embeddings=reducedEmbeddings,nClusters = 10)

##Once you run this snippet, it saves embeddings, reducedEmbeddings and labels arrays in cwd.
#After that you can play with hyperparams
#Scenario 1
embeddings=np.load("embeddings.npy")
reducedEmbeddings=reduceDimensions(embeddings=embeddings,min_dist= 0.01, n_neighbors= 40, n_components= 20)
labels=clusterImagesHDBSCAN(embeddings=reducedEmbeddings,nClusters = 10)
#Scenario 2
reducedEmbeddings=np.load("reducedEmbeddings.npy")
labels=clusterImagesHDBSCAN(embeddings=reducedEmbeddings,nClusters = 10)
#Scenario 3
labels=np.load("labels.npy")
for i in range(len(paths)):
    print(f"Label of {paths[i]}: {labels[i]}")

