This package is implemented for GPU systems. <br /> The algorithm clusters images by running HDBSCAN or Kmeans on UMAP projection of CLIP embeddings.
<br /><br /> For requirements:<br />
```java
Python Version: 3.11.5
Package                 Version
----------------------- ------------
datasets                2.16.1
numpy                   1.26.0
Pillow                  9.5.0
scikit-learn            1.3.2
torch                   2.1.1
torchvision             0.16.1
transformers            4.35.2
umap-learn              0.5.5
```
<br />
demo.py:<br />

```python 
from img2cluster import clusterImagesKmeans,clusterImagesHDBSCAN,save_embeddings,reduceDimensions
import numpy as np
import glob

yourBasePath=""
paths=glob.glob(yourBasePath+"/*")

embeddings=save_embeddings(imagePaths=paths,batchSize=32,nDataLoaderWorkers=4)
reducedEmbeddings=reduceDimensions(embeddings=embeddings,min_dist= 0.01, n_neighbors= 40, n_components= 20)
labels=clusterImagesHDBSCAN(embeddings=reducedEmbeddings,min_cluster_size = 300)
##or you can use kmeans for shorter run time and predifined num clusters
# labels=clusterImages(embeddings=reducedEmbeddings,nClusters = 10)

##Once you run this snippet, it saves embeddings, reducedEmbeddings and labels arrays in cwd.
#After that you can play with hyperparams
#Scenario 1
embeddings=np.load("embeddings.npy")
reducedEmbeddings=reduceDimensions(embeddings=embeddings,min_dist= 0.01, n_neighbors= 40, n_components= 20)
labels=clusterImagesHDBSCAN(embeddings=reducedEmbeddings,min_cluster_size = 300)
#Scenario 2
reducedEmbeddings=np.load("reducedEmbeddings.npy")
labels=clusterImagesHDBSCAN(embeddings=reducedEmbeddings,min_cluster_size = 300)
#Scenario 3
labels=np.load("labels.npy")
for i in range(len(paths)):
    print(f"Label of {paths[i]}: {labels[i]}")
```
<br />
Cosine similarity metric is used for UMAP.<br />
Used CLIP model: https://huggingface.co/openai/clip-vit-large-patch14-336<br />
To help understanding of how UMAP hyperparameters control projection: https://pair-code.github.io/understanding-umap/
