import umap
import numpy as np


def reduceDimensions(embeddings,min_dist=0.01,n_neighbors=40,n_components=20):
    reducedEmbeddings = umap.UMAP(metric="cosine",min_dist=min_dist,n_neighbors=n_neighbors,n_components=n_components)\
        .fit_transform(embeddings)
    np.save("reducedEmbeddings.py",reducedEmbeddings)
    return reducedEmbeddings

