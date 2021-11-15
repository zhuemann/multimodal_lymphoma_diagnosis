import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

import umap
import seaborn as sns

def make_u_map(data, targets):
#def make_u_map():
    print("inside u map")
    print(data)
    print(targets)
    #digits = load_digits()
    #print(type(digits))
    #print(type(digits.data))
    #print(digits.data)

    #reducer.fit(digits.data)

    reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
         force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
         local_connectivity=1.0, low_memory=False, metric='euclidean',
         metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
         n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
         output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
         set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
         target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
         transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)


    reducer = umap.UMAP(random_state=42,min_dist=1, spread = 10, a=1, b=1)
    reducer.fit(data)
    #embedding = reducer.transform(digits.data)
    embedding = reducer.transform(data)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert (np.all(embedding == reducer.embedding_))
    print(embedding.shape)
    print(embedding)
    plt.figure()
    #plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
    #plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='Spectral', s=5)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=targets, cmap='Spectral', s=5)

    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(1))
    plt.title('UMAP projection of the Digits dataset', fontsize=24);
    plt.show()