from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from report.utils import random_string

import os
from django.conf import settings


plt.switch_backend('Agg')


def kmeans_clustering(df, num_clusters):
    X = df.values
    X = X.T
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    return kmeans.labels_


def affinity_propagation_clustering(df, damping=0.5):
    vals = df.values
    vals = vals.T
    affinity_propagation = AffinityPropagation(damping=damping)
    affinity_propagation.fit(vals)
    return affinity_propagation.labels_


def tsne_reduce(df):
    X = df.values
    X = X.T
    return TSNE(n_components=2).fit_transform(X)


def plot_clusters(X,
                  labels,
                  names,
                  user_id,
                  show=False,
                  uri=settings.REPORT_PLOTS_URI):

    fig, ax = plt.subplots()
    x = X[:, 0]
    y = X[:, 1]
    ax.scatter(x, y, s=100, c=labels, marker='o')
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.3)

    for i, name in enumerate(names):
        t = ax.annotate(name, (x[i] + 5, y[i] + 5), bbox=bbox_props)
        bb = t.get_bbox_patch()
        bb.set_boxstyle("square", pad=0.6)

    if show:
        plt.show()
        return None
    else:
        path = os.path.join(uri, user_id)
        if not os.path.exists(path):
            os.makedirs(path)

        fig = random_string(10)
        fig_uri = "{}/{}.png".format(path, fig)
        plt.savefig(fig_uri)
        return path
