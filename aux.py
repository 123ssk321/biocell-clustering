from collections import Counter

from sklearn.metrics import precision_score, recall_score, f1_score, rand_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, OPTICS
import matplotlib.pyplot as plt
from tp2_aux import *


def purity_score(y_true, y_pred):
    p = 0
    n = len(y_true)
    count = Counter(y_pred)
    for i in np.unique(y_pred):
        n_i = count[i]
        pi_js = []
        for j in [1, 2, 3]:
            n_i_j = (y_true[y_pred == i] == j).sum(axis=0)
            pi_j = n_i_j / n_i
            pi_js.append(pi_j)
        pi = max(pi_js)
        p += (n_i / n) * pi

    return p


def measures(y_true, y_pred):
    purity = purity_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1_metric = f1_score(y_true, y_pred, average='weighted')
    rand_index = rand_score(y_true, y_pred)

    return purity, precision, recall, f1_metric, rand_index


def plot_measure_nclusters(metrics, n_clusters):
    fig = plt.figure(figsize=(12, 8))

    for metric, n in zip(metrics.keys(), range(1, 6)):
        plt.subplot(230 + n)

        plt.plot(n_clusters, metrics[metric]['agglomerative'], '-b', label='Agglomerative')
        plt.plot(n_clusters, metrics[metric]['spectral'], '-r', label='Spectral')
        plt.plot(n_clusters, metrics[metric]['kmeans'], '-g', label='KMeans')

        plt.ylim(0, 1)
        plt.xlabel('k clusters')
        plt.xticks(n_clusters, n_clusters)
        plt.legend()
        plt.title(f'{metric}')

    plt.subplot(230 + 6)

    plt.plot(n_clusters, metrics['loss']['kmeans'], '-g', label='KMeans')

    plt.xlabel('k clusters')
    plt.xticks(n_clusters, n_clusters)
    plt.legend()
    plt.title(f'kmeans loss')

    fig.align_labels()
    fig.subplots_adjust(hspace=0.4)
    plt.savefig('plot_measure_nclusters.png')
    plt.show()


def plot_measure_distances(metrics, distances):
    fig = plt.figure(figsize=(12, 8))

    for metric, n in zip(metrics.keys(), range(1, 6)):
        plt.subplot(230 + n)
        plt.plot(distances, metrics[metric], '-b', label='Agglomerative')
        plt.legend()
        plt.title(f'{metric}')

    fig.align_labels()
    plt.savefig('plot_measure_distances.png')
    plt.show()


def plot_measure_minsamples(metrics, distances):
    fig = plt.figure(figsize=(12, 8))

    for metric, n in zip(metrics.keys(), range(1, 6)):
        plt.subplot(230 + n)
        plt.plot(distances, metrics[metric], '-b', label='OPTICS')
        plt.legend()
        plt.title(f'{metric}')

    fig.align_labels()
    plt.savefig('plot_measure_minsamples.png')
    plt.show()


def optics_clustering(feats, indices, labels, ids, min_samples):
    optics_labels = OPTICS(min_samples=min_samples).fit_predict(feats)
    report_clusters(ids, optics_labels, 'OPTICS_clustering.html')
    return measures(labels[indices], optics_labels[indices] + 1)


def agglomerative_clustering(n_clusters, feats, indices, labels, ids, distance):
    agglomerative_labels = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance).fit_predict(
        feats)
    report_clusters(ids, agglomerative_labels, 'agglomerative_clustering.html')
    return measures(labels[indices], agglomerative_labels[indices] + 1)


def spectral_clustering(n_clusters, feats, indices, labels, ids):
    spectral_labels = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr').fit_predict(feats)
    report_clusters(ids, spectral_labels, 'spectral_clustering.html')
    return measures(labels[indices], spectral_labels[indices] + 1)


def kmeans_clustering(n_clusters, feats, indices, labels, ids):
    k_means = KMeans(n_clusters=n_clusters).fit(feats)
    k_means_labels = k_means.predict(feats)
    report_clusters(ids, k_means_labels, 'kmeans_clustering.html')
    return measures(labels[indices], k_means_labels[indices] + 1), -k_means.score(feats)


def clustering(feats, imgs_labeled, labels, ids, n_clusters):
    cluster_metrics = {'purity': {
        'agglomerative': [],
        'spectral': [],
        'kmeans': []
    }, 'precision': {
        'agglomerative': [],
        'spectral': [],
        'kmeans': []
    }, 'recall': {
        'agglomerative': [],
        'spectral': [],
        'kmeans': []
    }, 'f1': {
        'agglomerative': [],
        'spectral': [],
        'kmeans': []
    }, 'rand index': {
        'agglomerative': [],
        'spectral': [],
        'kmeans': []
    }, 'loss': {
        'kmeans': []
    }
    }

    for n in n_clusters:
        purity, precision, recall, f1, rand_index = agglomerative_clustering(n, feats, imgs_labeled, labels, ids, None)
        cluster_metrics['purity']['agglomerative'].append(purity)
        cluster_metrics['precision']['agglomerative'].append(precision)
        cluster_metrics['recall']['agglomerative'].append(recall)
        cluster_metrics['f1']['agglomerative'].append(f1)
        cluster_metrics['rand index']['agglomerative'].append(rand_index)

        purity, precision, recall, f1, rand_index = spectral_clustering(n, feats, imgs_labeled, labels, ids)
        cluster_metrics['purity']['spectral'].append(purity)
        cluster_metrics['precision']['spectral'].append(precision)
        cluster_metrics['recall']['spectral'].append(recall)
        cluster_metrics['f1']['spectral'].append(f1)
        cluster_metrics['rand index']['spectral'].append(rand_index)

        (purity, precision, recall, f1, rand_index), loss = kmeans_clustering(n, feats, imgs_labeled, labels, ids)
        cluster_metrics['purity']['kmeans'].append(purity)
        cluster_metrics['precision']['kmeans'].append(precision)
        cluster_metrics['recall']['kmeans'].append(recall)
        cluster_metrics['f1']['kmeans'].append(f1)
        cluster_metrics['rand index']['kmeans'].append(rand_index)
        cluster_metrics['loss']['kmeans'].append(loss)

    return cluster_metrics


def clustering_distances(feats, imgs_labeled, labels, ids, distances):
    distance_metrics = {'purity': [], 'precision': [], 'recall': [], 'f1': [], 'rand index': []}

    for distance in distances:
        purity, precision, recall, f1, rand_index = agglomerative_clustering(None, feats, imgs_labeled, labels, ids,
                                                                             distance)
        distance_metrics['purity'].append(purity)
        distance_metrics['precision'].append(precision)
        distance_metrics['recall'].append(recall)
        distance_metrics['f1'].append(f1)
        distance_metrics['rand index'].append(rand_index)

    return distance_metrics


def optics(feats, imgs_labeled, labels, ids, samples):
    metrics = {'purity': [], 'precision': [], 'recall': [], 'f1': [], 'rand index': []}

    for min_samples in samples:
        purity, precision, recall, f1, rand_index = optics_clustering(feats, imgs_labeled, labels, ids, min_samples)

        metrics['purity'].append(purity)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['rand index'].append(rand_index)

    return metrics
