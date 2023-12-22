from sklearn import preprocessing
from aux import *
from tp2_aux import *
from hierarchical_kmeans import *
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap


def main():
    dataset = images_as_matrix()

    dataset = preprocessing.normalize(dataset)

    pca_feats = PCA(n_components=6).fit_transform(dataset)

    kpca_feats = KernelPCA(n_components=6, kernel='rbf').fit_transform(dataset)

    isomap_feats = Isomap(n_components=6).fit_transform(dataset)

    feats = np.concatenate((np.concatenate((pca_feats, kpca_feats), axis=1), isomap_feats), axis=1)

    feats = preprocessing.normalize(feats)

    n_clusters = np.arange(1, 10, 1)
    distances = np.arange(0.1, 6.0, 0.1)
    min_samples = np.arange(1, 15, 1)

    ids_labels = np.loadtxt('labels.txt', delimiter=',')
    ids = ids_labels[:, 0]
    labels = ids_labels[:, 1]
    imgs_labeled = labels != 0

    cluster_metrics = clustering(feats, imgs_labeled, labels, ids, n_clusters)

    distance_metrics = clustering_distances(feats, imgs_labeled, labels, ids, distances)

    optics_metrics = optics(feats, imgs_labeled, labels, ids, min_samples)

    plot_measure_nclusters(cluster_metrics, n_clusters)
    plot_measure_distances(distance_metrics, distances)
    plot_measure_minsamples(optics_metrics, min_samples)

    # Generate reports
    best_ncluster_agg = 3
    agglomerative_clustering(best_ncluster_agg, feats, imgs_labeled, labels, ids, None)
    best_ncluster_spec = 3
    spectral_clustering(best_ncluster_spec, feats, imgs_labeled, labels, ids)
    best_ncluster_kmeans = 3
    kmeans_clustering(best_ncluster_kmeans, feats, imgs_labeled, labels, ids)


    biseckmeans = HierarchicalKMeans(3)
    res = biseckmeans.fit_predict(feats)
    report_clusters_hierarchical(ids, res, 'bisecting_kmeans_clustering.html')

    print('END')


if __name__ == "__main__":
    main()
