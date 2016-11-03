import graph_metrics
import data_load

if __name__ == '__main__':

    file_names = [
        'normal_no_outliers_adj_undir',
        'ad_no_outliers_adj_undir',
        'EMCI_no_outliers_adj_undir',
        'LMCI_no_outliers_adj_undir'
    ]

    file_labels = {
        'normal_no_outliers_adj_undir': 1,
        'ad_no_outliers_adj_undir': 2,
        'EMCI_no_outliers_adj_undir': 3,
        'LMCI_no_outliers_adj_undir': 4
    }

    feature_functions = [
        graph_metrics.closed_walks
    ]

    loaded_fv = data_load.load_feature_vectors(file_names, file_labels, feature_functions, True)

    labels = loaded_fv[:,-1]
    print('Number of data points : %i' % len(labels))
