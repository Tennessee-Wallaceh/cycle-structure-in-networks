import numpy as np, h5py
import sys, os

def load_data(file_name):
    """
    Load graph data from a file.
    Expect data files to be stored in data/ directory.
    ----------------------------------------------------------------------------
    file_name: str
    ----------------------------------------------------------------------------
    return: np.ndarray
    """

    print('Loading raw data from  data/%s.mat ...'%file_name)

    try:
        f = h5py.File('data\\' + file_name + '.mat', 'r')
        data = np.array([f[entry[0]] for entry in f.get(file_name)])
    except:
        sys.exit('Unable to load data/%s.mat'%file_name)

    print('Data information:\n\n\tentry count - %i'%len(data))

    # check that data has consistant shape and type
    if all(x.shape == data[0].shape and type(x) == type(data[0]) for x in data):
        print('\n\tentry type - %s'%type(data[0]))
        print('\n\tentry dimension - %s\n'%str(data[0].shape))
    else:
        sys.exit('Data loaded from data/%s.mat is inconsistent'%file_name)

    return data

def build_feature_vectors(file_names, file_labels, feature_functions, save=False):
    """
    Build feature vectors for a set of files and feature functions. Each feature
    function should be applied to a single adjacency matrix from a file.
    Save to file if needed. Files will be saved in data/features/ in the
    following format:
        - 'file_name1+file_name2__feature_function1.npy'
    ----------------------------------------------------------------------------
    file_names: list of file names containing data
    file_labels: dict mapping file names to numeric label
    feature_functions: list of functions that generate a feature vector from an
        adjacency matrix
    save: bool, whether or not the file should be saved
    ----------------------------------------------------------------------------
    return: np.ndarray, the 2d array of labelled feature vectors
    """

    def apply_feature_build_functions(a, f_n):

        feature_vectors = np.array([
            fcn(a) for fcn in feature_functions
        ])
        print(np.append(
            np.concatenate(feature_vectors),
            file_labels[f_n]
        ))

        return np.append(
            np.concatenate(feature_vectors),
            file_labels[f_n]
        )

    # from each file in file_names load the adjacency matricies, apply the
    # feature functions to each matrix and store in np ndarray, concatenate the
    # features from each file.
    feature_vectors = np.concatenate([
        np.array([
            apply_feature_build_functions(a, f_n) for a in load_data(f_n)
        ])
        for f_n in file_names
    ])

    if save:
        save_name = '+'.join(file_names) + '__' + '+'.join([f.__name__ for f in feature_functions])
        print('Saved feature vector %s in data/features/' % save_name)
        np.save('data/features/'+save_name, feature_vectors)

    return np.array(feature_vectors)

def load_feature_vectors(file_names, file_labels, feature_functions, save=False):
    """
    Try to load feature vectors for configuration specified by file_names and
    feature_functions, if the file can't be found re-build the vectors.
    Save to file if needed.
    ----------------------------------------------------------------------------
    file_names: list of file names containing data
    file_labels: dict mapping file names to numeric label
    feature_functions: list of functions that generate a feature vector from an
        adjacency matrix
    save: bool, whether or not the file should be saved when it is re-built
    ----------------------------------------------------------------------------
    return np.ndarray, feature vectors generated from feature functions for
        graphs stored in file names.
    """
    # format of saved files
    save_name = '+'.join(file_names) + '__' + '+'.join(
        [f.__name__ for f in feature_functions]
    )

    if os.path.isfile('data/features/'+save_name+'.npy'):
        print('Loaded %s.npy' % save_name)
        return np.load('data/features/'+save_name+'.npy')[0]

    # if file can't be found build the vectors
    return build_feature_vectors(file_names, file_labels, feature_functions, save)
