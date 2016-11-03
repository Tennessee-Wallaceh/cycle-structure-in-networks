import numpy as np, h5py
import sys

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
