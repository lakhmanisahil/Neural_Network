import numpy as np
import os, pickle, tarfile, urllib.request

def download_and_extract_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    folder = "cifar-10-batches-py"
    if not os.path.exists(folder):
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
    return folder

def load_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch[b'data'], np.array(batch[b'labels'])

def load_cifar10_data(selected=[0, 1, 2]):
    folder = download_and_extract_cifar10()
    X_train, Y_train = [], []
    for i in range(1, 6):
        Xb, Yb = load_batch(f"{folder}/data_batch_{i}")
        X_train.append(Xb)
        Y_train.append(Yb)
    X_train = np.vstack(X_train)
    Y_train = np.hstack(Y_train)
    X_test, Y_test = load_batch(f"{folder}/test_batch")
    
    # Select specific classes
    mask_train = np.isin(Y_train, selected)
    mask_test = np.isin(Y_test, selected)
    X_train, Y_train = X_train[mask_train], Y_train[mask_train]
    X_test, Y_test = X_test[mask_test], Y_test[mask_test]
    
    # Remap labels to 0,1,2
    label_map = {c: i for i, c in enumerate(selected)}
    Y_train = np.vectorize(label_map.get)(Y_train)
    Y_test = np.vectorize(label_map.get)(Y_test)
    
    # Normalize
    X_train = (X_train.astype(np.float32) / 255.0)-0.5
    X_test = (X_test.astype(np.float32) / 255.0)-0.5
    return X_train, Y_train, X_test, Y_test
