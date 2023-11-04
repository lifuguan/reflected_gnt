import os
import numpy as np

if os.path.exists('data/scannet') and os.path.exists('data/Replica'):
    replica_train = np.loadtxt('configs/replica_train_split.txt',dtype=str).tolist()
    replica_train = [replica_train] if type(replica_train) is str else replica_train
    replica_test  = np.loadtxt('configs/replica_test_split.txt',dtype=str).tolist()
    replica_test = [replica_test] if type(replica_test) is str else replica_test
    scannet_train_scans_320 = np.loadtxt('configs/scannetv2_train_split.txt',dtype=str).tolist()
    scannet_test_scans_320 = np.loadtxt('configs/scannetv2_test_split.txt',dtype=str).tolist()
    scannet_val_scans_320 = np.loadtxt('configs/scannetv2_val_split.txt',dtype=str).tolist()
    scannet_single = ['scannet/scene0376_02/black_320']
