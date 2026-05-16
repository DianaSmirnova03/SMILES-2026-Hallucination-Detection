# splitting.py
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(y, df=None):
    idx = np.arange(len(y))
    i_tr_val, i_te = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)
    i_tr, i_val = train_test_split(i_tr_val, test_size=0.15/(1-0.15), random_state=42, stratify=y[i_tr_val])
    return [(i_tr, i_val, i_te)]
