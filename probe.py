%%writefile probe.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

class HallucinationProbe:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=64, random_state=42)
        self.clf = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced', random_state=42)
        self.thr = 0.5

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        self.clf.fit(Xp, y)
        return self

    def fit_hyperparameters(self, Xv, yv):
        p = self.predict_proba(Xv)[:, 1]
        best_f1 = -1
        best_t = 0.5
        for t in np.unique(p):
            pr = (p >= t).astype(int)
            sc = f1_score(yv, pr, zero_division=0)
            if sc > best_f1:
                best_f1 = sc
                best_t = t
        self.thr = best_t
        return self

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.thr).astype(int)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        pr = self.clf.predict_proba(Xp)
        return pr
