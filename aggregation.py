with open('aggregation.py', 'w') as f:
    f.write('''import torch

def aggregate(hs, msk):
    lyr = [7, 15, 23]
    allf = []
    for l in lyr:
        h = hs[l]
        rl = msk.nonzero().squeeze()
        h_r = h[rl]
        mu = h_r.mean(dim=0)
        lst = h_r[-1]
        allf.append(mu)
        allf.append(lst)
    return torch.cat(allf)

def extract_geometric_features(hs, msk):
    lyr = [7, 15, 23]
    fe = []
    prev = None
    for l in lyr:
        h = hs[l]
        rl = msk.nonzero().squeeze()
        h_r = h[rl]
        mu = h_r.mean(dim=0)
        if prev is not None:
            dif = mu - prev
            cos_sim = torch.dot(mu, prev) / (mu.norm() * prev.norm() + 1e-8)
            fe.append(dif)
            fe.append(cos_sim.unsqueeze(0))
        prev = mu
    if fe:
        return torch.cat(fe)
    return torch.zeros(0)

def aggregation_and_feature_extraction(hs, msk, use_geometric):
    ag = aggregate(hs, msk)
    if use_geometric:
        g = extract_geometric_features(hs, msk)
        return torch.cat([ag, g])
    return ag
''')

# 2. Заменяем probe.py
with open('probe.py', 'w') as f:
    f.write('''import numpy as np
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
''')

# 3. Меняем флаг USE_GEOMETRIC на True в solution.py
with open('solution.py', 'r') as f:
    content = f.read()
content = content.replace('USE_GEOMETRIC = False', 'USE_GEOMETRIC = True')
with open('solution.py', 'w') as f:
    f.write(content) 
