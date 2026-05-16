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
