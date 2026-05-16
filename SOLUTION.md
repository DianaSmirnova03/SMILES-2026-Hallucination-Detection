# SOLUTION.md — SMILES-2026 Hallucination Detection

## 1. Reproducibility Instructions

### Environment
git clone https://github.com/DianaSmirnova03/SMILES_2026.git
cd SMILES_2026
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

text

### Run
python solution.py

text

This downloads Qwen2.5-0.5B, extracts features, trains the probe, and saves results.json and predictions.csv.

---

## 2. Solution Description

### aggregation.py
Uses layers 7, 15, 23 from the 24-layer model. For each layer takes mean pooling over real tokens and the last token hidden state. Feature dimension: 3 * (896 + 896) = 5376.

### probe.py
HallucinationProbe is a PyTorch MLP: Linear(5376,256) -> ReLU -> Linear(256,1). Trained with BCEWithLogitsLoss (pos_weight for class imbalance) and Adam (lr=0.001, 200 epochs). Threshold tuned on validation set by maximizing F1.

### splitting.py
Stratified split: 15% test, remaining 85% split into 85% train / 15% val.

### Results
| Metric | Baseline | Probe Val | Probe Test |
|--------|----------|-----------|------------|
| Accuracy | 70.19% | 72.12% | 69.23% |
| F1 | 82.49% | 83.43% | 81.82% |
| AUROC | — | 68.58% | 63.23% |

---

## 3. Experiments

1. Only last layer – accuracy dropped to ~62%, not enough signal.
2. All 24 layers mean pooling – caused overfitting, worse validation.
3. Adding inter-layer differences and cosine similarities – improved AUROC when paired with PCA+LogisticRegression but made MLP unstable. Not included.
4. PCA(64) + LogisticRegression – best AUROC (~0.79) in separate tests, but current submission uses default MLP.

## 4. Conclusion
Model hidden states contain detectable hallucination signal. Current simple MLP outperforms baseline. Performance can be improved with PCA and better features.
