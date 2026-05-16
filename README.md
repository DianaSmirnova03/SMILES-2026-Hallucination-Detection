# SOLUTION.md

## 1. How to run
- Clone repo and go to folder.
- `pip install -r requirements.txt`
- `python solution.py`
- It will make `results.json` and `predictions.csv`.

## 2. What I changed and why

### aggregation.py
I take hidden states from layers 7, 15 and 23 (not too early, not only last). For each layer I get mean of all real tokens and the last token. Then I also make geometric features: difference between mean vectors of consecutive layers and their cosine similarity. This gives the model clues about how the representation changes across layers.

I chose layer 7 because first layers just learn local things. Layer 15 is middle where facts are stored. Layer 23 is near final output. Difference features show if the model is uncertain (big changes).

### probe.py
I use StandardScaler to make features same scale, then PCA to 64 components because we have only 689 samples and too many features cause overfitting. Then a LogisticRegression with C=0.5 to not overfit and balanced class weight because hallucinated are less than truthful. I also tune the decision threshold on validation set for best F1.

### splitting.py
Just a simple 85% train+val, 15% test stratified split. Then from train+val I split 15% for validation. Stratify keeps class ratio same.

## 3. Experiments that did not work

### Using only last layer
I tried only the mean of the last layer. Accuracy was only ~0.62, much worse. So I added more layers.

### Using all 24 layers
I concatenated mean of every layer (24 * 896 = too many). With PCA to 64 I got worse validation accuracy than using just 3 layers, probably because early layers are noise.

### RandomForest instead of LogisticRegression
I tried RandomForest with 100 trees. It was very slow and gave same results, so not worth it.

### No PCA
Without PCA LogisticRegression overfit a lot, val accuracy was ~0.60. So PCA is needed.

## 4. Conclusion
The best working parts: using 3 middle+late layers, adding geometric features, PCA reduction, balanced logistic regression, and threshold tuning. Final val accuracy around 0.73 and test AUROC about 0.79. The model’s internal states clearly carry signal if you look at how the representation moves from layer to layer.
