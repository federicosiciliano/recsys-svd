# SVD-based backbone initialization in Neural RecSys

This work attempts to enhance the quality of the learned representations by
optimizing the item embeddings initialization (that typically consists of random
values) of modern neural recommenders through a traditional matrix factorization
technique, which is **Singular Value Decomposition (SVD)**, to be applied on the
interaction matrix built from the available training data.

Experiments were carried on the *Self-Attentive Sequential Recommender* (https://api.semanticscholar.org/CorpusID:52127932) using the *MovieLens 1M* dataset (available at https://grouplens.org/datasets/movielens/1m/). The model is built using *PyTorch* and *PyTorch Lightning*. 

Authors: 
- Federico Siciliano
- Maria Diana Calugaru