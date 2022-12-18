# LPFS
This repository contains PyTorch Implementation of paper:
  - [LPFS: Learnable Polarizing Feature Selection for Click-Through Rate Prediction](https://arxiv.org/abs/2206.00267)

### Run

Running LPFS:
```
python -u train.py --dataset $YOUR_DATASET --model $YOUR_MODEL \
        --gpu $GPU --lr $LR --l2 $L2 \
        --epsilon $EPSILON --lam $LAM \
```

You can choose `YOUR_DATASET` from \{Criteo, Avazu, KDD12\} and `YOUR_MODEL` from \{FM, DeeepFM, DCN, IPNN\}. Here we empirically set $EPSILON and $LAM based on the following.


| Dataset   | Criteo | Avazu | KDD12 |
| ---------- | ---------- | ---------- | ---------- |
| EPSILON   | 1e-1 | 4e-3 | 1e-3 |
| LAM       | 1e-1 | 25e-2 | 3e-1 |