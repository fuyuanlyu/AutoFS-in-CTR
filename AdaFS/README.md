# AdaFS
This repository contains PyTorch Implementation of KDD 2022 paper:
  - [AdaFS: Adaptive Feature Selection in Deep Recommender System](https://dl.acm.org/doi/abs/10.1145/3534678.3539204)

### Run

Running AdaFS:
```
python -u train.py --dataset $YOUR_DATASET --model $YOUR_MODEL \
        --gpu $GPU --lr $LR --l2 $L2 --arch_lr $ARCH_LR \
        --pretrain $PRETRAIN \
```

You can choose `YOUR_DATASET` from \{Criteo, Avazu, KDD12\} and `YOUR_MODEL` from \{FM, DeeepFM, DCN, IPNN\}. Here we empirically set $ARCH_LR=$LR and choose $PRETRAIN from \{1, 2, 5\}
