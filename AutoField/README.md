# OptFS
This repository contains PyTorch Implementation of WWW 2022 paper:
  - [AutoField: Automating Feature Selection in Deep Recommender Systems](https://dl.acm.org/doi/10.1145/3485447.3512071)

### Run

Running Search Process:
```
python -u search.py --dataset $YOUR_DATASET --model $YOUR_MODEL \
        --debug_mode 1 --gpu $GPU --lr $LR --l2 $L2 --arch_lr $ARCH_LR \
        --save_name $PATH_TO_SEARCH_RESULT \
```

You can choose `YOUR_DATASET` from \{Criteo, Avazu, KDD12\} and `YOUR_MODEL` from \{FM, DeeepFM, DCN, IPNN\}


Running Retrain Process:
```
python -u retrain.py --dataset $YOUR_DATASET --model $YOUR_MODEL \
        --debug_mode 0 --gpu $GPU --lr $LR --l2 $L2 \
        --arch_file $PATH_TO_SEARCH_RESULT
```
