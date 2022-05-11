from curses import use_default_colors
import os
import random

import numpy as np
import torch


def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_feats_sweep_dict(feats:list) -> dict:
    feats_dict = {feat:{'values':[True,False]} for feat in feats}

    return feats_dict

def get_wandb_config(wandb_config):
    feats = wandb_config['FEATS']
    use_feats = []
    for feat,v in wandb_config.items():
        if feat in feats and v:
            use_feats.append(feat)
    return use_feats

def get_cate_cols(preprocess):
    feats=preprocess.FEATS

    cate_cols=preprocess.CATS
    selected_cate_cols = []
    for cate in cate_cols:
        if cate in feats:
            selected_cate_cols.append(cate)

    return selected_cate_cols