import math
import os

import numpy as np
import torch
import wandb
import lightgbm as lgb
from wandb.lightgbm import wandb_callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import joblib

def run(args, train_data, valid_data, X_valid, y_valid):
    if args.model == 'lightgbm':
        model = lgb.train(
            {'objective': args.objective}, 
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=args.num_boost_round,
            callbacks=[
                wandb_callback(), 
                lgb.early_stopping(stopping_rounds = args.early_stopping_rounds), 
                lgb.log_evaluation(period = args.verbose_eval)
                ]
            )
    
    auc, acc = model_predict(model, X_valid, y_valid)
    save_model(args, model)
    wandb.log({'valid_auc':auc, 'valid_acc':acc})

def model_predict(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    return auc, acc

def inference(args, test_data):    
    model = load_model(args)

    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']
    # MAKE PREDICTION
    total_preds = model.predict(test_data[FEATS])

    write_path = os.path.join(args.output_dir, "submission.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))

def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    # load_state = torch.load(model_path)
    # model = get_model(args)
    model = joblib.load(model_path)

    # # load model state
    # model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model

def save_model(args, model):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Saving Model from:", model_path)

    joblib.dump(model, model_path)