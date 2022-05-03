import math
import os

import numpy as np
import torch
import wandb
import lightgbm as lgb
from catboost import CatBoostClassifier
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
    elif args.model == 'catboost':
        model = CatBoostClassifier(
            iterations=args.num_boost_round,
            learning_rate=0.001, # TODO lr 관련 파라미터 확인하기
            task_type='GPU' # TODO GPU 사용 가능할 때만 사용하록 if 문으로 변경
        )
        model.fit(
            train_data, 
            eval_set=valid_data, 
            use_best_model=True, 
            early_stopping_rounds=args.early_stopping_rounds, 
            verbose=args.verbose_eval)

    
    auc, acc = model_predict(args, model, X_valid, y_valid)
    save_model(args, model)
    wandb.log({'valid_auc':auc, 'valid_acc':acc})

def model_predict(args, model, X_valid, y_valid):
    if args.model == 'lightgbm':
        preds = model.predict(X_valid)
    elif args.model == 'catboost':
        preds = model.predict(X_valid, prediction_type='Probability')[:, 1]

    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    return auc, acc

def inference(args, test_data):    
    model = load_model(args)
    total_preds = model.predict(test_data)

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