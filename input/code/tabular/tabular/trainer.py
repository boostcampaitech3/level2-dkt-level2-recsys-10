from itertools import count
import math
import os

import numpy as np
import wandb
import lightgbm as lgb
from catboost import CatBoostClassifier
from wandb.lightgbm import wandb_callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

from .utils import get_cate_cols, get_wandb_config, get_cate_cols

def run(args, train_data, valid_data, X_valid, y_valid, preprocess):
    if args.sweep_feats:
    # DataLoad 시에 Feature를 변경함
        preprocess.FEATS = get_wandb_config(args)
        train_data = preprocess.get_train_data()
        valid_data = preprocess.get_valid_data()
        train_data, valid_data, X_valid, y_valid = preprocess.convert_dataset(train_data, valid_data)
        X_train = train_data
        y_train = valid_data
        preprocess.CATS = get_cate_cols(preprocess)
        print('********************preprocess.FEATS after*************************') 
        print(preprocess.FEATS)
        print(preprocess.CATS)

    if args.model == 'lightgbm':
        custom_loss = ["auc", lgb_acc]

        X_train = train_data
        y_train = valid_data
        #########################

        model = lgb.LGBMClassifier(
        learning_rate = args.learning_rate,
        n_estimators  = args.num_boost_round,
        max_depth = args.max_depth,
        path_smooth = 0.10
        )

        model.fit(
            X = X_train[preprocess.FEATS], 
            y = y_train,
            eval_set = [(X_train[preprocess.FEATS],y_train),(X_valid[preprocess.FEATS],y_valid)],
            eval_names=['train','validation'],
            eval_metric = custom_loss,
            verbose=args.verbose_eval,
            early_stopping_rounds=args.early_stopping_rounds
        )

    elif args.model == 'catboost':
        custom_loss = ["AUC", "Accuracy"]

        model = CatBoostClassifier(
            iterations=args.num_boost_round,
            learning_rate=args.learning_rate, # TODO lr 관련 파라미터 확인하기
            task_type='GPU', # TODO GPU 사용 가능할 때만 사용하록 if 문으로 변경
            custom_loss = custom_loss,
            max_depth = args.max_depth
        )
        model.fit(
            train_data, 
            eval_set=valid_data, 
            use_best_model=True, 
            # cat_features = preprocess.CATS,# TODO category feature 목록 넣기
            early_stopping_rounds=args.early_stopping_rounds, 
            verbose=args.verbose_eval)

    
    args.auc, acc = model_predict(args, model, X_valid, y_valid)
    print("------args.auc--------")
    print(args.auc)
    print(str(args.auc)[2:6])
    save_model(args, model)

    if args.model == 'lightgbm':
        eval_result = model.evals_result_
        list_run = ['train', 'validation']
        custom_loss = ["auc", "acc"]

        loop_len = len(eval_result["validation"]["auc"])
        for i in range(0, loop_len):
            wandb.log({
                "train_loss" : eval_result[list_run[0]]['binary_logloss'][i],
                "train_auc" : eval_result[list_run[0]][custom_loss[0]][i],
                "train_acc" : eval_result[list_run[0]][custom_loss[1]][i],
                "validation_auc" : eval_result[list_run[1]][custom_loss[0]][i],
                "validation_acc" : eval_result[list_run[1]][custom_loss[1]][i],
                "itration" : i
            })   

    elif args.model == 'catboost':
        eval_result = model.get_evals_result()
        print("------eval_result------")
        print(eval_result['validation'].keys())
        list_run = ['learn', 'validation']
        loop_len = len(eval_result["validation"]["AUC"])

        for i in range(0, loop_len):
            wandb.log({
                "train_loss" : eval_result[list_run[0]]['Logloss'][i],
                "train_acc" : eval_result[list_run[0]][custom_loss[1]][i],
                "validation_auc" : eval_result[list_run[1]][custom_loss[0]][i],
                "validation_acc" : eval_result[list_run[1]][custom_loss[1]][i],
                "itration" : i
            })

def lgb_acc(y_true, y_pred):
    y_pred = np.where(y_pred >= 0.5, 1., 0.)   
    return ('acc', np.mean(y_pred==y_true), False)
    
def custom_acc(y_pred, dataset):
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm.train
    y_true = dataset.get_label()
    y_pred = np.where(y_pred >= 0.5, 1., 0.)
    
    return ('acc', np.mean(y_pred==y_true), False)

def model_predict(args, model, X_valid, y_valid):
    if args.model == 'lightgbm':
        preds = model.predict(X_valid)
        print("------preds-----")
        print(preds)
    elif args.model == 'catboost':
        preds = model.predict(X_valid, prediction_type='Probability')[:, 1]
        

    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    return auc, acc

def inference(args, test_data):    
    model = load_model(args)
    # total_preds = model.predict(test_data)
    total_preds = model.predict_proba(test_data)[:,1]


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
    model_path = os.path.join(args.model_dir, str(args.auc)[2:6] + args.model_name)
    print("Saving Model from:", model_path)
    joblib.dump(model, model_path)