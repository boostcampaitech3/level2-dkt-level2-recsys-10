from args import parse_args
from tabular.dataloader import Preprocess
# from tabular.utils import get_feats_sweep_dict

# link:  https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5

args = parse_args(mode="train")

if args.sweep_feats:
    # FEATS 튜닝
    sweep_config = {
        'name' : args.model + ' : ' + args.sweep_name,
        'method': args.sweep_method,
        'metric' : {
            'name': 'validation_auc',
            'goal': 'maximize'   
            },
        # 'parameters' : {} # {'f1' :{ 'values' : [True, False]}, {f2 : ...} ... }
        }

else:
    # HyperParams 튜닝
    sweep_config = {
    'name' : args.model + ' : ' + args.sweep_name,
    'method': args.sweep_method,
    'metric' : {
        'name': 'validation_auc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },
        'num_boost_round' :{
            'distribution': 'int_uniform',
            'min': 100,
            'max': 3000
            },

        # 'early_stopping_rounds':{
        #     'distribution': 'int_uniform',
        #     'min': 10,
        #     'max': 300
        #     },

        'max_depth':{
            # 'values':[i for i in range(-1,30,2)] # LGBM
            # 'distribution': 'int_uniform',
            # 'min': 3,
            # 'max': 12  
            'values':[i for i in range(1,16,2)] # CatBoost 최대 16
            },

        'num_leaves':{
            'values':[i for i in range(20,3000,20)] # LGBM
            },

        'min_data_in_leaf':{
            'values':[i for i in range(200,10000,100)] # LGBM
            },

        'lambda_l1':{
            'values':[i for i in range(0,100,5)] # LGBM
            },

        'lambda_l2':{
            'values':[i for i in range(0,100,5)] # LGBM
            },

        'min_gain_to_split': {
            'distribution': 'uniform',
            'min': 0,
            'max': 15
            },
        
        'bagging_fraction': {
            'values':[i * 0.01 for i in range(20,95,10)] # LGBM
            },

        'feature_fraction': {
            'values':[i * 0.01 for i in range(20,95,10)] # LGBM
            },

        'bagging_freq': {
            'values':[0, 1] # LGBM
            },

        'path_smooth': {
            'distribution': 'uniform',
            'min': 1e-8,
            'max': 1e-3
            },
        
        'max_bin': {
            'distribution': 'int_uniform',
            'min': 10,
            'max': 255
            },

        }

    }
