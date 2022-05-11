from args import parse_args
from tabular.dataloader import Preprocess
# from tabular.utils import get_feats_sweep_dict

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
            'min': 1,
            'max': 3000
            },

        'early_stopping_rounds':{
            'distribution': 'int_uniform',
            'min': 10,
            'max': 300
            },
        'max_depth':{
            # 'values':[i for i in range(-1,30,2)] # LGBM  
            'values':[i for i in range(1,16,2)] # CatBoost 최대 16
            },
    }
}