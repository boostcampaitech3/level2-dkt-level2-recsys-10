from optparse import Values
from args import parse_args
args = parse_args(mode="train")

sweep_config = {
    'name' : args.sweep_name,
    'method': args.sweep_method,
    'metric' : {
        'name': 'valid_auc',
        'goal': 'maximize'   
        },
    'parameters' : {
        # 훈련 파라미터
        'lr': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },
        'batch_size':{
            'values': [16, 64, 256, 1024]
        },
        'scheduler':{
            'values' : ['plateau','linear_warmup']
        },
        'optimizer':{
            'values' : ['adam','adamW']
        },

        # 모델 파라미터
        'max_seq_len' : {
            'values' : [5,10,20,30,50,100]
        },
        'hidden_dim' : {
            'values': [8, 16 ,32, 64] # hidden_size는 h_heads의 배수여야함
        },
        'n_layers':{
            'values': [1,2,3,4,5]
        },
        'n_heads':{  
            'values': [1,2,4,8]  # Multi-Head-Attention 모델에만 적용 
        },
        'drop_out':{
            'values': [0.1, 0.2, 0.5, 0.7]
        },

    }
}