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
        'max_seq_len':{
            'distribution': 'int_uniform',
            'min': 5,
            'max': 300
            },
        'hidden_dim' : {
            'values': [8, 16 ,32, 64] # hidden_size는 h_heads의 배수여야함
        },
        'n_heads':{  
            'values': [1,2,4,8]  # Multi-Head-Attention 모델에만 적용 
        },
        'n_layers': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 10
            },
        'drop_out': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.9
            },
    }
}