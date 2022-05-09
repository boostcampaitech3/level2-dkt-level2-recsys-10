sweep_config = {
    'name' : 'lr_test',
    'method': 'bayes',
    'metric' : {
        'name': 'valid_auc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'lr': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            }
    }
}