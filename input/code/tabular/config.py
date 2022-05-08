sweep_config = {
    'name' : 'lr_test',
    'method': 'bayes',
    'metric' : {
        'name': 'validation_auc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            }
    }
}