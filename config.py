model_params = {
    'embed_dim': 512,
    'model_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
}

optimizer_params = {
    'learning_rate': 5e-4,
    'warmup': 100
}

training_params = {
    'batch_size': 128,
}

config = {
    'model_name': 'TestModel',
    'model_params': model_params,
    'optimizer_params': optimizer_params,
    'training_params': training_params,
}