model_params = {
    'embed_dim': 512,
    'model_dim': 512,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3
}

training_params = {
    'num_warmup_steps': 100,
    'learning_rate': 5e-4,
    'dropout': 0.1,
    'label_smoothing': 0.0,
    'batch_size': 128,
    'warm_up': True
}

config = {
    'model_name': 'TestModel',
    'src_language': 'de',
    'tgt_language': 'en',
    'gpu': 'mps',
    **model_params,
    **training_params
}