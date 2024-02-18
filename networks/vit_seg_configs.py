import ml_collections

def get_base_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (7, 7, 7)})
    config.hidden_size = 512 #768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024 #3072
    config.transformer.num_heads = 8 #12
    config.transformer.num_layers = 4 #4
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0

    config.classifier = 'seg'
    config.representation_size = None
    
    config.temp = 0.1
    config.mb_size = 10
    
    config.embedding_dim = 32
    config.vit_patches_size = 16
    
    config.base_width = 64 # 32
    config.decoder_channels = (256, 128, 64, 16)
    config.decoder_use_bn = False
    config.activation = 'softmax'
    return config

def get_vit_3d_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_base_config()
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (2, 3, 4)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '/shenlab/lab_stor4/xychen/segStructuralSemanticConstraint/BrainSegmentation/MultiProxiesNoAge/TransUNet/networks/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (config.base_width * 8, config.base_width * 4, config.base_width * 1, config.base_width * 1) # (256, 128, 64, 32) # (base_width * 8, ...)
    config.skip_channels = [config.base_width * 8, config.base_width * 4, config.base_width * 1] # (base_width * 8, base_width * 4, base_width // 2) : feature maps from encoder
    config.n_skip = 3
    config.activation = 'softmax'

    return config


