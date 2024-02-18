# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from . import vit_seg_configs as configs
from .vit_seg_modeling_encoder import FeatureExtractor
from .loss import BalancedCELoss, DiceLoss


logger = logging.getLogger(__name__)

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


#class Attention(nn.Module):
#    def __init__(self, config, vis):
#        super(Attention, self).__init__()
#        self.vis = vis
#        self.num_attention_heads = config.transformer["num_heads"]
#        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#        self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#        self.query = Linear(config.hidden_size, self.all_head_size)
#        self.key = Linear(config.hidden_size, self.all_head_size)
#        self.value = Linear(config.hidden_size, self.all_head_size)
#
#        self.out = Linear(config.hidden_size, config.hidden_size)
#        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
#
#        self.softmax = Softmax(dim=-1)
#
#    def transpose_for_scores(self, x):
#        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#        x = x.view(*new_x_shape)
#        return x.permute(0, 2, 1, 3)
#
#    def forward(self, hidden_states):
#        mixed_query_layer = self.query(hidden_states)
#        mixed_key_layer = self.key(hidden_states)
#        mixed_value_layer = self.value(hidden_states)
#
#        query_layer = self.transpose_for_scores(mixed_query_layer)
#        key_layer = self.transpose_for_scores(mixed_key_layer)
#        value_layer = self.transpose_for_scores(mixed_value_layer)
#
#        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#        attention_probs = self.softmax(attention_scores)
#        weights = attention_probs if self.vis else None
#        attention_probs = self.attn_dropout(attention_probs)
#
#        context_layer = torch.matmul(attention_probs, value_layer)
#        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#        context_layer = context_layer.view(*new_context_layer_shape)
#        attention_output = self.out(context_layer)
#        attention_output = self.proj_dropout(attention_output)
#        return attention_output, weights

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.num_attention_heads, 1, 1))), requires_grad=True)

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        
        self.use_dropout = False
        if config.transformer["attention_dropout_rate"] > 0.0:
            self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
            self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
            self.use_dropout = True

        # self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # cosine attention
        attention_scores = (F.normalize(query_layer, dim=-1) @ F.normalize(key_layer, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attention_scores = attention_scores * logit_scale

        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # attention_probs = self.softmax(attention_scores)
        
        ## Numerical stable alternative ##
        max_along_axis = torch.max(attention_scores, dim=-1, keepdim=True).values
        exp_logits = torch.exp(attention_scores-max_along_axis)
        attention_probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
        
        weights = attention_probs if self.vis else None
        if self.use_dropout:
            attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        if self.use_dropout:
            attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        
        self.use_dropout = False
        if config.transformer["dropout_rate"] > 0.0:
            self.dropout = Dropout(config.transformer["dropout_rate"])
            self.use_dropout = True

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        assert len(img_size) == 3
        
        grid_size = [int(x / config.vit_patches_size) for x in img_size]
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1], img_size[2] // 16 // grid_size[2])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16, patch_size[2] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) * (img_size[2] // patch_size_real[2])
        
        self.hybrid_model = FeatureExtractor(base_width=config.base_width, block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor, in_channels=in_channels)
        
        in_channels = self.hybrid_model.width * 16 # in_channels: 1024
        
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        
        if config.hidden_size != 512:
            self.modality_embedding = Linear(in_features=512, out_features=config.hidden_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        
        self.use_dropout = False
        if config.transformer["dropout_rate"] > 0.0:
            self.dropout = Dropout(config.transformer["dropout_rate"])
            self.use_dropout = True


    def forward(self, x, modality=None):
        x, features, full = self.hybrid_model(x)
        
        # x: [B, 1024, 8, 8, 8]
        # features[0]: [B, 512, 16, 16, 16]
        # features[1]: [B, 256, 32, 32, 32]
        # features[3]: [B,  64, 64, 64, 64]
        
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/3), n_patches^(1/3), n_patches^(1/3))
        
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        
        embeddings = x + self.position_embeddings
        
        # print("\nembeddings: ", embeddings.size())
        # print("modality: ", modality.size())
        # print("modality[:, None, :]: ", modality[:, None, :].size())
        # assert 0
        
        # if self.config.hidden_size != 512:
        #     embeddings = embeddings + self.modality_embedding(modality)[:, None, :]
        # else:
        #     embeddings = embeddings + modality[:, None, :]
        
        if self.use_dropout:
            embeddings = self.dropout(embeddings)
        
        return embeddings, features, full


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x # x: [B, 512, 768]
        x = self.attention_norm(x)
        x, weights = self.attn(x) # x: [B, 512, 768]
        x = x + h
        
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            # hidden_states: [B, 512, 768]
            
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=self.in_channels)
        
        self.encoder = Encoder(config, vis)
    
    def forward(self, input_volumes):
        embedding_output, features, full = self.embeddings(input_volumes)
        
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        
        return encoded, attn_weights, features, full

class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def convkxkxk(cin, cout, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, groups=groups)

class Conv3dNormReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = convkxkxk(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        
        if use_batchnorm:
            bn_or_in = nn.BatchNorm3d(out_channels)
        else:
            bn_or_in = nn.GroupNorm(out_channels, out_channels, eps=1e-6)
        
        relu = nn.ReLU(inplace=True)
        
        super(Conv3dNormReLU, self).__init__(conv, bn_or_in, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dNormReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dNormReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv3d)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv3dNormReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=config.decoder_use_bn,
        )
        
        in_channels = [head_channels] + self.config.skip_channels[1:]
        skip_channels = self.config.skip_channels
        out_channels = self.config.skip_channels[1:] + [self.config.skip_channels[-1]]
        
        use_batchnorms = [config.decoder_use_bn] * len(in_channels)
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, use_bn) for in_ch, out_ch, sk_ch, use_bn in zip(in_channels, out_channels, skip_channels, use_batchnorms)
        ]
        
        self.blocks = nn.ModuleList(blocks)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.conv1 = Conv3dNormReLU(out_channels[-1]+skip_channels[-1],
                                    out_channels[-1],
                                    kernel_size=3,
                                    padding=1,
                                    use_batchnorm=config.decoder_use_bn)
        
        self.conv2 = nn.Conv3d(out_channels[-1], out_channels[-1], kernel_size=3, padding=1)

    def forward(self, hidden_states, features=None, level1=None):
        
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, d, h, w, hidden)
        d, h, w = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, d, h, w)
        x = self.conv_more(x)
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        
        x = self.up(x)
        if level1 is not None:
            x = torch.cat([x, level1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2, in_channels=1, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, in_channels, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=1,
        )
        self.config = config
    
    def forward(self, x, gt=None):
        
        x, attn_weights, features, full = self.transformer(x)  # (B, n_patch, hidden)
        
        x = self.decoder(x, features, full)
        
        logits = self.segmentation_head(x)
        
        return logits

CONFIGS = {
    'ViT-3D': configs.get_vit_3d_config(),
}
