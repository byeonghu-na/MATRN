import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from fastai.vision import *

from modules.attention import *
from modules.model import Model, _default_tfmer_cfg
from modules.transformer import (PositionalEncoding,
                                 TransformerEncoder,
                                 TransformerEncoderLayer)


class BaseSemanticVisual_backbone_feature(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_alignment_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_alignment_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_alignment_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_alignmentl_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_alignment_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_alignment_num_layers, 2)

        self.mask_example_prob = ifnone(config.model_alignment_mask_example_prob, 0.9)
        self.mask_candidate_prob = ifnone(config.model_alignment_mask_candidate_prob, 0.9)
        self.num_vis_mask = ifnone(config.model_alignment_num_vis_mask, 10)
        self.nhead = nhead

        self.d_model = d_model
        self.use_self_attn = ifnone(config.model_alignment_use_self_attn, False)
        self.loss_weight = ifnone(config.model_alignment_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.model = TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder_tfm = PositionalEncoding(d_model, dropout=0, max_len=8*32)

        mode = ifnone(config.model_alignment_attention_mode, 'nearest')
        self.model2_vis = PositionAttention(
            max_length=config.dataset_max_length + 1,  # additional stop token
            mode=mode
        )
        self.cls_vis = nn.Linear(d_model, self.charset.num_classes)
        self.cls_sem = nn.Linear(d_model, self.charset.num_classes)
        self.w_att = nn.Linear(2 * d_model, d_model)

        v_token = torch.empty((1, d_model))
        self.v_token = nn.Parameter(v_token)
        torch.nn.init.uniform_(self.v_token, -0.001, 0.001)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

    def forward(self, l_feature, v_feature, lengths_l=None, v_attn=None, l_logits=None, texts=None, training=True):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, E, H, W)
            lengths_l: (N,)
            v_attn: (N, T, H, W)
            l_logits: (N, T, C)
            texts: (N, T, C)
        """
        padding_mask = self._get_padding_mask(lengths_l, self.max_length)

        l_feature = l_feature.permute(1, 0, 2)  # (T, N, E)
        N, E, H, W = v_feature.size()
        v_feature = v_feature.view(N, E, H*W).contiguous().permute(2, 0, 1)  # (H*W, N, E)

        if training:
            n, t, h, w = v_attn.shape
            v_attn = v_attn.view(n, t, -1) # (N, T, H*W)
            for idx, length in enumerate(lengths_l):
                if np.random.random() <= self.mask_example_prob:
                    l_idx = np.random.randint(int(length))
                    v_random_idx = v_attn[idx, l_idx].argsort(descending=True).cpu().numpy()[:self.num_vis_mask,]
                    v_random_idx = v_random_idx[np.random.random(v_random_idx.shape) <= self.mask_candidate_prob]
                    v_feature[v_random_idx, idx] = self.v_token

        if len(v_attn.shape) == 4:
            n, t, h, w = v_attn.shape
            v_attn = v_attn.view(n, t, -1) # (N, T, H*W)

        zeros = v_feature.new_zeros((h*w, n, E))  # (H*W, N, E)
        base_pos = self.pos_encoder_tfm(zeros)  # (H*W, N, E)
        base_pos = base_pos.permute(1, 0, 2) # (N, H*W, E)

        base_pos = torch.bmm(v_attn, base_pos) # (N, T, E)
        base_pos = base_pos.permute(1, 0, 2) # (T, N, E)

        l_feature = l_feature + base_pos

        sv_feature = torch.cat((v_feature, l_feature), dim=0)  # (H*W+T, N, E)
        sv_feature = self.model(sv_feature)  # (H*W+T, N, E)

        sv_to_v_feature = sv_feature[:H*W]  # (H*W, N, E)
        sv_to_s_feature = sv_feature[H*W:]  # (T, N, E)

        sv_to_v_feature = sv_to_v_feature.permute(1, 2, 0).view(N, E, H, W)
        sv_to_v_feature, _ = self.model2_vis(sv_to_v_feature)  # (N, T, E)
        sv_to_v_logits = self.cls_vis(sv_to_v_feature)  # (N, T, C)
        pt_v_lengths = self._get_length(sv_to_v_logits)  # (N,)

        sv_to_s_feature = sv_to_s_feature.permute(1, 0, 2)  # (N, T, E)
        sv_to_s_logits = self.cls_sem(sv_to_s_feature)  # (N, T, C)
        pt_s_lengths = self._get_length(sv_to_s_logits)  # (N,)

        f = torch.cat((sv_to_v_feature, sv_to_s_feature), dim=2)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * sv_to_v_feature + (1 - f_att) * sv_to_s_feature

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'logits': logits, 'pt_lengths': pt_lengths, 'loss_weight':self.loss_weight*3,
                'v_logits': sv_to_v_logits, 'pt_v_lengths': pt_v_lengths,
                's_logits': sv_to_s_logits, 'pt_s_lengths': pt_s_lengths,
                'name': 'alignment'}
