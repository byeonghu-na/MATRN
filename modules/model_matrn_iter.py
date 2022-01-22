import torch
import torch.nn as nn
from fastai.vision import *

from .model_vision import BaseVision
from .model_language import BCNLanguage
from .model_semantic_visual_backbone_feature import BaseSemanticVisual_backbone_feature


class MATRN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.test_bh = ifnone(config.test_bh, None)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.semantic_visual = BaseSemanticVisual_backbone_feature(config)

    # def forward(self, images, *args):
    def forward(self, images, texts):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            lengths_l = l_res['pt_lengths']
            lengths_l.clamp_(2, self.max_length)  # TODO:move to langauge model

            if 'attention' in self.semantic_visual.pe:
                v_attn_input = v_res['attn_scores'].clone().detach()
            else:
                v_attn_input = None

            if self.semantic_visual.mask in ['semantic_dropout_top1', 'semantic_dropout_bottom1', 'semantic_dropout_adv1']:
                l_logits_input = l_res['logits'].clone().detach()
            else:
                l_logits_input = None

            if self.semantic_visual.mask in ['semantic_dropout_adv1']:
                texts_input = texts[0]
            else:
                texts_input = None

            a_res = self.semantic_visual(l_res['feature'], v_res['backbone_feature'], lengths_l=lengths_l, v_attn=v_attn_input, l_logits=l_logits_input, texts=texts_input, training=self.training)

            if 'prev_logits' in a_res.keys():  # predictor == 'transformer_bigram'
                a_prev_res = {'logits': a_res['prev_logits'], 'pt_lengths': a_res['pt_prev_lengths'], 'loss_weight': a_res['loss_weight'],
                              'name': 'alignment'}
                all_a_res.append(a_prev_res)
            elif 'v_logits' in a_res.keys():  # predictor == 'multimodal_v2'
                a_v_res = {'logits': a_res['v_logits'], 'pt_lengths': a_res['pt_v_lengths'], 'loss_weight': a_res['loss_weight'],
                              'name': 'alignment'}
                all_a_res.append(a_v_res)
                a_s_res = {'logits': a_res['s_logits'], 'pt_lengths': a_res['pt_s_lengths'], 'loss_weight': a_res['loss_weight'],
                              'name': 'alignment'}
                all_a_res.append(a_s_res)
            all_a_res.append(a_res)
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            if self.test_bh is None:
                return a_res, all_l_res[-1], v_res
            elif self.test_bh == 'final':
                return a_res, all_l_res[-1], v_res
            elif self.test_bh == 'semantic':
                return all_a_res[-2], all_l_res[-1], v_res
            elif self.test_bh == 'visual':
                return all_a_res[-3], all_l_res[-1], v_res
            else:
                raise NotImplementedError

