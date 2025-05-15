# Functions in the Class are summarized as:
# 1. __init__: Initialization
# 2. build_backbone: Backbone-building
# 3. build_loss: Loss-function-building
# 4. features: Feature-extraction
# 5. classifier: Classification
# 6. get_losses: Loss-computation
# 7. forward: Forward-propagation

# Reference:
# @inproceedings{10.1145/3394171.3413700,
#     author = {Chugh, Komal and Gupta, Parul and Dhall, Abhinav and Subramanian, Ramanathan},
#     title = {Not made for each other- Audio-Visual Dissonance-based Deepfake Detection and Localization},
#     year = {2020},
#     isbn = {9781450379885},
#     publisher = {Association for Computing Machinery},
#     address = {New York, NY, USA},
#     url = {https://doi.org/10.1145/3394171.3413700},
#     doi = {10.1145/3394171.3413700},
#     abstract = {We propose detection of deepfake videos based on the dissimilarity between the audio and visual modalities, termed as the Modality Dissonance Score (MDS). We hypothesize that manipulation of either modality will lead to dis-harmony between the two modalities, e.g., loss of lip-sync, unnatural facial and lip movements, etc. MDS is computed as the mean aggregate of dissimilarity scores between audio and visual segments in a video. Discriminative features are learnt for the audio and visual channels in a chunk-wise manner, employing the cross-entropy loss for individual modalities, and a contrastive loss that models inter-modality similarity. Extensive experiments on the DFDC and DeepFake-TIMIT Datasets show that our approach outperforms the state-of-the-art by up to 7\%. We also demonstrate temporal forgery localization, and show how our technique identifies the manipulated video segments.},
#     booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
#     pages = {439–447},
#     numpages = {9},
#     keywords = {neural networks, modality dissonance, deepfake detection and localization, contrastive loss},
#     location = {Seattle, WA, USA},
#     series = {MM '20}
# }

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base_detector import AbstractDetector
from utils.registry import DETECTOR


@DETECTOR.register_module(module_name='FRADE')
class FRADEDetector(AbstractDetector):
    def __init__(self, config,):
        super().__init__()
        self.config = config
        self.video_backbone, self.audio_backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.train_stage_idx = 0

        # From https://github.com/abhinavdhall/deepfake/blob/main/ACM_MM_2020/model.py
        self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_lip.weight.data.fill_(1)
        self.final_bn_lip.bias.data.zero_()
        self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        initialize_weights(self.final_fc_lip)

        self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_aud.weight.data.fill_(1)
        self.final_bn_aud.bias.data.zero_()
        self.final_fc_aud = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        initialize_weights(self.final_fc_aud)

    def build_backbone(self, config):
        audio_backbone = Audio_RNN()
        video_backbone = ResNet2d3d_full(block=[BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d],
                                         layers=[2, 2, 2, 2], track_running_stats=False)
        return video_backbone, audio_backbone

    def build_loss(self, config):
        # prepare the loss function
        contrastive_loss = LOSSFUNC[config['loss_func'][0]]
        video_cross_entropy = LOSSFUNC[config['loss_func'][1]]
        audio_cross_entropy = LOSSFUNC[config['loss_func'][2]]

        return {'L1': contrastive_loss(), 'L2': video_cross_entropy(), 'L3': audio_cross_entropy()}

    def features(self, data_dict: dict) -> torch.tensor:
        video_feature = self.video_backbone(data_dict['video'])
        audio_feature = self.audio_backbone(data_dict['audio'])
        return {'video': video_feature, 'audio': audio_feature}

    def classifier(self, features: torch.tensor) -> torch.tensor:
        video_output = self.final_bn_lip(features['video'])
        video_output = self.final_fc_lip(video_output)
        audio_output = self.final_bn_aud(features['audio'])
        audio_output = self.final_fc_aud(audio_output)

        return {'video': video_output, 'audio': audio_output}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        loss1 = self.loss_func['L1'](pred_dict['mds'], data_dict['label'])
        loss2 = self.loss_func['L2'](pred_dict['cls']['video'], data_dict['label'])
        loss3 = self.loss_func['L2'](pred_dict['cls']['audio'], data_dict['label'])

        return {'overall': loss1 + loss2 + loss3, 'L1': loss1, 'L2': loss2, 'L3': loss3}

    # def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
    #     label = data_dict['label']
    #     # pred = pred_dict['cls']
    #     prob = pred_dict['prob']
    #     # auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
    #     auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), prob.detach())
    #
    #     metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
    #     return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # print(data_dict['video'].shape, data_dict['audio'].shape)
        features = self.features(data_dict)
        # print(features['video'].shape, features['audio'].shape)
        pred = self.classifier(features)

        mds_score = torch.norm(features['video'] - features['audio'], p=2, dim=1, keepdim=True)
        prob = (mds_score > 0.6).float()
        # we use this in most cases!!!
        # # get the probability of the pred
        # prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            'cls': pred,
            'prob': prob,
            # 'feat': features,
            'mds': mds_score}
        return pred_dict
