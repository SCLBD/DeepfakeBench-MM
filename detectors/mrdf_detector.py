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
from metrics.base_metrics_class import calculate_metrics_for_train

from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from losses import LOSSFUNC
import detectors.utils.avhubert.hubert as hubert
import detectors.utils.avhubert.hubert_pretraining as hubert_pretraining


@DETECTOR.register_module(module_name='MDS')
class MDSDetector(AbstractDetector):
    def __init__(self, config, margin_contrast=0.0, margin_audio=0.0, margin_visual=0.0, weight_decay=0.0001, learning_rate=0.0002, distributed=False):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)

        self.embed = 768
        self.dropout = 0.1

        self.feature_extractor_audio_hubert = self.model.feature_extractor_audio
        self.feature_extractor_video_hubert = self.model.feature_extractor_video

        self.project_audio = nn.Sequential(LayerNorm(self.embed), nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))

        self.project_video = nn.Sequential(LayerNorm(self.embed), nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))

        self.project_hubert = nn.Sequential(self.model.layer_norm, self.model.post_extract_proj,
                                            self.model.dropout_input)

        self.fusion_encoder_hubert = self.model.encoder

        self.final_proj_audio = self.model.final_proj
        self.final_proj_video = self.model.final_proj
        self.final_proj_hubert = self.model.final_proj

        self.video_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        self.audio_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        # #
        self.mm_classifier = nn.Sequential(nn.Linear(self.embed, self.embed), nn.ReLU(inplace=True),
                                           nn.Linear(self.embed, 2))

        self.loss_func = self.build_loss(config)


    def build_backbone(self, config):
        return hubert.AVHubertModel(cfg=hubert.AVHubertConfig,
                                          task_cfg=hubert_pretraining.AVHubertPretrainingConfig,
                                          dictionaries=hubert_pretraining.AVHubertPretrainingTask)

    def build_loss(self, config):
        # prepare the loss function
        contrastive_loss = LOSSFUNC[config['loss_func'][0]]
        video_cross_entropy = LOSSFUNC[config['loss_func'][1]]
        audio_cross_entropy = LOSSFUNC[config['loss_func'][2]]

        return {'L1': contrastive_loss(), 'L2': video_cross_entropy(), 'L3': audio_cross_entropy()}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        # pred = pred_dict['cls']
        prob = pred_dict['prob']
        # auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), prob.detach())

        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def features(self, data_dict: dict) -> torch.tensor:
        a_features = self.feature_extractor_audio_hubert(data_dict['audio']).transpose(1, 2)
        v_features = self.feature_extractor_video_hubert(data_dict['video']).transpose(1, 2)
        av_features = torch.cat([a_features, v_features], dim=2)
        return {'video': v_features, 'audio': a_features, 'fused': av_features}

    def classifier(self, features: torch.tensor) -> torch.tensor:
        a_cross_embeds = features['audio'].mean(1)
        v_cross_embeds = features['video'].mean(1)

        a_features = self.project_audio(features['audio'])
        v_features = self.project_video(features['video'])
        av_features = self.project_hubert(features['fused'])

        a_embeds = a_features.mean(1)
        v_embeds = v_features.mean(1)

        a_embeds = self.audio_classifier(a_embeds)
        v_embeds = self.video_classifier(v_embeds)

        av_features, _ = self.fusion_encoder_hubert(av_features, padding_mask=None) # .
        m_logits = self.mm_classifier(av_features[:, 0, :])

        return {'m_logit': m_logits, 'a_logit': a_embeds, 'v_logit': v_embeds}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        loss1 = self.loss_func['L1'](pred_dict['mds'], data_dict['label'])
        loss2 = self.loss_func['L2'](pred_dict['cls']['video'], data_dict['label'])
        loss3 = self.loss_func['L2'](pred_dict['cls']['audio'], data_dict['label'])

        return {'overall': loss1 + loss2 + loss3, 'L1': loss1, 'L2': loss2, 'L3': loss3}


    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)

        # we use this in most cases!!!
        # # get the probability of the pred
        prob = torch.softmax(pred['m_logit'], dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            'cls': pred['m_logit'],
            'prob': prob,
            # 'feat': features,
            }
        return pred_dict

