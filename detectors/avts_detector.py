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
from einops import rearrange, repeat

from backbones.c3dr18 import C3DResNet18
from backbones.seresnet18 import SEResNet18
from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from losses import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

@DETECTOR.register_module(module_name='AVTS-stage2')
class AVTSStage2Detector(AbstractDetector):
    def __init__(self, config, img_in_dim=1, last_dim=512, frames_per_clip=5, num_classes=1, fake_classes=1, mode='VA', relu_type = 'prelu', predict_label=False, aud_feat='mfcc'):
        super().__init__()
        self.img_in_dim = img_in_dim
        self.last_dim = last_dim
        self.frames_per_clip = frames_per_clip
        self.mode = mode
        self.aud_feat = aud_feat
        self.predict_label = predict_label
        self.relu_type = relu_type

        self.video_backbone, self.audio_backbone = self.build_backbone({})

        self.temporal_classifier = TemporalTransformer(
            frames_per_clip=self.frames_per_clip,
            num_classes=num_classes,
            dim=last_dim*2,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        audio_backbone = SEResNet18(layers=[2, 2, 2, 2], num_filters=[self.last_dim//8, self.last_dim//4, self.last_dim//2, self.last_dim])
        video_backbone = C3DResNet18(in_dim=self.img_in_dim, last_dim=self.last_dim, relu_type=self.relu_type)
        return video_backbone, audio_backbone

    def build_loss(self, config):
        # prepare the loss function
        cross_entropy = LOSSFUNC[config['loss_func'][0]]

        return {'cross_entropy': cross_entropy()}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        # pred = pred_dict['cls']
        prob = pred_dict['prob']
        # auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), prob.detach())

        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def features(self, data_dict: dict) -> torch.tensor:
        with torch.no_grad():
            video_feature = self.video_backbone(data_dict['video'])
            audio_feature = self.audio_backbone(data_dict['audio'])
        return {'video': video_feature, 'audio': audio_feature}

    def classifier(self, features) -> torch.tensor:
        feats = torch.cat([features['video'], features['audio']], -1)
        logits, cls_feature = self.temporal_classifier(feats)
        # print(cls_feature.shape)
        return logits

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        ce_loss = self.loss_func['cross_entropy'](pred_dict['cls'], data_dict['label'])
        return {'overall': ce_loss, 'CE': ce_loss}


    def forward(self, data_dict: dict, inference=False) -> dict:
        data_dict['video'] = data_dict['video'][:, :, :5, :, :]
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=data_dict['video'].device).reshape(1, 3, 1, 1, 1)
        data_dict['video'] = (data_dict['video'] * weights).sum(dim=1, keepdim=True)
        data_dict['audio'] = data_dict['audio'][:, :20, :].unsqueeze(1)
        # print(data_dict['audio'].shape, data_dict['video'].shape)
        features = self.features(data_dict)
        pred = self.classifier(features)
        pred = torch.cat([-pred, pred], dim=1)
        prob = torch.softmax(pred, dim=1)[:, 1]

        pred_dict = {
            'cls': pred,
            'prob': prob,
            # 'feat': features,
        }

        return pred_dict

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, frames_per_clip, num_classes=1, dim=256, depth=6, heads=12, mlp_dim=1024, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, frames_per_clip + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        feature = x

        return self.mlp_head(x), feature

