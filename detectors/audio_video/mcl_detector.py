import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from panns_inference.inference import AudioTagging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backbones.audio_backbones.Cnn14_backbone import Cnn14
from backbones.video_backbones.ViViT_backbone import ViViT
from detectors.abstract_detector import AbstractDetector
from tools.registry import DETECTOR, LOSSFUNC


@DETECTOR.register_module(module_name='MCL-Pretrain')
class MCLStage1Detector(AbstractDetector):
    """Image-teacher pretraining supervised by the selected video frame."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])
        self.train_stage_idx = 0

    def build_backbone(self, config):
        """Construct the EfficientNet image teacher with its local initialization."""
        backbone = imageTeacher(224, 1000)
        return backbone

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict, inference=False):
        """Return image logits/features and the selected frame index.

        Training chooses a random frame; inference chooses the middle frame.
        """
        B, C, T, H, W = data_dict['video'].shape
        if inference:
            frame_idx = T // 2
        else:
            frame_idx = random.randint(0, T - 1)
        image = data_dict['video'][:, :, frame_idx]
        # image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

        logit, feature = self.backbone(image)
        return {'logit': logit, 'feature': feature, 'frame_idx': frame_idx}

    def classifier(self, features):
        """Unused interface hook; the image teacher produces logits in ``features``."""
        pass

    def get_losses(self, data_dict, pred_dict):
        """Return CE using ``frame_label`` at the selected ``frame_idx``."""
        # Use the per-frame label at the exact frame picked in `features`, so the supervision matches the image actually fed to the backbone.
        frame_idx = pred_dict['frame_idx']
        frame_label = data_dict['frame_label'][:, frame_idx].long()
        loss = self.loss_func_dict['CE'](pred_dict['cls'], frame_label)
        return {'overall': loss}

    def forward(self, data_dict, inference=False):
        """Return image ``cls``, fake-class ``prob`` and selected ``frame_idx``."""
        features = self.features(data_dict, inference)
        prob = torch.softmax(features['logit'], dim=1)[:, 1]
        pred_dict = {'cls': features['logit'], 'prob': prob, 'frame_idx': features['frame_idx']}
        return pred_dict


@DETECTOR.register_module(module_name='MCL')
class MCLStage2Detector(AbstractDetector):
    """MCL video-student training with image-teacher and audio features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])
        self.train_stage_idx = 0
        self.head = nn.Sequential(
            nn.Linear(1000, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )

    def build_backbone(self, config):
        """Construct MCL using ``stage1_ckpt`` and ``audio_ckpt`` initializations."""
        audio_ckpt = config.get('audio_ckpt')
        teacher_ckpt = config.get('stage1_ckpt')
        backbone = AVNET(teacher_ckpt, audio_ckpt)
        return backbone

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict, inference):
        """Return student, teacher and composed features plus selected frame index.

        Expects 30-frame video and waveform audio. The image teacher uses a random
        frame in training and the middle frame in inference.
        """
        B, _, T = data_dict['video'].shape[:3]
        if inference:
            frame_idx = T // 2
        else:
            frame_idx = random.randint(0, T - 1)
        image = data_dict['video'][:, :, frame_idx]
        # image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

        v_hat, ifeature, afeature, vfeature, iv, av, i_hat, a_hat, av_hat, noise_feat, iv_hat = self.backbone(
            data_dict['video'], image, data_dict['audio'], inference
        )
        return {
            'noise_feat': noise_feat,
            'i_feat': ifeature,
            'a_feat': afeature,
            'v_feat': vfeature,
            'iv_feat': iv,
            'av_feat': av,
            'a_hat': a_hat,
            'v_hat': v_hat,
            'av_hat': av_hat,
            'i_hat': i_hat,
            'iv_hat': iv_hat,
            'frame_idx': frame_idx,
        }

    def classifier(self, features):
        """Map noise-filtered fusion features to two-class logits."""
        pred = self.head(features)
        return pred

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Combine unimodal, contrastive, consistency and fusion objectives.

        Uses overall, audio, video and selected-frame authenticity labels.
        """
        frame_idx = pred_dict['frame_idx']
        frame_label = data_dict['frame_label'][:, frame_idx].long()

        loss_v = self.loss_func_dict['L_v'](pred_dict['v_hat'], data_dict['video_label'])
        loss_v_weight = self.loss_weight_dict['L_v']
        loss_a = self.loss_func_dict['L_a'](pred_dict['a_hat'], data_dict['audio_label'])
        loss_a_weight = self.loss_weight_dict['L_a']

        la = 0.5 * self.loss_func_dict['L_nce'](
            pred_dict['v_feat'], pred_dict['a_feat'], data_dict['video_label'], data_dict['audio_label']
        ) + 0.5 * self.loss_func_dict['L_nce'](
            pred_dict['v_feat'], pred_dict['av_feat'], data_dict['video_label'], data_dict['label']
        )

        # NOTE: ``iv_feat`` is a pure visual composition (frame + video), so both sides of the contrastive loss should
        # use the visual label. Using the overall audio-visual ``label`` would mis-supervise RVFA samples where the visual
        # modality is real but the audio is fake (paper's assumption l_I = l_V does not hold on partial-fake datasets).
        li = 0.5 * self.loss_func_dict['L_nce'](
            pred_dict['v_feat'], pred_dict['i_feat'], data_dict['video_label'], frame_label
        ) + 0.5 * self.loss_func_dict['L_nce'](
            pred_dict['v_feat'], pred_dict['iv_feat'], data_dict['video_label'], data_dict['video_label']
        )

        # Paper Eq.(5)(6): JSD is computed on 2-class PREDICTION distributions, not raw 1000-d feature vectors. ``v_hat``
        # and ``iv_hat`` are both produced by the shared ``videoStudent.mlp_head`` (see ``AVNET.forward``).
        loss_jsd = self.loss_func_dict['L_jsd'](pred_dict['v_hat'], pred_dict['iv_hat'])

        loss_dis = self.loss_func_dict['L_av'](pred_dict['av_hat'], data_dict['label'])
        loss_nfa = self.loss_func_dict['L_av'](pred_dict['cls'], data_dict['label'])

        loss = (loss_v_weight * loss_v + loss_a_weight * loss_a + la + li + loss_jsd) + loss_dis + loss_nfa

        return {
            'overall': loss,
            'L_v': loss_v,
            'L_a': loss_a,
            'L_nce': la + li,
            'L_jsd': loss_jsd,
            'L_dis': loss_dis,
            'L_nfa': loss_nfa,
        }

    def forward(self, data_dict: dict, inference=False) -> dict:
        """Return fusion ``cls``/``prob``, frame index and auxiliary feature/logit pairs."""
        features_dict = self.features(data_dict, inference)
        noise_feat = features_dict['noise_feat']
        pred = self.classifier(noise_feat)
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            'cls': pred,
            'prob': prob,
            'frame_idx': features_dict['frame_idx'],
            'a_hat': features_dict['a_hat'],
            'a_feat': features_dict['a_feat'],
            'v_hat': features_dict['v_hat'],
            'v_feat': features_dict['v_feat'],
            'av_hat': features_dict['av_hat'],
            'av_feat': features_dict['av_feat'],
            'iv_hat': features_dict['iv_hat'],
            'iv_feat': features_dict['iv_feat'],
            'i_hat': features_dict['i_hat'],
            'i_feat': features_dict['i_feat'],
        }
        return pred_dict


class imageTeacher(nn.Module):
    def __init__(self, img_size, num_class):
        super(imageTeacher, self).__init__()
        self.backbone = EfficientNet.from_pretrained(
            'efficientnet-b7', weights_path='pretrained/EfficientNet/efficientnet-b7-dcc49843.pth', num_classes=2
        )
        self.linear = nn.Linear(2560, num_class)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.backbone._dropout(x)
        f = self.linear(x)
        x = self.backbone._fc(x)

        return x, f


class attBlock(nn.Module):
    def __init__(self, input_dim, normalization_sign=False):
        super().__init__()
        half_input_dim = int(input_dim / 2)
        self.mlp = nn.Linear(input_dim, half_input_dim)
        self.normalization = normalization_sign

    def forward(self, f1, f2):
        """Compose an auxiliary modality feature into the video feature.

        Implements paper Eq.(4):  f_{I,V} = f_V + H_{I,V}(f_I, f_V)
        where ``f1`` is the auxiliary modality (e.g. frame feature f_I) and
        ``f2`` is the video feature f_V. The residual is ADDED TO f2.
        """
        if self.normalization:
            f1_n = F.normalize(f1, dim=1, p=2.0, eps=1e-10)
            f2_n = F.normalize(f2, dim=1, p=2.0, eps=1e-10)
            residual = torch.cat((f1_n, f2_n), 1)
        else:
            residual = torch.cat((f1, f2), 1)
        residual = self.mlp(residual)  # H_{I,V}(f1, f2)
        feature = f2 + residual  # paper Eq.(4): f_V + H(...)
        return feature


class avBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_A, f_V):
        audio_attention = 1 + self.sigmoid(f_A)
        visual_attention = 1 + self.sigmoid(f_V)

        modulated_by_audio = audio_attention * f_V
        modulated_by_visual = visual_attention * f_A

        f_fused = modulated_by_audio + modulated_by_visual

        return f_fused


class AVNET(nn.Module):
    def __init__(self, teacher_ckpt, audio_ckpt):
        super(AVNET, self).__init__()
        self.featurelen = 1000
        self.imgsize = 224
        self.backbone = "vivit"
        # ``num_frames`` MUST match ``data_config.num_frames`` in the YAML,
        # otherwise ``pos_embedding`` shape broadcasting will fail in ViViT.
        self.videoStudent = ViViT(image_size=224, patch_size=16, num_classes=2, num_frames=30, dim=self.featurelen)
        self.imageTeacher = imageTeacher(self.imgsize, self.featurelen)
        self.audioTeacher = audioTeacher(527, self.featurelen, True)

        checkpoint = torch.load(teacher_ckpt, map_location='cpu')
        self.imageTeacher.load_state_dict({k.replace('backbone.', '', 1): v for k, v in checkpoint.items()})
        print("Successed load image teacher!")

        # Freeze the ENTIRE image teacher (backbone + linear + cls_head) in
        # Stage 2, matching the paper's "teacher network is frozen" statement.
        for p in self.imageTeacher.parameters():
            p.requires_grad = False
        self.imageTeacher.eval()

        self.audio_tagger = Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527
        )

        checkpoint = torch.load(audio_ckpt)
        self.audio_tagger.load_state_dict(checkpoint['model'])
        print("Successed load audio checkpoint!")

        self.iv = attBlock(self.featurelen * 2, normalization_sign=True)
        self.av = avBlock()

        self.NFA = NFA(self.featurelen)

        self.dis_av = nn.Linear(self.featurelen, 2)

    def train(self, mode=True):
        """Override to keep the frozen image teacher in ``eval`` mode even when
        the outer model / DDP wrapper calls ``.train()``. Otherwise BatchNorm
        running stats and Dropout would drift during Stage 2 training.
        """
        super().train(mode)
        self.imageTeacher.eval()
        return self

    def forward(self, video, image, audio, inference):

        # Teacher is frozen in Stage 2 -- run its forward under no_grad to
        # save memory / compute (no autograd graph is built for the teacher).
        with torch.no_grad():
            i_hat, ifeature = self.imageTeacher(image)
        clipwise_output = self.audio_tagger(audio)['clipwise_output']
        a_hat, afeature = self.audioTeacher(clipwise_output)
        v_hat, vfeature = self.videoStudent(video)

        afeature = afeature.squeeze()
        if len(afeature.shape) == 1:
            afeature = afeature.unsqueeze(0)

        iv = self.iv(ifeature, vfeature)
        av = self.av(afeature, vfeature)

        # Paper Eq.(5)(6): JSD is computed in the PREDICTION space between
        # ``P_V`` (video prediction) and ``P_{I,V}`` (frame-video prediction).
        # We reuse the video student's classification head ``mlp_head`` so
        # that both predictions live in the same output distribution space.
        iv_hat = self.videoStudent.mlp_head(iv)

        av_hat = self.dis_av(av)

        # av作为NFA block的输入 做最后的预测
        noise_feat = self.NFA(av, inference)

        return v_hat, ifeature, afeature, vfeature, iv, av, i_hat, a_hat, av_hat, noise_feat, iv_hat


class NFA(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=2):
        """
        初始化模型

        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            num_classes (int): 分类类别数 (默认为2: 真假分类)
        """
        super(NFA, self).__init__()

        self.noise_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.softplus = nn.Softplus()

    def apply_noise(self, f, lambda1, lambda2):
        """
        应用噪声扰动: f' = α ⊙ f + β
        其中 α ~ N(1, λ1), β ~ N(0, λ2)
        """
        batch_size, feature_dim = f.shape

        # 生成乘性噪声 α ~ N(1, λ1*I)
        # λ1是方差，标准差是sqrt(λ1)
        alpha = 1 + torch.randn_like(f) * torch.sqrt(lambda1.unsqueeze(1))

        # 生成加性噪声 β ~ N(0, λ2*I)
        beta = torch.randn_like(f) * torch.sqrt(lambda2.unsqueeze(1))

        f_perturbed = alpha * f + beta

        return f_perturbed

    def forward(self, f, inference=False):
        """
        前向传播

        参数:
            f (torch.Tensor): 输入特征, 形状为 (batch_size, input_dim)

        返回:
            tuple: (logits, lambda1, lambda2, f_perturbed)
                - logits: 分类器输出, 形状为 (batch_size, num_classes)
                - lambda1: 乘性噪声方差
                - lambda2: 加性噪声方差
                - f_perturbed: 扰动后的特征
        """
        batch_size = f.size(0)

        params = self.noise_mlp(f)

        # 确保参数为正数 (使用softplus确保>0)
        lambda_params = self.softplus(params)
        lambda1 = lambda_params[:, 0]
        lambda2 = lambda_params[:, 1]

        # 在训练模式下应用噪声，推理时可选   不过论文好像没有说推理时用不用噪声
        if not inference:
            f_perturbed = self.apply_noise(f, lambda1, lambda2)
        else:
            f_perturbed = f

        return f_perturbed


class audioTeacher(nn.Module):
    def __init__(self, audiolen, out_len, predict):
        super(audioTeacher, self).__init__()
        self.linear = nn.Linear(audiolen, out_len)
        self.predict = predict

        self.linear_2 = nn.Linear(out_len, 2)

    def forward(self, x):
        f = self.linear(x)
        return self.linear_2(f), f
