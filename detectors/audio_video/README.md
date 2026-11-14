# Audio-video detectors

The constructor receives `model_config`. The matching `data_config` controls
video sampling, the audio representation, and the inputs supplied by the dataset.
See the [configuration guide](../../configs/detectors/audio_video/README.md)
and the linked detector's `forward` and `get_losses` methods for its batch and
prediction fields.

| Method | Registered `model_name` | Implementation |
|---|---|---|
| Baseline | `C3DResNet18+SEResNet+MLP` | [C3DResNet18_SEResNet_MLP](c3dresnet18_seresnet_detector.py) |
| AVFF | `AVFF` | [AVFFDetector](avff_detector.py) |
| AVFF adversarial pretraining | `AVFF-Pretrain-Adv` | [AVFFPretrainAdv](avff_detector.py) |
| AVH | `AVH` | [AVHDetector](avh_detector.py) |
| FauForensics | `FauForensics` | [FauForensicsDetector](fauforensics_detector.py) |
| FGMDF | `FGMDF` | [FGMDFDetector](fgmdf_detector.py) |
| FRADE | `FRADE` | [FRADEDetector](frade_detector.py) |
| MCL | `MCL` | [MCLStage2Detector](mcl_detector.py) |
| MCL image-teacher pretraining | `MCL-Pretrain` | [MCLStage1Detector](mcl_detector.py) |
| SpeechForensics | `SpeechForensics` | [SpeechForensicsDetector](speechforensics_detector.py) |

Video batches use `(B, C, T, H, W)`. Audio varies by method. The supplied
configurations use MFCC for Baseline, log-mel features for AVFF, FauForensics
and FRADE, log-filterbanks for AVH and SpeechForensics, and waveforms for FGMDF
and MCL.

- **AVFF** combines AudioMAE and MARLIN. Its adversarial pretraining stage
  uses complementary reconstruction and contrastive losses for the generator,
  and separate audio and video discriminators. The
  [pretraining configuration](../../configs/detectors/audio_video/pretraining/avff_pretrain_adv.yaml)
  defines the two optimization stages. The classification detector can initialize
  from a pretraining checkpoint.
- **AVH** uses frozen AV-HuBERT audio and video features with a trainable
  classification head. **SpeechForensics** uses a frozen AV-HuBERT speech
  checkpoint and scores audio-video alignment. Its `load_checkpoint` method
  restores the saved model and task configuration.
- **FauForensics** combines speech, video and facial-action features.
  **FGMDF** builds a graph from audio and video features. Both use overall,
  audio, and video authenticity labels in their losses.
- **FRADE** combines a vision transformer with audio and cross-modal modules.
  Its loss combines classification, cross-modal contrast, and feature clustering.
- **MCL** uses a video student and audio backbone with a frozen image teacher.
  The teacher is trained separately. Stage-specific inputs and frame supervision
  are documented in the two implementation classes.

See the [shared detector interface](../README.md) for the model contract and
the [pretrained weights guide](../../pretrained/README.md) for required
initialization files.
