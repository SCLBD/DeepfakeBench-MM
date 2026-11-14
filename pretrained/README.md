# Pretrained weights

Download the initialization weights needed for your detector and save them at the
paths below, relative to `pretrained/`. See the [weights README](../weights/README.md)
for the release status of the final detector checkpoints.

| Detector | Path | Source |
| --- | --- | --- |
| AVFF | `audiomae/pretrained.pth` | [AudioMAE AudioSet pretrained checkpoint](https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view) |
| AVFF | `marlin/marlin_vit_base_ytf.full.pt` | [MARLIN ViT Base, full checkpoint](https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin_vit_base_ytf.full.pt) |
| AVH | `avhubert/self_large_vox_433h.pt` | [AV-HuBERT Large + Self-Training, VSR fine-tuned on LRS3-433h](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt) |
| FauForensics | `whisper/whisper_base.pt` | [OpenAI Whisper multilingual Base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt) |
| FauForensics | `CSN/CSN_32x2_R101.pyth` | [PyTorchVideo CSN R101](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/CSN_32x2_R101.pyth) |
| FauForensics | `MEFARG/au_112_12cls2.pth` | [FauForensics AU checkpoint reference](https://github.com/JWang-cs21/FauForensics/blob/main/utils/pretrain.py#L6) |
| FRADE | `jx_vit_base_p16_224-80ecf9dd.pth` | [timm ViT Base/16, 224](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) |
| MCL | `EfficientNet/efficientnet-b7-dcc49843.pth` | [EfficientNet-PyTorch B7](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth) |
| MCL | `Cnn14_mAP=0.431.pth` | [PANNs Cnn14](https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1) |

The exact public download for the FauForensics AU checkpoint has not been
identified. The benchmark copy contains `module.`-prefixed parameters. The
current loader does not strip that prefix, so those parameters are not loaded
during initialization.

AVFF adversarial pretraining accepts a raw backbone state dictionary or a
checkpoint with a `model` or `state_dict` entry. Supervised AVFF rebuilds the
AudioMAE backbone after its initial load. The AVFF method-stage checkpoint
supplies the AudioMAE parameters for this stage. The historical checkpoint path
in the supervised configurations is separate from the current adversarial
pretraining recipe.

Set `pretrained` to the AVFF method-stage checkpoint or `stage1_ckpt` to the
appropriate MCL dataset-specific stage-one teacher. These are benchmark
checkpoints maintained separately from the public initialization files above.
