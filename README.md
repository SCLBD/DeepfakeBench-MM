# DeepfakeBench-MM and Mega-MMDF

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .10](https://img.shields.io/badge/Release-0.4-brightgreen) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)

Authors: Kangran Zhao*, Yupeng Chen*, Xiaoyu Zhang*, Yize Chen, Weinan Guan, Baicheng Chen, Chengzhe Sun, Soumyya Kanti Datta, Qingshan Liu, Siwei Lyu, Baoyuan Wu†

(*Equal contribution, †Corresponding author)

<div style="text-align:center;">
  <img src="figures/benchmark.png" style="max-width:80%;">
</div>

---

## 🧠 Overview

Welcome to **DeepfakeBench-MM**, your one-stop solution for multimodal deepfake detection! Key contributions include:

- 💽 **[Mega-MMDF Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J4DVAA)**  
  The largest publicly available **multimodal deepfake detection dataset** to date. To mitigate potential social impact caused by Deepfake data, we require request before accessing this dataset.

- 🧪 **DeepfakeBench-MM Benchmark**  
  A modular and extensible **benchmark codebase** for training and evaluating multimodal deepfake detection methods. Supported Datasets are models are listed below.

|                  | Paper                                                                                                                                                            |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AVTS             | [Hearing and Seeing Abnormality: Self-Supervised Audio-Visual Mutual Learning for Deepfake Detection](https://ieeexplore.ieee.org/document/10095247) ICASSP 2023 |
| MRDF             | [Cross-Modality and Within-Modality Regularization for Audio-Visual Deepfake Detection](https://ieeexplore.ieee.org/document/10447248) ICASSP2024                |
| AVFF             | [AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection](https://arxiv.org/abs/2406.02951v1) CVPR 2024                                                   |
| Vanilla Baseline | -                                                                                                                                                                |
| Ensemble 1       | -                                                                                                                                                                |
| Ensemble 2       | -                                                                                                                                                                |
| Qwen2.5-Omni     | [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) arxiv 2025                                                                                     |
| Video-Llama2     | [VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLM](https://arxiv.org/abs/2406.07476) arxiv 2024                            |

| Dataset           | Real Videos | Fake Videos | Total Videos | Forgery Methods | Original Repository                                                                           |
|-------------------|-------------|-------------|--------------|-----------------|-----------------------------------------------------------------------------------------------|
| FakeAVCeleb_v1.2  | 500         | 21,044      | 21,544       | 4               | [Hyper-link](https://github.com/DASH-Lab/FakeAVCeleb)                                         |
| LAV-DF            | 36,431      | 99,873      | 136,304      | 2               | [Hyper-link](https://github.com/ControlNet/LAV-DF)                                            |
| IDForge_v1        | 80,000      | 170000      | 25,0000      | 6               | [Hyper-link](https://github.com/xyyandxyy/IDForge?tab=readme-ov-file)                         |
| AVDeepfake1M      | 286,721     | 860,039     | 1,146,760    | 3               | [Hyper-link](https://github.com/ControlNet/AV-Deepfake1M)                                     |
| Mega-MMDF         | 100,000     | 1,100,000   | 1,200,000    | 28              | [Hyper-link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J4DVAA) |



---


## ⏳ Quick Start
### 1️⃣  Installation
<a href="#top">[Back to top]</a>
```
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
pip install -r requirements.txt
```
### 2️⃣  Data Preprocessing
<a href="#top">[Back to top]</a>

All datasets must be preprocessed to a unified format, including:

- Audio-video stream separation
- Audio resampling and video frame rate adjustment
- Face alignment and cropping
- Audio segmentation

After preprocessing, a JSON file is generated to organize audio/video clips with their corresponding labels and metadata.

Preprocessed version and corresponding JSON files are in preparation and will be released soon. 🛡️ **Copyright of the above datasets belongs to their original providers.**


Example command:
```
python preprocess/fakeavceleb_preprocesor.py
```
Thanks to our modular design, additional datasets can be integrated with ease. More details can be found in `preprocess/README.md`.

### 3️⃣  Training
<a href="#top">[Back to top]</a>

Our benchmark provides flexible training scripts with support for various configurations, including model architecture, optimizer, batch size, number of epochs, etc.

To train a custom model:

1. **Define your model**:  
   Inherit from `detectors/abstract_detectors.py` and implement required methods. We decouple `forward()` into `features()` and `classifier()` to encourage backbone reuse.

2. **Define a customized loss**:  
   Inherit from `losses/abstract_loss.py`.

3. **Register your components**:  
   Add them into `utils/registry.py`.

4. **Configure your experiment**:  
   - `configs/path.yaml`: paths for logs, datasets, JSON files  
   - `configs/detectors/${YourModel}.yaml`: model, training, validation settings

5. **Run training**:
```
# With out DDP:
python train.py --detector_path configs/detectors/${YourModel}.yaml

# With DDP:
python train.sh ${num_GPUs} --detector_path configs/detectors/${YourModel}.yaml
```
Optional arguments (overriding config settings):

| Argument           | Description                               |
| ------------------ | ----------------------------------------- |
| `--train_datasets` | `[list]` Training datasets to concatenate |
| `--val_datsets`    | `[list]` Validation datasets              |
| `--save_ckpt`      | `[bool]` Save checkpoint after each epoch |
| `--use_transcoded` | `[bool]` Use transcoded preprocessed data |
| `--log-dir`        | `[str]` Custom log directory path         |



### 4️⃣  Evaluation
<a href="#top">[Back to top]</a>

To evaluate a trained model on both in-domain and cross-domain datasets:
```
python test.py --detector_path configs/detectors/${YourModel}.yaml --weights_path ${YourWeight}.yaml
```
This will report performance metrics including accuracy, AUC, and more, depending on the configuration.

## 📝 Reference

For unimodal benchmark, refer to our <a href="https://github.com/SCLBD/DeepfakeBench">DeepfakeBench </a> for further information. To provide more convenient codebase for both unimodal and multimodal Deepfake benchmark, we plan to merge these two benchmark in the future. Thanks for your interest.

If interested, you can read our recent works about deepfake detection, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).
```
@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}

@inproceedings{UCF_YAN_ICCV2023,
 title={Ucf: Uncovering common features for generalizable deepfake detection},
 author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
 booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
 pages={22412--22423},
 year={2023}
}

@inproceedings{LSDA_YAN_CVPR2024,
  title={Transcending forgery specificity with latent space augmentation for generalizable deepfake detection},
  author={Yan, Zhiyuan and Luo, Yuhao and Lyu, Siwei and Liu, Qingshan and Wu, Baoyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{10677972,
  author={Jia, Shan and Lyu, Reilin and Zhao, Kangran and Chen, Yize and Yan, Zhiyuan and Ju, Yan and Hu, Chuanbo and Li, Xin and Wu, Baoyuan and Lyu, Siwei},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={Can ChatGPT Detect DeepFakes? A Study of Using Multimodal Large Language Models for Media Forensics}, 
  year={2024},
  volume={},
  number={},
  pages={4324-4333},
  keywords={Deepfakes;Machine learning algorithms;Forensics;Large language models;Conferences;Training data;Focusing;Deepfake Detection;Multimodal Large Language Models;Media Forensics;GPT4V},
  doi={10.1109/CVPRW63382.2024.00436}
}

@article{chen2024textit,
  title={$$\backslash$textit $\{$X$\}$\^{} 2$-DFD: A framework for e $$\{$X$\}$ $ plainable and e $$\{$X$\}$ $ tendable Deepfake Detection},
  author={Chen, Yize and Yan, Zhiyuan and Lyu, Siwei and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2410.06126},
  year={2024}
}

@article{yan2025orthogonalsubspacedecompositiongeneralizable,
  title={Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection},
  author={hiyuan Yan and Jiangming Wang and Peng Jin and Ke-Yue Zhang and Chengchun Liu and Shen Chen and Taiping Yao and Shouhong Ding and Baoyuan Wu and Li Yuan},
  journal={arXiv preprint arXiv:2411.15633},
  year={2024}
}
```


## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, deepfake detection, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or kangranzhao@link.cuhk.edu.cn. We look forward to collaborating with you in pushing the boundaries of deepfake detection.
