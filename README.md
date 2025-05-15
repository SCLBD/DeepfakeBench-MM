# DeepfakeBench-MM

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .10](https://img.shields.io/badge/Release-0.4-brightgreen) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)

<b> Authors: Kangran Zhao*, Yupeng Chen*, Xiaoyu Zhang*, Yize Chen, Weinan Guan, Baicheng Chen, Chengzhe Sun, Soumyya Kanti Datta, Qingshan Liu, Siwei Lyu, Baoyuan Wu†

<div style="text-align:center;">
  <img src="figures/benchmark.png" style="max-width:60%;">
</div>

## Quick Start
### 1. Installation
<a href="#top">[Back to top]</a>
```
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
pip install -r requirements.txt
```
### 2. Data Preprocessing
<a href="#top">[Back to top]</a>

All datasets need to be preprocessed to a unified setting, undergoing audio & video stream separation, audio sampling and video frame rate adjustment, video face alignment & cropping, audio segmentation. Subsequently, a JSON file is assembled using preprocessed audio & video clips with their labels and metadata. Run the command to preprocess one dataset.
```
python preprocess/fakeavceleb_preprocesor.py
```
More dataset can be extended based on our moduler codebase design. More details can be referred to `preprocess/README.md`

### 3. Training
<a href="#top">[Back to top]</a>

Our benchmark provides training scripts supporting various settings, e.g., model, optimizer, epochs, batch size, etc. The following steps are necessary if you would like to construct your own model and train it.
1. Inherit a detector class from `detectors/abstract_detectors.py`. Implement the abstract methods in it. In our benchmark, we split `forward()` into `features()` and `classifier()` to enable potential usage of backbone structures.
2. Implement your specific loss function class inherited from `losses/abstract_loss.py`.
3. Remember to register your backbones (if any), models, and losses to `utils/reigstry.py`.
4. Set up configuration files: (1) `configs/path.yaml`, which includes path settings in this project, e.g., log, dataset, and json directory; (2) 'configs/detectors/${YourModel}.yaml', which includes settings of models, datasets, training, and validation. Refer to `configs/detectors/${Detector}.yaml` for more details.
5. Run the command to start up training
```
# With out DDP:
python train.py --detector_path configs/detectors/${YourModel}.yaml

# With DDP:
python train.sh ${num_GPUs} --detector_path configs/detectors/${YourModel}.yaml

# optional arguments that overrides settings in 4.
--train_datasets: [list] training datassets, which will be concatenated together for training
--val_datsets: [list] validation datasets
--save_ckpt: [bool] argument enabled to save checkpoint every epoch
--use_transcoded: [bool] argument enabled to use the transcoded version of preprocessed data 
--log-dir: [str] customized log directory
```

### 4. Evaluation
<a href="#top">[Back to top]</a>

After training, you may run the evaluation script to evaluate within-dataset and cross-datasets performance. Here is an example:
```
python test.py --detector_path configs/detectors/${YourModel}.yaml --weights_path ${YourWeight}.yaml
```


## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, deepfake detection, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or kangranzhao@link.cuhk.edu.cn. We look forward to collaborating with you in pushing the boundaries of deepfake detection.
