# Data Preprocess
This section gives details of how to run preprocessing code and customize your datasets.
### 1. Two import config files: 
Path configurations: `configs/path.yaml`. You may modify these settings according to your storage paths.
```yaml
original_dir: xxx       # directory where your put original dataset folder
transcode_dir: xxx      # directory where the transcoded version will be saved
preprocessed_dir: xxx   # directory where the preprocessed version will be saved
json_dir: xxx           # directory where the assembled JSON file will be stored
```
Preprocess configurations: `preprocess/config.yaml`.
```yaml
transcode: false          # set true to transcode video clip into frames.npz format, otherwise, into frames.mp4 format
num_proc: 128             # the number of processors to be used, leave it empty to use all preprocessors.

# video and audio output settings
length: 1.0               # clip length (s)
FPS: 25                   # video FPS
sample_rate: 16000        # audio sample rate
clip_amount: 1            # the number of clips cropped and segmented from a video
```
Besides, in `ArgumentParser()` used in every preprocessor script. You may use `--split` to declare which data split will be preprocessed, avoiding redundant preprocessing on training split of test datasets.

### 2. Customize your dataset.
This directory is organized as follows:

(1) `utils/tools.py`: preprocessing utilities

(2) `utils/*.dat` and `utils/*.npy`: face alignment models and template

(3) `base_preprocessor.py`: high-level methods using preprocessing utilities

(4) `${dataset_name}_preprocessor.py`: exact preprocessor class

You need to implement parsing of your dataset annotations in your own `${dataset_name}_preprocessor.py`

