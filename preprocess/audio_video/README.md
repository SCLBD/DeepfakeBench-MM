# Audio-video preprocessing

Prepare aligned audiovisual media and metadata for the audio-video dataset loaders.

## Supported datasets

| Dataset | Preprocessor |
|---|---|
| FakeAVCeleb | [fakeavceleb_preprocessor.py](fakeavceleb_preprocessor.py) |
| LAV-DF | [lav_df_preprocessor.py](lav_df_preprocessor.py) |
| PolyGlotFake | [polyglotfake_preprocessor.py](polyglotfake_preprocessor.py) |
| IDForge | [idforge_preprocessor.py](idforge_preprocessor.py) |
| Mega-MMDF | [mega_mmdf_preprocessor.py](mega_mmdf_preprocessor.py) |
| TalkingHeadBench | [talkingheadbench_preprocessor.py](talkingheadbench_preprocessor.py) |
| DigiFakeAV | [digifakeav_preprocessor.py](digifakeav_preprocessor.py) |
| MMDF | [mmdf_preprocessor.py](mmdf_preprocessor.py) |

## Quick start

**1. Prepare the tools and resources.** Install the project dependencies, make FFmpeg available on `PATH`, and download the two dlib models and mean-face template following [dlib_tools/README.md](../dlib_tools/README.md). Put all three files in `preprocess/dlib_tools/`.

**2. Arrange the raw data.** Set `original_dir`, `preprocessed_dir` and `json_dir` in [configs/path.yaml](../../configs/path.yaml). Under `original_dir`, keep the layouts expected by the selected adapter:

```text
original_dir/
  FakeAVCeleb_v1.2/
    <category>/<race>/<gender>/<identity>/*.mp4
  LAV-DF/
    metadata.json
    {train,dev,test}/*.mp4
  PolyGlotFake/
    json_file/real_json_file/all_real_video.json
    json_file/fake_Json_file/all_fake_video.json
    real/<language>/*.mp4
    fake/to_<language>/*.mp4
  IDForge_v1/main/
    {train,val,test}.csv
    <category>/.../*.mp4
  Mega-MMDF/
    {train,val,test}.json
    attributes.json
    {RARV,RAFV,FARV,FAFV}/pipeline*/.../*.mp4
  TalkingHeadBench/
    fake/<method>/{train,val,test}/*.mp4
    fake/additional_dataset/<method>/*.mp4
    real/real_dataset_split.json
    audio/ff++/*.wav
  DigiFakeAV/
    <category_directory>/*.mp4
  MMDF/
    {train,val,test}/{real,fake}/<model_id>/*.mp4
```

FakeAVCeleb, PolyGlotFake and DigiFakeAV read the bundled [CSV](fakeavceleb_splits/README.md), [PolyGlotFake JSON](polyglotfake_splits/README.md) and [DigiFakeAV JSON](digifakeav_splits/README.md) split files directly. They do not need to be copied into `original_dir`. For DigiFakeAV, preserve the directory names in its lists, such as `FVFA_<method>_...`, `FVRA_<method>_...` and `real`; each raw MP4 must contain both streams.

TalkingHeadBench also reads real source videos from `original_dir/CelebV-HQ/35666/` and `original_dir/FaceForensics++/original_sequences/youtube/raw/videos/`.

**3. Run an adapter from the repository root.** For example:

```bash
# --split: all (default), train, val or test; selects the splits to process.
# --num_proc: number of worker processes; default: 1.
# --split_dir: optional alternative split directory, supported by FakeAVCeleb,
#              PolyGlotFake and DigiFakeAV; default: the adapter's bundled directory.
python preprocess/audio_video/fakeavceleb_preprocessor.py --split all --num_proc 8
```

Use the corresponding script in the table for another dataset. `--split all` produces the complete manifest; split-specific assembly overwrites the JSON and leaves uncollected splits empty.

## Outputs

For a raw clip, processed media and auxiliary files have this layout:

```text
preprocessed_dir/audio-video/
  <dataset>/.../
    clip.wav
    clip.done
    clip/
      2_102.mp4
      2_102.npy
```

The MP4 contains aligned 256 x 256 frames at 25 FPS. The WAV is 16 kHz mono audio on the normalized source timeline. The NPY stores aligned landmarks with shape `(T, 68, 2)`; the MP4/NPY stem records the retained source-frame interval `[start, end)`.

Assembly writes `json_dir/audio-video/<dataset_name>.json`, for example:

```json
{
  "train": [
    {
      "label": 0,
      "audio_label": 0,
      "video_label": 0,
      "original_path": "FakeAVCeleb_v1.2/.../clip.mp4",
      "path": "FakeAVCeleb_v1.2/.../clip/2_102.mp4",
      "method": "real"
    }
  ],
  "val": [],
  "test": []
}
```

`original_path` is relative to `original_dir`, and `path` is relative to `preprocessed_dir/audio-video/`. Authentic labels are 0 and forged labels are 1. Dataset adapters may add temporal forgery annotations or separate raw audio/video paths.
