#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ "$#" -eq 0 ]; then
  echo "Usage: bash commands/robustness/generate_robustness_data.sh DATASET [DATASET ...]" >&2
  exit 2
fi

python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_h264 --param 23 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_h264 --param 30 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_h264 --param 37 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_downscale --param 0.6666666666666666 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_downscale --param 0.4444444444444444 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_downscale --param 0.3333333333333333 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_frame_drop --param 0.05 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_frame_drop --param 0.1 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_frame_drop --param 0.2 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type audio_aac --param 64k --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type audio_aac --param 32k --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type audio_aac --param 16k --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type audio_dropout --param 0.05 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type audio_dropout --param 0.1 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type audio_dropout --param 0.2 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_delay --param 0.08 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_delay --param 0.16 --seed 0 --datasets "$@"
python datasets/utils/perturbations/run_perturbation_on_audio_video_datasets.py --type video_delay --param 0.32 --seed 0 --datasets "$@"
