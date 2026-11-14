#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Run from the project root; uses vanilla Mega-MMDF checkpoints.

python test.py --detector_path configs/detectors/audio_video/misaligned_test/avff_misaligned_real.yaml --weight_path weights/AVFF_train_on_Mega-MMDF.pth --log_dir logs/misaligned_test/AVFF_train_on_Mega-MMDF_shift_audio_1 --test_datasets Mega-MMDF

python test.py --detector_path configs/detectors/audio_video/misaligned_test/avh_misaligned_real.yaml --weight_path weights/AVH_train_on_Mega-MMDF.pth --log_dir logs/misaligned_test/AVH_train_on_Mega-MMDF_shift_audio_1 --test_datasets Mega-MMDF

python test.py --detector_path configs/detectors/audio_video/misaligned_test/baseline_misaligned_real.yaml --weight_path weights/Baseline_train_on_Mega-MMDF.pth --log_dir logs/misaligned_test/Baseline_train_on_Mega-MMDF_shift_audio_1 --test_datasets Mega-MMDF

python test.py --detector_path configs/detectors/audio_video/misaligned_test/fauforensics_misaligned_real.yaml --weight_path weights/FauForensics_train_on_Mega-MMDF.pth --log_dir logs/misaligned_test/FauForensics_train_on_Mega-MMDF_shift_audio_1 --test_datasets Mega-MMDF

python test.py --detector_path configs/detectors/audio_video/misaligned_test/fgmdf_misaligned_real.yaml --weight_path weights/FGMDF_train_on_Mega-MMDF.pth --log_dir logs/misaligned_test/FGMDF_train_on_Mega-MMDF_shift_audio_1 --test_datasets Mega-MMDF

python test.py --detector_path configs/detectors/audio_video/misaligned_test/frade_misaligned_real.yaml --weight_path weights/FRADE_train_on_Mega-MMDF.pth --log_dir logs/misaligned_test/FRADE_train_on_Mega-MMDF_shift_audio_1 --test_datasets Mega-MMDF

python test.py --detector_path configs/detectors/audio_video/misaligned_test/mcl_misaligned_real.yaml --weight_path weights/MCL_train_on_Mega-MMDF.pth --log_dir logs/misaligned_test/MCL_train_on_Mega-MMDF_shift_audio_1 --test_datasets Mega-MMDF
