"""Availability checks for paired files in the standardized audio-video layout."""


from pathlib import Path


def validate_audio_video_files(records, data_root, with_landmark=False):
    """
    Require every referenced paired file before using an alternate data version.
    Args:
        records (list): Prepared metadata or indexed records with relative video paths.
        data_root (str or Path): Selected prepared data directory.
        with_landmark (bool): Also require the corresponding NPY annotations.
    Returns:
        num_files (int): Number of unique required files checked. Missing files raise
            FileNotFoundError; this function never generates data or falls back to clean files.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f'Data directory does not exist: {data_root}. Generate the requested data version first.')
    required = set()
    for record in records:
        video = record['path']
        required.add(data_root / video)
        required.add(data_root / (str(Path(video).parent) + '.wav'))
        if with_landmark:
            required.add(data_root / video.replace('mp4', 'npy'))
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        examples = '\n'.join(missing[:5])
        raise FileNotFoundError(
            f'{len(missing)} required files are missing under {data_root}.\n{examples}\n'
            'Generate the requested data version before evaluation; existing complete files can be skipped.'
        )
    return len(required)


if __name__ == '__main__':
    # Run after preparing the corresponding local example files.
    unit_test_params = {
        'records': [{'path': 'ExampleDataset/clip/0_25.mp4'}],
        'data_root': Path('data/preprocessed/audio-video'),
        'with_landmark': False,
    }
    print(f'Checked {validate_audio_video_files(**unit_test_params)} unique files.')
