"""Dataset classes are exported on demand so data-generation tools stay lightweight."""


from importlib import import_module


_classes = {
    'AbstractDataset': 'datasets.abstract_dataset',
    'ImageDataset': 'datasets.image.image_dataset',
    'VideoDataset': 'datasets.video.video_dataset',
    'AudioDataset': 'datasets.audio.audio_dataset',
    'AudioVideoDataset': 'datasets.audio_video.audio_video_dataset',
    'AudioVideoMaskModalityDataset': 'datasets.audio_video.audio_video_mask_modality_dataset',
    'AudioVideoRealDataset': 'datasets.audio_video.audio_video_real_dataset',
    'AudioVideoMisalignedRealDataset': 'datasets.audio_video.audio_video_misaligned_real_dataset',
}
__all__ = list(_classes)


def __getattr__(name):
    """
    Load a public dataset class only when it is requested from this package.
    Args:
        name (str): Public class name listed in __all__.
    Returns:
        dataset_class (type): Requested class. Operators and presenters can be
            imported without first importing PyTorch or the decoding backends.
    """
    if name not in _classes:
        raise AttributeError(f'Module {__name__!r} has no attribute {name!r}.')
    value = getattr(import_module(_classes[name]), name)
    globals()[name] = value
    return value
