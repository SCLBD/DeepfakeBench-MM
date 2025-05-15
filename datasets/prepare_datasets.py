import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data.distributed import DistributedSampler

from datasets.audio_video_dataset import AudioVideoDataset


def prepare_training_data(config):
    """
    Prepare Dataset and Dataloader for training.
    Args:
        config: [dict] dictionary for one training stage
    """
    if config.get('dataset_type', None) is None:
        train_dataset = AudioVideoDataset(config, mode='train')
    else:
        raise NotImplementedError(f'Dataset {config["dataset_type"]} is not implemented')

    if config['ddp']:
        sampler = DistributedSampler(train_dataset)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=config['train_batch_size'],
                                                        num_workers=config.get('num_workers', 4),
                                                        # collate_fn=train_dataset.collate_fn,
                                                        sampler=sampler)
    else:
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=config['train_batch_size'],
                                                        shuffle=True,
                                                        num_workers=config.get('num_workers', 4),
                                                        # collate_fn=train_dataset.collate_fn,
                                                        )

    return train_data_loader


def prepare_val_or_test_data(config):
    """
    Prepare Dataset and Dataloader for validation.
    Args:
        config: [dict] dictionary for one training stage
    """
    def get_val_data_loader(dataset_name):
        tmp_config = config.copy()
        tmp_config['val_dataset'] = dataset_name
        if config.get('dataset_type', None) is None:
            val_dataset = AudioVideoDataset(config=tmp_config, mode='val')
        else:
            raise NotImplementedError(f'Dataset {config["dataset_type"]} is not implemented')

        test_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                       batch_size=config['val_batch_size'],
                                                       shuffle=False,
                                                       num_workers=config.get('num_workers', 4),
                                                       # collate_fn=val_dataset.collate_fn,
                                                       drop_last=False)

        return test_data_loader

    # Different from training dataset, which is to concatenate all data from different dataset, val dataset needs to be
    # seperated with all others. One Dataset and Dataloader is instantiated for one dataset.
    val_data_loaders = {}
    for val_dataset_name in config['val_dataset']:
        val_data_loaders[val_dataset_name] = get_val_data_loader(val_dataset_name)

    return val_data_loaders