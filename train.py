import argparse
import copy
import datetime
import numpy as np
import os
import random
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import warnings
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# from metrics.choose_metric import choose_metric
from datasets.audio_video_dataset import AudioVideoDataset
from detectors import DETECTOR
from metrics.utils import parse_metric
from optimizers.choose_optimizer import choose_optimizer
from schedulers.choose_scheduler import choose_scheduler
from trainers.base_trainer import Trainer
from utils.logger import create_logger, RankFilter

def arg_parse():
    """
    Args Parser from command line, overriding settings in config YAML files.
    """
    parser = argparse.ArgumentParser(description="training settings, overriding settings in config files")
    parser.add_argument("--detector_path", type=str, required=True, help="path to detector config files")
    parser.add_argument("--ddp", action="store_true", default=False, help="set true when using DDP")
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument("--train_datasets", nargs="+", type=str,
                        help="command line to override training datasets selection in config file")
    parser.add_argument("--val_datasets", nargs="+", type=str,
                        help="command line to override validation datasets selection in config file")
    parser.add_argument("--save_ckpt", action="store_true", help="enable to save ckeckpoint for every epoch")
    parser.add_argument("--use_transcoded", action="store_true", help="enable to use transcoded version data loading")

    parser.add_argument("--log_dir", type=str, help="command line to override log root in config file")

    return parser.parse_args()


def init_seed(config):
    """
    Initialize random seed
    Args:
        config: [dict] config containing "seed"
    """
    if config.get("seed", None) is None:
        config["seed"] = random.randint(1, 10000)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    if config["cuda"]:
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_training_data(config):
    """
    Prepare Dataset and Dataloader for training.
    Args:
        config: [dict] config
    Returns:
        train_data_loader: [torch.data.utils.DataLoader] train data loader concatenating training dataset(s)
    """
    if config.get("dataset_type", None) is None:
        train_dataset = AudioVideoDataset(config, mode="train")
    else:
        raise NotImplementedError(f"Dataset {config['dataset_type']} is not implemented")

    sampler = DistributedSampler(train_dataset) if config["ddp"] else None
    return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["train_batch_size"], shuffle=True,
                                       num_workers=config.get("num_workers", 4), drop_last=True, sampler=sampler)


def prepare_validation_data(config):
    """
    Prepare Dataset and Dataloader for validation.
    Args:
        config: [dict] config
    Returns:
        val_data_loaders: [dict] a dict of validation data loader, dataset_name -> val_data_loader
    """
    def get_val_data_loader(dataset_name):
        tmp_config = config.copy()
        tmp_config["val_dataset"] = dataset_name
        if config.get("dataset_type", None) is None:
            val_dataset = AudioVideoDataset(config=tmp_config, mode="val")
        else:
            raise NotImplementedError(f"Dataset {config['dataset_type']} is not implemented")

        sampler = DistributedSampler(train_dataset) if tmp_config["ddp"] else None
        return torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config["val_batch_size"], shuffle=False,
                                           num_workers=config.get("num_workers", 4), drop_last=False, sampler=sampler)

    val_data_loader_dict = {}
    for val_dataset_name in config["val_dataset"]:
        val_data_loader_dict[val_dataset_name] = get_val_data_loader(val_dataset_name)

    return val_data_loader_dict


def choose_metric(config):
    metric_scoring = config.get('metric_scoring', 'auc')
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))

    return metric_scoring


def main():
    # load config files and args from command line
    args = arg_parse()
    with open(os.path.join(os.path.dirname(__file__), "configs", "path.yaml"), "r") as f:
        config = yaml.safe_load(f)
    with open(args.detector_path, "r") as f:
        config.update(yaml.safe_load(f))
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # create logger
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(config["log_dir"], "train", f"{config['model_name']}_{time_now}")
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, "training.log"))
    logger.info(f"Save log to {log_dir}")

    # init seed and DDP, if used
    init_seed(config)
    torch.cuda.set_device(args.local_rank)
    if config["ddp"]:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    # create model
    model = DETECTOR[config["model_name"]](config)
    if config.get("init_weight", None) is not None:
        ckpt = torch.load(config["init_weight"], map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        logger.info(f"Missing keys (parameters that model needs but does not exist in checkpoint): {missing_keys}")
        logger.info(f"Unexpected keys (parameters that model does not need but exist in checkpoint): {unexpected_keys}")

    # prepare dataset, optimizer, scheduler (if any), validation metric
    train_data_loader = prepare_training_data(config)
    val_data_loader_dict = prepare_validation_data(config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(optimizer, config)
    metric_func = choose_metric(config)

    # trainer and training
    trainer = Trainer(config, model, optimizer, scheduler, metric_func, logger, log_dir)
    for epoch in range(config["num_epochs"]):
        best_metric = trainer.train_epoch(epoch=epoch, train_data_loader=train_data_loader,
                                          val_data_loaders=val_data_loader_dict)
        if scheduler is not None:
            scheduler.step()

        if best_metric is not None:
            logger.info(f"Epoch [{epoch}] ends with {config['metric_scoring']}: {parse_metric(best_metric)}")

    logger.info(f"Finish training with best validation metric {config['metric_scoring']}: {parse_metric(best_metric)}")


if __name__ == "__main__":
    main()
