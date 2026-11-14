"""Train a configured detector with the basic or alternating training loop."""

import argparse
import datetime
import os
import shutil
import sys
import torch
import torch.distributed as dist
import warnings
import yaml

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

from tools.init_seed import set_seed
from trainers.alternative_trainer import AlternativeTrainer
from trainers.basic_trainer import BasicTrainer


def arg_parse():
    """
    Parse the detector configuration, logging, resume and distributed options.
    Returns:
        args (argparse.Namespace): Command-line training settings.
    """
    parser = argparse.ArgumentParser(description='Deepfake Detection Training Command Line Settings')
    parser.add_argument('--detector_path', required=True, type=str, help='Path to model config file')
    parser.add_argument('--log_dir', type=str, default=None, help='Exact training run directory; defaults to a timestamped run')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume the latest full state in the selected run directory')

    # Arguments below needn't to claim.
    parser.add_argument('--ddp', action='store_true', help='Use DDP training')
    parser.add_argument('--local_rank', default=0, type=int, help='local_rank for distributed training')

    return parser.parse_args()


def prepare_training_directory(config, log_dir=None, resume=False):
    """Select and prepare the actual run directory before constructing a trainer.

    Args:
        config (dict): Model settings, path_config, and initialized rank/ddp settings.
            path_config.log_dir is the parent root for automatically named runs.
        log_dir (str | None): Exact run directory. None creates train/model_timestamp
            below the configured logging root.
        resume (bool): Preserve existing files when True. Otherwise remove the
            selected run directory and create an empty directory for a fresh run.

    Returns:
        str: Resolved absolute run directory, also stored in path_config.log_dir.
            config.resume stores the boolean flag. Only rank zero prepares files;
            distributed workers receive the same directory or preparation error.
    """
    distributed = config.get('ddp', False)
    result = [None, None]
    if not distributed or config.get('rank', 0) == 0:
        try:
            if log_dir is None:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                log_dir = os.path.join(config['path_config']['log_dir'], 'train',
                                       f'{config["model_name"]}_{timestamp}')
            requested_path = os.path.abspath(os.path.expanduser(log_dir))
            if os.path.islink(requested_path):
                raise ValueError('The training run directory must not be a symbolic link.')
            run_dir = os.path.realpath(requested_path)
            if os.path.dirname(run_dir) == run_dir:
                raise ValueError('A filesystem root cannot be used as the training run directory.')

            # Catch configuration mistakes that could replace source code or input data.
            protected = [(project_root, False), (os.path.expanduser('~'), False)]
            for name in ('configs', 'preprocess', 'datasets', 'backbones', 'detectors', 'losses', 'metrics',
                         'optimizers', 'schedulers', 'tools', 'trainers', 'analysis', 'tests', 'pretrained',
                         'weights', 'references', '.git'):
                protected.append((os.path.join(project_root, name), True))
            for name in ('original_dir', 'preprocessed_dir', 'json_dir'):
                if config['path_config'].get(name):
                    protected.append((os.path.join(project_root, config['path_config'][name]), True))
            for path, include_children in protected:
                path = os.path.realpath(path)
                try:
                    shared = os.path.normcase(os.path.commonpath([run_dir, path]))
                except ValueError:  # Different Windows drives do not overlap.
                    continue
                if shared == os.path.normcase(run_dir) or include_children and shared == os.path.normcase(path):
                    raise ValueError(f'Training run directory {run_dir!r} overlaps protected path {path!r}.')

            if os.path.exists(run_dir) and not os.path.isdir(run_dir):
                raise ValueError(f'Training run directory is not a directory: {run_dir!r}.')
            if not resume and os.path.isdir(run_dir):
                shutil.rmtree(run_dir)
            os.makedirs(run_dir, exist_ok=True)
            result[0] = run_dir
        except Exception as error:
            if not distributed:
                raise
            result[1] = f'{type(error).__name__}: {error}'
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(f'Unable to prepare the training run directory: {result[1]}')
    config['path_config']['log_dir'] = result[0]
    config['resume'] = resume
    return result[0]


def main():
    """Load configuration, initialize runtime state and run the selected trainer."""
    args = arg_parse()

    with open(os.path.join(os.path.dirname(__file__), 'configs/path.yaml'), 'r') as f:
        path_config = yaml.safe_load(f)

    with open(args.detector_path, 'r') as f:
        detector_config = yaml.safe_load(f)
        if args.seed is not None:
            detector_config['seed'] = args.seed

    config = {**detector_config, **{'path_config': path_config}}
    if config['train_type'] not in ('basic', 'alternative'):
        raise ValueError(f'Unknown training type: {config["train_type"]!r}.')

    # torchrun sets these env vars. Use them if present for robustness.
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank or 0))
    if args.ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    config['ddp'] = args.ddp
    config['local_rank'] = local_rank
    config['rank'] = rank
    config['world_size'] = world_size

    # set random seed
    set_seed(config['seed'] + config['rank'], deterministic=config.get('deterministic', True))
    prepare_training_directory(config, log_dir=args.log_dir, resume=args.resume)
    if config['train_type'] == 'basic':
        trainer = BasicTrainer(config)
    elif config['train_type'] == 'alternative':
        trainer = AlternativeTrainer(config)

    trainer.run()

if __name__ == '__main__':
    main()
