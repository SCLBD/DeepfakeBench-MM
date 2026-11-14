import json
import numpy as np
import os
import re
import sys
import torch
import torch.distributed as dist

from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.builder import choose_dataset
from metrics.builder import MetricBuilder
from optimizers.builder import choose_optimizer
from schedulers.builder import choose_scheduler
from tools.init_seed import build_generator, seed_worker
from tools.logger import get_logger
from tools.recorder import Recorder
from tools.registry import DETECTOR


class BasicTrainer:
    """Train a detector with one optimizer and validate configured datasets."""

    def __init__(self, config):
        """Build the model, data loaders, logging, and optimization state.

        Args:
            config: Training configuration, including model/data and runtime settings.
                path_config.log_dir is the prepared run directory; resume is a boolean
                selecting whether to restore the latest full training state there.
        """
        self.config = config

        # Create logger and log directory.
        self.log_dir = self.config['path_config']['log_dir']
        self.logger = get_logger(self.log_dir, rank=self.config['rank'], world_size=self.config['world_size'])
        self.logger.info(f'Save log to {self.log_dir}')
        self.logger.info(f'Config: {json.dumps(self.config, indent=4)}')

        # Prepare model and move model to current CUDA device if available
        self.logger.info('Preparing model...')
        self.model = DETECTOR[self.config['model_name']](self.config['model_config'])
        self.device = torch.device(f'cuda:{self.config["local_rank"]}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            lr = int(self.config['local_rank'])
            torch.cuda.set_device(lr)
            self.model = self.model.to(lr)
        else:
            self.model = self.model.to(self.device)
        if self.config['ddp']:
            local_rank = int(self.config['local_rank'])
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )

        # Prepare training and validation datasets and dataloader.
        self.logger.info('Preparing training and validation data...')
        train_dataset = choose_dataset(
            data_type=self.config['data_type'],
            dataset_list=self.config['train_datasets'],
            data_config=self.config['data_config'],
            aug_config=self.config.get('aug_config'),
            mode='train'
        )
        g = build_generator(self.config['seed'] + self.config['rank'])
        sampler = DistributedSampler(train_dataset) if self.config.get('ddp', False) else None
        self.train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.get('num_workers', 4),
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        self.val_data_loader_dict = {}
        for val_dataset in self.config.get('val_datasets', []):
            dataset = choose_dataset(
                data_type=self.config['data_type'],
                dataset_name=val_dataset,
                data_config=self.config['data_config'],
                mode='val'
            )
            g = build_generator(self.config['seed'] + self.config['rank'])
            sampler = DistributedSampler(dataset) if self.config.get('ddp', False) else None
            self.val_data_loader_dict[val_dataset] = DataLoader(
                dataset=dataset,
                batch_size=self.config['val_batch_size'],
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=self.config.get('num_workers', 4),
                drop_last=False,
                pin_memory=True,
                persistent_workers=True,
                worker_init_fn=seed_worker,
                generator=g
            )

        # Prepare training components.
        self.logger.info('Preparing optimizer and scheduler...')
        self.optimizer = choose_optimizer(self.model.parameters(), self.config['optimizer'])
        scheduler_config = self.config.get('scheduler')
        self.scheduler_interval = scheduler_config.get('interval') if scheduler_config else None
        total_iters = self.config['num_epochs']
        if self.scheduler_interval == 'step':
            total_iters *= len(self.train_data_loader)
        self.scheduler = choose_scheduler(self.optimizer, scheduler_config, total_iters=total_iters)
        if len(self.config.get('val_datasets', [])) != 0 and self.config.get('metric_scoring'):
            self.metric_factory = MetricBuilder(self.config['metric_scoring'], self.config['val_datasets'])

        self.start_epoch = 0
        if self.config.get('resume'):
            self._resume_training(self.log_dir)

    def _resume_training(self, resume_path):
        """Restore the latest epoch training state from a run directory when present."""
        state_dir = os.path.join(resume_path, 'training_states')
        if not os.path.exists(resume_path) or not os.path.exists(state_dir):
            self.logger.info(f'Resume path {resume_path} does not exists. Training from scratch.')
            return

        state_files = [f for f in os.listdir(state_dir) if f.endswith('_full_state.pth')]
        if not state_files:
            self.logger.warning(f"No full_state files found in {state_dir}. Training from scratch.")
            return

        # Select the latest training state file
        state_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        state_file = os.path.join(state_dir, state_files[-1])

        self.logger.info(f'Resuming from {state_file}')
        checkpoint = torch.load(state_file, map_location=self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint.get('scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        if self.config['ddp']:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    def _save_ckpt(self, save_dir, ckpt_name, epoch=None, full_state=False):
        """Save model weights or full epoch training state on rank zero."""
        if self.config['rank'] != 0:
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{ckpt_name}.pth')

        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()

        if full_state:  # Save all training stage at {log_dir}/training_states/epochX.pth to resume training.
            torch.save({
                'epoch': epoch, 'model': model_state, 'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            }, save_path)
            self.logger.info(f'Training state saved: {save_path}')
        else:  # Save only model parameters at {log_dir}/checkpoints/epochX_stepY.pth for testing.
            torch.save(model_state, save_path)
            self.logger.info(f'Checkpoint saved: {save_path}')

    def _train_step(self, data_dict):
        """Update the optimizer for one batch and return its recorded training outputs."""
        predictions = self.model(data_dict)
        if self.config['ddp']:
            losses = self.model.module.get_losses(data_dict, predictions)
        else:
            losses = self.model.get_losses(data_dict, predictions)

        if 'overall' not in losses.keys():
            raise KeyError('detector.get_losses() must return a dict with key "overall" as the optimization loss.')

        self.optimizer.zero_grad()
        losses['overall'].backward()
        self.optimizer.step()

        if self.scheduler is not None and self.scheduler_interval == 'step':
            self.scheduler.step()

        return losses, predictions

    @torch.no_grad()
    def _val_step(self, data_dict):
        """Return inference-mode detector predictions for a validation batch."""
        return self.model(data_dict, inference=True)

    def _val_one_dataset(self, dataset_name, val_dataloader):
        """Collect validation records or losses and update one dataset metric."""
        if self.config['rank'] == 0:
            pbar = tqdm(total=len(val_dataloader), desc=f'[Validating {dataset_name}]', dynamic_ncols=True)
        val_recorder_loss = defaultdict(Recorder)
        records_on_current_gpu = []
        self.model.eval()
        with torch.no_grad():
            for iteration, data_dict in enumerate(val_dataloader):
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.device)

                predictions = self._val_step(data_dict)

                # record loss or predicted probability to be fake
                batch_size = data_dict.get('label', torch.tensor([0])).shape[0]
                if self.config['metric_scoring'] == 'loss':
                    if self.config['ddp']:
                        losses = self.model.module.get_losses(data_dict, predictions)
                    else:
                        losses = self.model.get_losses(data_dict, predictions)

                    for name, value in losses.items():
                        v = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
                        val_recorder_loss[name].update(v, num=batch_size)
                else:
                    for idx in range(batch_size):
                        single_data = {'path': data_dict['path'][idx],
                                       'label': data_dict['label'][idx].detach().cpu().item(),
                                       'prob': predictions['prob'][idx].detach().cpu().item(),
                                       }
                        records_on_current_gpu.append(single_data)

                # update progress bar
                if self.config['rank'] == 0:
                    pbar.update(1)

        # close progress bar
        if self.config['rank'] == 0:
            pbar.close()

        # gather loss or prediction results and calculate metric score
        if self.config['ddp']:
            dist.barrier()

        if self.config['metric_scoring'] == 'loss':
            # Gather losses if ddp
            if self.config['ddp']:
                self.logger.info('Synchronizing loss recorder...')
                for name, value in val_recorder_loss.items():
                    val_recorder_loss[name].sync(self.device)
                    dist.barrier()
                self.logger.info('Synchronizing loss recorder [Done]')
            # Update loss record
            self.metric_factory.update(val_recorder_loss['overall'], dataset_name)
        else:
            # Gather prediction records if ddp
            if self.config['ddp']:
                self.logger.info('Prediction records synchronizing...')
                gathered_records = [None] * self.config['world_size']
                dist.all_gather_object(gathered_records, records_on_current_gpu)
                all_records = []
                for i, record in enumerate(gathered_records):
                    if record is not None:
                        all_records.extend(record)

                dist.barrier()
                self.logger.info('Prediction records synchronized.')
            else:
                all_records = records_on_current_gpu

            # Update the prediction-based metric.
            self.metric_factory.update(all_records, dataset_name)

        # clear loss recorders:
        for name, recorder in val_recorder_loss.items():
            recorder.clear()

    def _val_epoch(self, epoch, iteration):
        """Validate every configured dataset, save improved weights, and resume training mode."""
        self.model.eval()
        for key, value in self.val_data_loader_dict.items():
            self._val_one_dataset(key, value)
            if self.config['ddp']:
                dist.barrier()

        # Only rank zero stores metric records and saves best checkpoints.
        if len(self.val_data_loader_dict) > 0 and self.config['rank'] == 0:
            update_list = self.metric_factory.update_best()
            for update_item in update_list:
                self._save_ckpt(os.path.join(self.log_dir, 'checkpoints'), f'{update_item}_best')

            self.logger.info(f'[Epoch {epoch} Iter {iteration + 1}/{len(self.train_data_loader)}] Validation Result:\n'
                             f'{self.metric_factory.parse_metrics(include_latest=True)}')

        self.model.train()

    def train_epoch(self, epoch):
        """Train one epoch and validate at the configured iteration points.

        Args:
            epoch: Zero-based epoch index used by logs and checkpoint names.
        """
        # find out steps to start validation
        val_every_epoch = self.config.get('val_every_epoch', 1)
        val_per_epoch = self.config.get('val_per_epoch', 1)
        val_steps = np.linspace(
            0, len(self.train_data_loader), min(val_per_epoch, len(self.train_data_loader)) + 1
        ).astype(np.int32).tolist()[1:]
        record_per_epoch = self.config.get('record_per_epoch', 10)
        log_steps = np.linspace(
            0, len(self.train_data_loader), min(record_per_epoch, len(self.train_data_loader)) + 1
        ).astype(np.int32).tolist()[1:]

        if self.config['rank'] == 0:
            pbar = tqdm(total=len(self.train_data_loader), desc=f'[Epoch {epoch}] Training',
                        postfix={'Total Loss': 'N/A'}, dynamic_ncols=True)
        self.model.train()
        train_recorder_loss = defaultdict(Recorder)
        for iteration, data_dict in enumerate(self.train_data_loader):
            # move tensors to CUDA if available
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            losses, predictions = self._train_step(data_dict)

            # record losses
            batch_size = data_dict['label'].shape[0]
            for name, value in losses.items():
                v = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
                train_recorder_loss[name].update(v, num=batch_size)

            # update tqdm on main and record to log file
            if self.config['rank'] == 0:
                pbar.set_postfix({'Total Loss': f'{train_recorder_loss["overall"].average():.6g}'})
                pbar.update(1)
            if (iteration + 1) in log_steps:
                loss_log_str = f'[Epoch {epoch} Iter {iteration + 1}/{len(self.train_data_loader)}] Training Losses:'
                for k, v in train_recorder_loss.items():
                    loss_log_str += f' {k} = {v.average() if v.average() is not None else "N/A"}'
                self.logger.info(loss_log_str)

            # validation
            if (epoch + 1) % val_every_epoch == 0 and (iteration + 1) in val_steps:
                # ensure each process finishes its work before validation starts
                if self.config['ddp']:
                    dist.barrier()

                # start validation
                self._val_epoch(epoch, iteration)

                # Save model weights at this validation point.
                self._save_ckpt(os.path.join(self.log_dir, 'checkpoints'), f'epoch{epoch + 1}_step{iteration + 1}')

        # Advance epoch schedules before run() saves the resumable training state.
        if self.scheduler is not None and self.scheduler_interval == 'epoch' and len(self.train_data_loader) > 0:
            self.scheduler.step()

        # clear loss recorders:
        for name, recorder in train_recorder_loss.items():
            recorder.clear()

        # close progress bar
        if self.config['rank'] == 0:
            pbar.close()

    def run(self):
        """Run the remaining epochs and save resumable state after each epoch."""
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            if self.config['ddp']:
                self.train_data_loader.sampler.set_epoch(epoch)

            self.train_epoch(epoch=epoch)
            if self.config['ddp']:
                dist.barrier()

            self._save_ckpt(os.path.join(self.log_dir, 'training_states'), f'epoch{epoch + 1}_full_state',
                            epoch=epoch, full_state=True)
