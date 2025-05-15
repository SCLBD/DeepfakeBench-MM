import torch
import torch.distributed as dist
import os
import sys

from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets.audio_video_dataset import AudioVideoDataset
from optimizers.choose_optimizer import choose_optimizer
from schedulers.choose_scheduler import choose_scheduler
from utils.recorder import Recorder


class SingleStageTrainer:
    def __init__(self, model, config, path_config, logger):
        self.config = config
        self.path_config = path_config
        self.logger = logger
        self.device = torch.device(f"cuda:{config.get('local_rank', 0)}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if config.get("world_size", 1) > 1:
            self.model = DDP(self.model, device_ids=[config["local_rank"]])

        # Prepare training dataset
        if config.get("dataset_type", None) is None:
            train_dataset = AudioVideoDataset({**path_config, **config}, mode="train")
        else:
            raise NotImplementedError(f'Dataset {config["dataset_type"]} is not implemented')
        train_sampler = DistributedSampler(train_dataset) if config.get("world_size", 1) > 1 else None
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['train_batch_size'],
                                           num_workers=config.get("num_workers", 4), shuffle=(train_sampler is None),
                                           sampler=train_sampler)

        # Prepare validation datasets
        self.val_dataloader_dict = {}
        for dataset_name in config['val_dataset']:
            if config.get("dataset_type", None) is None:
                val_dataset = AudioVideoDataset({**path_config, **config, **{"val_dataset": dataset_name}}, mode="val")
            else:
                raise NotImplementedError(f'Dataset {config["dataset_type"]} is not implemented')
            val_sampler = DistributedSampler(val_dataset) if config.get("world_size", 1) > 1 else None
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['val_batch_size'],
                                        num_workers=config.get("num_workers", 4), shuffle=(val_sampler is None),
                                        sampler=val_sampler)
            self.val_dataloader_dict[dataset_name] = val_dataloader

        self.optimizer = choose_optimizer(self.model.parameters(), config["optimizer"])
        self.scheduler = choose_scheduler(self.optimizer, config["scheduler"]) \
            if config.get("scheduler", None) is not None else None

        # Todo: TensorBoard
        if config.get("val_metric", "loss") in ["acc", "auc", "ap"]:
            self.best_metric = 0
        elif config.get("val_metric", "loss") in ["eer", "loss"]:
            self.best_metric = float("inf")
        else:
            raise NotImplementedError('Metric initialization value is not defined.')

        # Training variables
        self.current_epoch = 0

    def train_step(self, data_dict):
        """
        Training code for one step
        Args:
            data_dict: [dict] data dict loaded from DataLoader
        Returns:
            losses: [dict] loss dict, e.g. 'loss1': 0.91, 'loss2': 1.52, 'overall': 2.43
            predictions: [dict] model output, referring to model forward() return value
        """
        predictions = self.model(data_dict)
        if self.config.get('world_size', 1) > 1:
            losses = self.model.module.get_losses(data_dict, predictions)
        else:
            losses = self.model.get_losses(data_dict, predictions)

        self.optimizer.zero_grad()
        losses['overall'].backward()
        self.optimizer.step()

        return losses, predictions

    def train_epoch(self):
        """
        Training code for one epoch
        """
        train_loss_recorder_dict = defaultdict(Recorder)
        if self.config["local_rank"] == 0:
            pbar = tqdm(total=len(self.train_dataloader), desc=f"[Epoch {self.current_epoch}] Training",
                        postfix={"Total Loss": "N/A"}, dynamic_ncols=True)

        self.model.train()
        for batch_idx, data_dict in enumerate(self.train_dataloader):
            batch_size = data_dict["label"].shape[0]

            # put data onto GPU if available
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            # train one step
            losses, prediction = self.train_step(data_dict)

            # record loss values
            for loss_name, loss_value in losses.item():
                train_loss_recorder_dict[loss_name].update(loss_value.item(), num=batch_size)

            if self.config["local_rank"] == 0:
                pbar.set_postfix({"Total Loss": train_loss_recorder_dict["overall"].average()})
                pbar.update(1)

        if self.config["local_rank"] == 0:
            pbar.close()

        for loss_name in train_loss_recorder_dict.keys():
            self.logger.info(f"{loss_name} = {train_loss_recorder_dict[loss_name].average():.06f}")
            train_loss_recorder_dict[loss_name].clear()

    @torch.no_grad()
    def val_step(self, data_dict):
        """
        Validation code for one step
        Args:
            data_dict: [dict] data dict loaded from DataLoader
        Returns:
            losses: [dict] loss dict, e.g. 'loss1': 0.91, 'loss2': 1.52, 'overall': 2.43
            predictions: [dict] model output, referring to model forward() return value
        """
        predictions = self.model(data_dict, inference=True)
        if self.config.get('world_size', 1) > 1:
            losses = self.model.module.get_losses(data_dict, predictions)
        else:
            losses = self.model.get_losses(data_dict, predictions)

        return losses, predictions


    @torch.no_grad()
    def validate_one_dataset(self, val_dataset_name):
        """
        Validation code for one epoch on one dataset
        Returns:
            total_loss: [float] total average loss
            prediction_list: [list] list of probability predicted
            label_list: [list] list of numeric labels
        """
        val_loss_recorder_dict = defaultdict(Recorder)
        if self.config["local_rank"] == 0:
            pbar = tqdm(total=len(self.val_dataloader_dict[val_dataset_name]),
                        desc=f"[Epoch {self.current_epoch}] Validating",
                        dynamic_ncols=True)

        prediction_list = []
        label_list = []
        for batch_idx, data_dict in enumerate(self.train_dataloader):
            batch_size = data_dict["label"].shape[0]

            # put data onto GPU if available
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.device)

            # validation one step
            losses, prediction = self.train_step(data_dict)

            # record loss values
            for loss_name, loss_value in losses.item():
                val_loss_recorder_dict[loss_name].update(loss_value.item(), num=batch_size)

            # record prediction and label
            prediction_list += list((prediction["prob"].cpu().detach().numpy()))
            label_list += list((prediction["label"].cpu().detach().numpy()))

            if self.config["local_rank"] == 0:
                pbar.update(1)

        if self.config["local_rank"] == 0:
            pbar.close()

        # In DDP case, synchronize loss, prediction_list, label_list
        if self.config.get("world_size", 1) > 1:
            for loss_name in val_loss_recorder_dict.keys():
                val_loss_recorder_dict[loss_name].sync(self.device)
                self.logger.info(f"{loss_name} = {val_loss_recorder_dict[loss_name].average():.06f}")

            prediction_list = self.sync_numeric_list(prediction_list)
            label_list = self.sync_numeric_list(label_list)

        total_loss = val_loss_recorder_dict['overall'].average()
        for loss_name in val_loss_recorder_dict.keys():
            val_loss_recorder_dict[loss_name].clear()

        return total_loss, prediction_list, label_list

    @torch.no_grad()
    def validate_on_all_datasets(self):
        validation_result = {}
        for dataset_name in self.val_dataloader_dict.keys():
            total_loss, prediction_list, label_list = self.validate_one_dataset(dataset_name)
            validation_result[dataset_name] = {"total loss": total_loss,
                                               "predictions": prediction_list,
                                               "label_list": label_list}
            # Todo: 这里写算metric的

            # Todo: 这里写metric比较的

            # Todo: 这里写ckpt保存的（每个都保存和保存最好的）


    def sync_numeric_list(self, local_list):
        """
        Synchronize numeric list (float list or int list) in DDP case
        Args:
            local_list: [list] list on local device
        Returns:
            global_list: [list] synchronized list, storing according to order or preprocesses.
        """
        assert self.config.get("world_size", 1) > 1
        # convert to Tensor
        local_tensor = torch.tensor(local_list, device=self.device)
        local_size = torch.tensor([len(local_list)], device=self.device)

        # synchronize length of local lists
        size_list = [torch.empty_like(local_size) for _ in range(self.config["world_size"])]
        dist.all_gather(size_list, local_size)
        max_size = max(s.item() for s in size_list)

        # pad torch to max length
        padded_tensor = torch.zeros(max_size, dtype=local_tensor.dtype, device=self.device)
        padded_tensor[:len(local_list)] = local_tensor
        tensor_list = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]

        # collect data
        dist.all_gather(tensor_list, padded_tensor)

        # merge based on rank list
        global_list = []
        for tensor, size in zip(tensor_list, size_list):
            global_list.extend(tensor[:size].cpu().tolist())

        return global_list



