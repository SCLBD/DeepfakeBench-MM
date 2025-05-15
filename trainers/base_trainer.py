import pickle

import numpy as np
import torch
import torch.distributed as dist
import os
import sys

# from torch.utils.tensorboard import SummaryWriter

from metrics.utils import get_test_metrics

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from tqdm import tqdm

from utils.recorder import Recorder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, config, model, optimizer, scheduler, metric_scoring, logger, log_dir):
        """
        Initialization of basic trainer
        Args:
            config: [dict] config from YAML file and command args
            model: [nn.Module] model to train and validate
            optimizer: []
            scheduler: []
            metric_scoring: [str] metric name used in validation
            logger: [logger] logger
            log_dir: [str] log root, used for TensorBoard
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.log_dir = log_dir

        self.metric_scoring = metric_scoring
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf') if self.metric_scoring not in ['eer', 'loss'] else float('inf')))
        self.speed_up()  # move model to GPU

    # def get_writer(self, phase, dataset_key, metric_key):
    #     writer_key = f"{phase}-{dataset_key}-{metric_key}"
    #     if writer_key not in self.writers:
    #         # update directory path
    #         writer_path = os.path.join(self.log_dir, phase, dataset_key, metric_key, "metric_board")
    #         os.makedirs(writer_path, exist_ok=True)
    #         # update writers dictionary
    #         self.writers[writer_key] = SummaryWriter(writer_path)
    #     return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config["ddp"]:
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config["local_rank"]], find_unused_parameters=True,
                             output_device=self.config["local_rank"])
            # self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def set_train(self):
        self.model.train()
        self.train = True

    def set_eval(self):
        self.model.eval()
        self.train = False

    # def save_ckpt(self, phase, dataset_key, ckpt_info=None):
    #     save_dir = os.path.join(self.log_dir, phase, dataset_key)
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f"ckpt_best.pth")
    #     if self.config['ddp']:
    #         torch.save(self.model.module.state_dict(), save_path)
    #     else:
    #         torch.save(self.model.state_dict(), save_path)
    #
    #     self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_ckpt(self, save_dir, ckpt_name):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{ckpt_name}.pth')
        if self.config['ddp']:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self, data_dict):
        predictions = self.model(data_dict)
        if self.config["ddp"]:
            losses = self.model.module.get_losses(data_dict, predictions)
        else:
            losses = self.model.get_losses(data_dict, predictions)

        self.optimizer.zero_grad()
        losses["overall"].backward()
        self.optimizer.step()

        return losses, predictions

    def train_epoch(self, epoch, train_data_loader, val_data_loaders=None):
        self.logger.info(f"===> Epoch[{epoch}] start!")
        val_times_per_epoch = self.config.get('val_frequency', 1)
        val_steps = np.linspace(0, len(train_data_loader), val_times_per_epoch + 1).astype(np.int32).tolist()[1:]

        # define training recorder
        train_recorder_loss = defaultdict(Recorder)

        if self.config["local_rank"] == 0:
            pbar = tqdm(total=len(train_data_loader), desc=f"[Epoch {epoch}] Training", postfix={"Total Loss": "N/A"}, dynamic_ncols=True)

        self.set_train()
        for iteration, data_dict in enumerate(train_data_loader):
            batch_size = data_dict["label"].shape[0]
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(device)

            losses, predictions = self.train_step(data_dict)

            # store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value.item(), num=batch_size)

            # update tqdm
            if self.config["local_rank"] == 0:
                pbar.set_postfix({"Total Loss": f'{train_recorder_loss["overall"].average():.6g}'})
                pbar.update(1)

            # run val
            if (iteration + 1) in val_steps:
                current_step = epoch * len(train_data_loader) + iteration + 1
                # write loss info to logger
                if self.config['local_rank'] == 0:
                    self.logger.info(f"[Iter: {current_step}] Training losses: ")
                    for k, v in train_recorder_loss.items():
                        self.logger.info(f"{k} = {v.average() if v.average() is not None else 'N/A'}")

                if (not self.config['ddp']) or (self.config['ddp'] and self.config['local_rank'] == 0):
                    val_best_metric = self.val_epoch(epoch, iteration, val_data_loaders, epoch * len(train_data_loader) + iteration + 1)

                # torch.cuda.empty_cache()

        for name, recorder in train_recorder_loss.items():  # clear loss recorder
            recorder.clear()

        if self.config.get('save_ckpt', False):
            self.save_ckpt(os.path.join(self.log_dir, 'train'), f'epoch{epoch + 1}')
        # save_dir = os.path.join(self.log_dir, 'train')
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"epoch{epoch + 1}.pth")
        # if self.config['ddp']:
        #     torch.save(self.model.module.state_dict(), save_path)
        # else:
        #     torch.save(self.model.state_dict(), save_path)

        return val_best_metric

    def val_one_dataset(self, data_loader):
        # define val recorder
        val_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists = []
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            # get data
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key] is not None and key != 'path':
                    data_dict[key] = data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            # feature_lists += list(predictions['feat'].cpu().detach().numpy()) # Todo:

            # compute all losses for each batch data
            if self.config['ddp']:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)

            # print(type(losses))
            # for key, value in losses.items():
            #     print(type(value), value.device, value.shape)
            #     print(f'{key}:{value}')
                # print()
            # store data by recorder
            for name, value in losses.items():
                val_recorder_loss[name].update(value)

        return val_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt']:
                # self.save_ckpt('val', key, f"{epoch}+{iteration}")
                self.save_ckpt(os.path.join(self.log_dir, 'val', key), 'best')
            self.save_metrics('val', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                # writer = self.get_writer('val', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                # writer.add_scalar(f'val_losses/{k}', v_avg, global_step=step)
                loss_str += f"val-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k == 'dataset_dict':
                continue
            metric_str += f"val-metric, {k}: {v}    "
            # tensorboard-2. metric
            # writer = self.get_writer('val', key, k)
            # writer.add_scalar(f'val_metrics/{k}', v, global_step=step)
        # if 'pred' in metric_one_dataset:
        #     acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
        #     metric_str += f'val-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            # writer.add_scalar(f'val_metrics/acc_real', acc_real, global_step=step)
            # writer.add_scalar(f'val_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)

    def val_epoch(self, epoch, iteration, val_data_loaders, step):
        # set model to eval mode
        self.set_eval()

        # define val recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}
        # val for all val data
        keys = val_data_loaders.keys()
        for key in keys:
            # save the val data_dict    # Todo: ???
            # data_dict = val_data_loaders[key].dataset.data_dict
            # self.save_data_dict('val', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.val_one_dataset(
                val_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps)
                                                  # img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name] += value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            # if type(self.model) is AveragedModel:
            #     metric_str = f"Iter Final for SWA:    "
            #     for k, v in metric_one_dataset.items():
            #         metric_str += f"val-metric, {k}: {v}    "
            #     self.logger.info(metric_str)
            #     continue
            self.save_best(epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset)

        if len(keys) > 0 and self.config.get('save_avg', False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> val Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions


