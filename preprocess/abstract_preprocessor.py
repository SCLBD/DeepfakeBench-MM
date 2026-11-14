import abc
import json
import os
import sys
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class AbstractPreprocessor:
    def __init__(self, dataset_name, dataset_type, split):
        """
        Initialization method of Preprocessor class.
        Args:
            dataset_name (str): Name of the dataset.
            dataset_type (str): Type of the dataset, e.g., video, image_backbones, audio_backbones, audio_backbones-video, etc.
            split (str): Split of the dataset to be preprocessed, options: ['all', 'train', 'val', 'test'].
        """
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        with open(os.path.join(project_root, 'configs/path.yaml'), 'r') as f:
            self.path = yaml.safe_load(f)

        self.train_data_list, self.val_data_list, self.test_data_list = [], [], []
        if split in ['all', 'train']:
            self.train_data_list = self.collect_data_attributes('train')
            print(f'Totally training {len(self.train_data_list)} found.')
        if split in ['all', 'val']:
            self.val_data_list = self.collect_data_attributes('val')
            print(f'Totally validation {len(self.val_data_list)} data found.')
        if split in ['all', 'test']:
            self.test_data_list = self.collect_data_attributes('test')
            print(f'Totally testing {len(self.test_data_list)} data found.')
        self.data_list = self.train_data_list + self.val_data_list + self.test_data_list

    @abc.abstractmethod
    def preprocess_all_data(self, num_proc):
        pass

    @abc.abstractmethod
    def assemble_data_split(self, split, data_list):
        pass

    def assemble_to_json(self):
        all_data_json = {'train': [], 'val': [], 'test': []}
        all_data_json['train'] = self.assemble_data_split('train', self.train_data_list)
        all_data_json['val'] = self.assemble_data_split('val', self.val_data_list)
        all_data_json['test'] = self.assemble_data_split('test', self.test_data_list)
        os.makedirs(os.path.join(self.path['json_dir'], self.dataset_type), exist_ok=True)
        with open(os.path.join(self.path['json_dir'], self.dataset_type, f'{self.dataset_name}.json'), 'w') as f:
            json.dump(all_data_json, f, indent=4)

    # @abc.abstractmethod
    # def convert_lmdb(self):
    #     pass