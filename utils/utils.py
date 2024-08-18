import json
import logging

import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
# TODO 设置日志输出
def setup_logger(logger_name='dbtext', log_file_path=None):
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s')
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger
# TODO dataloader
def get_dataloader(config,train=True):
    name = config.data_loader.name
    # TODO 加载数据集
    if name == "Mnist":
        from utils.data_loaders import MnistDataLoader
        DatasetIter = MnistDataLoader
    # TODO 加载迭代器
    train_iter = DatasetIter(config.data_loader.path,
                             config.data_loader.batch_size,
                             config.data_loader.shuffle,
                             config.data_loader.validation_split,
                             config.data_loader.num_workers,train
                             )
    test_iter = train_iter.split_validation()
    return train_iter,test_iter
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
# TODO Metric class
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
    # TODO clear
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
    #TODO update
    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]
    # TODO avg
    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
