# @Time : 2024-08-10 22:06
# @Author : wyj
# @File : train.py
# @Describe :
import os
from trainer import Trainer
import hydra
import torch
from torch import nn
from utils import setup_logger
from utils import get_dataloader
import model.model as Mnist
import model.loss as loss
import model.metric as metric
def main(cfg):
    # TODO 配置日志目录
    logger = setup_logger(
        os.path.join(cfg.meta.root_dir, "train.log"))
    # TODO 设置device
    device = cfg.meta.device
    logger.info(f"device:{device}")
    # TODO dataloader
    train_iter,test_iter = get_dataloader(config=cfg)
    # TODO 加载模型
    model = getattr(Mnist,cfg.model.name)()
    logger.info(f"model:{model}")
    # TODO 将模型加载到设备上
    model = model.to(cfg.meta.device)
    # TODO 多GPU
    if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    # TODO loss
    criterion = getattr(loss, cfg.loss.name)
    logger.info(f"loss:{cfg.loss.name}")
    # TODO metric
    metrics = [getattr(metric, cfg.metric[met]) for met in cfg.metric]
    # TODO optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim,cfg.optimizer.type)(trainable_params)
    logger.info(f"optimizer:{cfg.optimizer.type}")
    # TODO lr_scheduler
    lr_scheduler = getattr(torch.optim.lr_scheduler,cfg.lr_scheduler.type)(optimizer,cfg.lr_scheduler.step_size)
    logger.info(f"lr_scheduler:{cfg.lr_scheduler.type}")
    # TODO Trainer
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=cfg,
                      device=device,
                      data_loader=train_iter,
                      valid_data_loader=test_iter,
                      lr_scheduler=lr_scheduler)
    #TODO train
    logger.info("start train")
    trainer.train()

@hydra.main(config_path="./config.yaml")
def train(cfg):
    main(cfg)

if __name__ == "__main__":
    train()
