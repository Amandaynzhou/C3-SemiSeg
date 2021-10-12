import datetime
import json

import math
import os
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils import tensorboard

import utils.lr_scheduler
from models.embedding import create_embedding_layer
from utils import helpers
from utils.checkpoint import load_state_dict
from utils.dist_utils import synchronize, simple_group_split, convert_sync_bn, broadcast_value
from utils.logger import setup_logger


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT

    if config[name]['type'] == 'Adam':
        # remove extra params
        try:
            del config[name]['args']['weight_decay']
            del config[name]['args']['momentum']
        except:
            pass
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class BaseTrainer:
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, rank=-1,
                 distributed=False, local_rank=-1, args=None):

        self.model = model
        self.args = args
        # init embedding module if contrastive
        self.contrastive = config['trainer']['contrastive']
        if self.contrastive:
            contrastive_comps = self.init_contrastive_modules(config)
            self.embeddingModule = contrastive_comps['embed']
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.improved = False
        self.rank = rank
        self.local_rank = local_rank
        self.distributed = distributed
        self.to_tensor = transforms.ToTensor()
        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.pretrain_path = cfg_trainer['pretrain_path']
        self.cls_balance_threshold = cfg_trainer['cb_threshold']
        # CHECKPOINTS & TENSOBOARD
        start_time = datetime.datetime.now().strftime('%m-%d_%H')
        _base_name = '{arch}_temp{temperature}_{head}'.format(
            arch=self.config['name'],
            temperature=self.config['contrastive']['temperature'],
            head=args.p_head,
        )
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], self.config['name'], start_time + _base_name)
        if self.rank <= 0:
            helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config-deepv2.json')
        writer_dir = os.path.join(cfg_trainer['log_dir'], self.config['name'], start_time + _base_name)
        self.writer_dir = writer_dir
        if self.rank <= 0:
            with open(config_save_path, 'w') as handle:
                json.dump(self.config, handle, indent=4, sort_keys=True)
            self.writer = tensorboard.SummaryWriter(writer_dir)
            # add logger
        self.logger = setup_logger(self.__class__.__name__, writer_dir, distributed_rank=self.rank)
        self.logger.info(json.dumps(self.config, sort_keys=True, indent=4))
        if self.pretrain_path != '.':
            self._load_pretrain_model()
        # SETTING THE DEVICE
        self.model = self._parallel_model(self.model)
        if self.contrastive:
            self.embeddingModule = self._parallel_model(self.embeddingModule)
        # OPTIMIZER
        self._set_optimizer(config)

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.not_improved_count = 0
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        if resume: self._resume_checkpoint(resume)

    def init_contrastive_modules(self, config):
        embeddingModule = create_embedding_layer(self.args.p_head,
                                                 embed_dim=config['contrastive']['embed_dim'])
        return {'embed': embeddingModule}

    def _parallel_model(self, model):
        if self.rank == -1:
            # not distributed learning
            self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
            model = torch.nn.DataParallel(model, device_ids=availble_gpus)
            model.to(self.device)
        else:
            self.device = self.local_rank
            if self.config["use_synch_bn"]:
                process_group = simple_group_split(dist.get_world_size(), dist.get_rank(),
                                                   self.config['sync_group_nums'])
                convert_sync_bn(model, process_group)
            model.cuda(self.local_rank)
            print(f'local rank{self.local_rank}, ')
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank],
                                                              find_unused_parameters=True)
        return model

    def _reset_monitor(self):
        self.not_improved_count = 0
        self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf

    def _set_optimizer(self, config, ):
        if self.rank <= 0:
            self.logger.info('set optim %s' % self.config['optimizer']['type'])
        if self.config['optimizer']['differential_lr']:
            if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                           torch.nn.parallel.DistributedDataParallel):

                trainable_params = [
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_decoder_params())},
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_backbone_params()),
                     'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p: p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p: p.requires_grad, self.model.get_backbone_params()),
                                     'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = [{'params': filter(lambda p: p.requires_grad, self.model.parameters())}]
        # add contrastive module
        if self.contrastive:
            contrastive_params = [
                {'params': filter(lambda p: p.requires_grad, self.embeddingModule.module.parameters()),
                 'lr': config['optimizer']['args']['lr']}]
            trainable_params = trainable_params + contrastive_params
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(self.optimizer, self.epochs,
                                                                                        len(self.train_loader))

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def train(self):

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_loader.dataset.set_epoch(epoch - 1)
            results = self._train_epoch(epoch)

            if self.rank <= 0:
                # log for train data
                self.logger.info(f'\n         ## Info for train epoch {epoch} ## ')
                lr = self.lr_scheduler.get_last_lr()
                self.logger.info('learning rate = {:.6}'.format(lr[0]))
                for k, v in results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')
            if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)
                # LOGGING INFO
                if self.rank <= 0:
                    self.logger.info(f'\n         ## Info for val epoch {epoch} ## ')
                    for k, v in results.items():
                        self.logger.info(f'         {str(k):15s}: {v}')
            synchronize()
            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)

            if self.rank <= 0:
                try:
                    if self.mnt_mode == 'min':
                        self.improved = (results[self.mnt_metric] < self.mnt_best)
                    else:
                        self.improved = (results[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break
                if self.improved:
                    self.mnt_best = results[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1
                if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                    if self.improved:
                        self._save_checkpoint(epoch, save_best=True)
                        if self.contrastive:
                            self._save_embedding_module(epoch, save_best=True)
            synchronize()
            if self.args.distributed:
                self.not_improved_count = broadcast_value(self.not_improved_count, self.rank, 0)
            if self.not_improved_count > self.early_stoping:
                self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                self.logger.warning('Training Stoped')
                break
            # SAVE CHECKPOINT
            if epoch % self.save_period == 0 and self.rank <= 0:
                self._save_checkpoint(epoch)
                if self.contrastive:
                    self._save_embedding_module(epoch)
            synchronize()

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")
        else:
            filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
            self.logger.info(f'\nSaving a checkpoint: {filename} ...')
            torch.save(state, filename)

    def _save_embedding_module(self, epoch, save_best=False):
        state = {
            'arch': type(self.embeddingModule).__name__,
            'epoch': epoch,
            'state_dict': self.embeddingModule.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_embedding_module.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_embedding_module.pth")
        else:
            filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}-embedding-module.pth')
            self.logger.info(f'\nSaving a checkpoint: {filename} ...')
            torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        if self.rank <= 0:
            self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            if self.rank == 0:
                self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            if self.rank == 0:
                self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.rank == 0:
            self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _load_pretrain_model(self):
        if self.rank <= 0: self.logger.info(f'load pretrain weight from {self.pretrain_path}')
        load_state_dict(self.model, torch.load(self.pretrain_path, map_location=torch.device('cpu'))['state_dict'])

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
