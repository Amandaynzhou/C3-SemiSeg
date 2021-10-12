import argparse
import json
import pdb
import os
import torch
import torch.distributed as dist
import warnings
from torch.backends import cudnn

import dataloaders
import models
from trainer import create_trainer
from utils import losses
from utils.dist_utils import print_dist_information
from utils.misc import set_seed


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    if 'ddp' in config.keys():
        config[name]['args'].update(config['ddp'])
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config):
    set_seed(args.seed)
    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    model.initialize(args, -1, -1, )
    print(f'\n{model}\n')
    # LOSS
    loss = getattr(losses, 'MaskCrossEntropyLoss2d')(ignore_index=config['ignore_index'])
    # TRAINING
    trainer = create_trainer(args.trainer,
                             model=model,
                             loss=loss,
                             resume=args.resume,
                             config=config,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             args=args)
    trainer.train()


def main_worker(config, args):
    world_size = args.world_size
    rank = args.rank = int(os.environ['LOCAL_RANK'])
    gpu = args.gpu = rank
    print('Use GPU(local rank): {} for training, rank {}'.format(gpu, rank))
    dist.init_process_group(backend='nccl')
    if rank == 0:
        print_dist_information()
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    # add ddp setting
    loader_args = {'distributed': True, 'world_size': world_size, 'rank': rank}
    config['train_loader']['args'].update(loader_args)
    config['val_loader']['args'].update(loader_args)
    config['unlabel_loader']['args'].update(loader_args)
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    print('val loader', len(val_loader))
    print('rank{}, init dataloader with {} samples'.format(rank, len(train_loader)))
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    model.initialize(args, rank, gpu)
    loss = getattr(losses, 'MaskCrossEntropyLoss2d')(ignore_index=config['ignore_index'])
    trainer = create_trainer(args.trainer,
                             model=model,
                             loss=loss,
                             resume=args.resume,
                             config=config,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             rank=rank, distributed=True, local_rank=gpu,  args=args)
    trainer.train()


def cloud_modify_config(config):
    config['train_loader']['args']['data_dir'] = config['train_loader']['args']['data_dir']. \
        replace('../', '/cache/')
    config['val_loader']['args']['data_dir'] = config['val_loader']['args']['data_dir']. \
        replace('../', '/cache/')
    try:
        config['unlabel_loader']['args']['data_dir'] = config['unlabel_loader']['args']['data_dir']. \
            replace('../', '/cache/')
    except:
        pass
    config['arch']['args']['cloud'] = True
    return config


def server_modify_config(config):
    config['train_loader']['args']['data_dir'] = '/home/adas/Semantic_data/cs_data'
    config['val_loader']['args']['data_dir'] = '/home/adas/Semantic_data/cs_data'
    try:
        config['label_gen_loader']['args']['data_dir'] = '/home/adas/Semantic_data/cs_data'
    except:
        pass
    try:
        config['unlabel_loader']['args']['data_dir'] = '/home/adas/Semantic_data/cs_data'
    except:
        pass
    return config


def modify_to_cloud_params(config, args):
    # / home / adas / Semantic_data / cs_data /

    if args.pretrain != '.':
        config['trainer']['pretrain_path'] = args.pretrain
    if args.cloud:
        config = cloud_modify_config(config)
    if args.sup_only:
        config['train_loader']['args']['pseudo_label_dir'] = '.'
        config['trainer']['pretrain_path'] = '.'
    return config, args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # PARSE THE ARGS
    global parser
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('-c', '--config', default='config-deepv2.json', type=str,
                        help='Path to the config file (default: config-deepv2.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--num_servers', type=int, default=1)
    parser.add_argument('--idx_server', type=int, default=0)
    parser.add_argument('--ngpus_per_node', type=int, default=4,
                        help='num of gpus for each server')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('-ddp', '--distributed', action='store_true')
    parser.add_argument('--cloud', action='store_true', help='run on cloud setting')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=14, type=int, help='batch size for labeled data, if it ha unlabeled data'
                                                                   'the ratio is 1:1, so the total num per batch is twice')
    parser.add_argument('--split_seed', default=12345, type=int)
    parser.add_argument('--n_sup', default=-1, type=int)
    parser.add_argument('--rampup', default=16000, type=int)
    parser.add_argument('--contrastive_rampup', default=16000, type=int)
    parser.add_argument('--pretrain', '-p', default='.')
    parser.add_argument('--name', default='')
    parser.add_argument('--end_cb', type=float, default=0.4)
    parser.add_argument('--jitter', default=0, type=float)
    parser.add_argument('--gray', default=0, type=float)
    parser.add_argument('--blur', default=0, type=int)
    parser.add_argument('--trainer', default='base', type=str)
    parser.add_argument('--sup_only', action='store_true', help='')
    parser.add_argument('--unlabel_start_epoch', default=10, type=int)
    parser.add_argument('--contrastive_start_epoch', default=10, type=int)
    parser.add_argument('--consist_weight', default=50, type=float)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_alpha', type=float)
    parser.add_argument('--contrastive_loss_weight', default=1.0, type=float)
    parser.add_argument('--temperature', default=0.07, help='temprature for contrastive loss', type=float)
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--cb_threshold', action='store_true')
    parser.add_argument('--gamma', default=0., type=float, help='weight decay in cb threshold')
    parser.add_argument('--embed', default=32, type=int, help='embedding dimension')
    parser.add_argument('--epochs', type=int, default=40, )
    parser.add_argument('--autoaug', action='store_true')
    parser.add_argument('--p_head', default='linear', help='projection head type')
    parser.add_argument('--kld', action='store_true', help='kld regularization to prevent overconfidence, not use in project')
    parser.add_argument('--kld_weight', default=0.1, type=float)
    parser.add_argument('--contrastive_cross_model', action='store_true')
    parser.add_argument('--contrastive_cross_set', action='store_true')
    parser.add_argument('--hard_neg_num', default=64, type=int)
    parser.add_argument('--optimizer_type', default='Adam')
    parser.add_argument('--siam_weak', action='store_true')  # ablation study
    parser.add_argument('--siam_strong', action='store_true')  # ablation study
    parser.add_argument('--cutmix_ablation', action='store_true')  # ablation study
    parser.add_argument('--differential_lr', default=True, type=str2bool)
    parser.add_argument('--mask_contrast', action='store_true')
    parser.add_argument('--random_sample', action='store_true')  # ablation study
    parser.add_argument('--mask_unlabel', action='store_true', help='only used in -1')
    parser.add_argument('--ab_loss', action='store_true')
    parser.add_argument('--unfreeze_bn', action='store_true', help='unfreeze backbone bn')
    parser.add_argument('--no_fix_teacher_bn',action='store_true',)
    args, _ = parser.parse_known_args()
    config = json.load(open(args.config))
    config['optimizer']['type'] = args.optimizer_type
    config['arch']['args']['freeze_bn'] = not args.unfreeze_bn
    config['optimizer']['differential_lr'] = args.differential_lr
    config['train_loader']['args']['color_jitter'] = 0
    config['train_loader']['args']['random_grayscale'] = 0
    config['train_loader']['args']['gaussian_blur'] = 0
    config['unlabel_loader']['args']['color_jitter'] = args.jitter
    config['unlabel_loader']['args']['random_grayscale'] = args.gray
    config['unlabel_loader']['args']['gaussian_blur'] = args.blur
    config['unlabel_loader']['args']['siam_weak'] = args.siam_weak
    config['unlabel_loader']['args']['siam_strong'] = args.siam_strong
    config['trainer']['epochs'] = args.epochs
    config['trainer']['gamma'] = args.gamma
    config['trainer']['consist_weight'] = args.consist_weight
    config['trainer']['unlabel_start_epoch'] = args.unlabel_start_epoch
    config['trainer']['contrastive'] = args.contrastive
    if args.cutmix:
        config['trainer']['cutmix'] = True
    if args.ema:
        config['trainer']['ema'] = True
        config['trainer']['ema_alpha'] = args.ema_alpha
    if args.autoaug:
        config['unlabel_loader']['args']['autoaug'] = True
    if args.cb_threshold:
        config['trainer']['cb_threshold'] = True
    if config['trainer']['contrastive']:
        config['contrastive']['neg_num'] = args.hard_neg_num
        config['contrastive']['p_head'] = args.p_head
        config['contrastive']['embed_dim'] = args.embed
        config['contrastive']['rampup'] = args.contrastive_rampup
        config['contrastive']['start_epoch'] = args.contrastive_start_epoch
        config['contrastive']['visualize'] = args.visual
        config['arch']['args']['return_feat_and_logit'] = True
        config['contrastive']['contrastive_loss_weight'] = args.contrastive_loss_weight
        config['contrastive']['temperature'] = args.temperature
    else:
        config['arch']['args']['return_feat_and_logit'] = False
    config['name'] = config['name'] + 'n_sup{}_split{}_cb{}'.format(args.n_sup, args.split_seed, args.end_cb)
    config['optimizer']['args']['lr'] = args.lr
    config['train_loader']['args']['batch_size'] = args.batch_size
    config['unlabel_loader']['args']['batch_size'] = args.batch_size
    config['train_loader']['args'].update({'pretrain_from': config['arch']['args']['pretrain_from']})
    config['train_loader']['args']['n_sup'] = args.n_sup
    config['unlabel_loader']['args']['n_sup'] = args.n_sup
    config['train_loader']['args']['split_seed'] = args.split_seed
    config['val_loader']['args'].update({'pretrain_from': config['arch']['args']['pretrain_from']})
    config['trainer']['rampup'] = int(args.rampup)
    config, args = modify_to_cloud_params(config, args)
    if args.end_cb != 0.4:
        config['train_loader']['args']['end_percent'] = float(args.end_cb)
        config['unlabel_loader']['args']['end_percent'] = float(args.end_cb)
    if args.name != '':
        config['name'] = args.name
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    ngpus_per_node = args.ngpus_per_node
    world_size = args.num_servers * ngpus_per_node
    idx_server = args.idx_server
    args.world_size = world_size
    print('world size:%d' % world_size)
    if args.distributed:
        main_worker(config, args)
    else:
        main(config)
