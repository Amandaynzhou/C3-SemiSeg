import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
import dataloaders
import models
from collections import OrderedDict
from utils.metrics import eval_metrics

from dataloaders.labels.cityscapes import trainId2label
import pdb

ignore_label = 255

ID_TO_TRAINID = {   7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 31: 16, 32: 17, 33: 18}


def main():
    TRAINID_TO_ID = {}
    for k,v in ID_TO_TRAINID.items():
        TRAINID_TO_ID[v] = k
    TRAINID_TO_NAME = {}
    from utils.metrics import seg_metrics
    args = parse_arguments()
    print(args.n_sup)
    print(args.model)
    config = json.load(open(args.config))
    SEG_METRICS = seg_metrics()
    dataset_type = config['train_loader']['type']
    assert dataset_type == 'CityScapes'
    loader = getattr(dataloaders, config['val_loader']['type'])(**config['val_loader']['args'])
    # Model
    model = getattr(models, config['arch']['type'])(19, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    SUFIX = '_gtFine_labelIds.png'
    image_path = os.path.join(config['val_loader']['args']['data_dir'], 'leftImg8bit', args.split)
    image_paths, label_paths = [], []
    label_path = os.path.join(config['val_loader']['args']['data_dir'],
                              'gtFine_trainvaltest', 'gtFine', args.split)
    for city in os.listdir(image_path):
        image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.png'))))
        label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))

    # add cityscape outputpath
    output_dir = os.path.dirname(args.model).replace('saved', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.environ['CITYSCAPES_RESULTS'] = output_dir
    with torch.no_grad():
        tbar = tqdm(loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            prediction = model(data.to(device))
            prediction = F.softmax(prediction, dim=1)
            seg_metric = eval_metrics(prediction, target.to(device), 19)
            SEG_METRICS._update_seg_metrics(*seg_metric)
            pixAcc, mIoU, _ =SEG_METRICS._get_seg_metrics().values()
            tbar.set_description('INFER | PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(
                                            pixAcc, mIoU))

    seg_metrics = SEG_METRICS._get_seg_metrics()
    for k, v in list(seg_metrics.items())[:-1]:
        print(f'{k}:{v}\n')
    # print class-wise results
    cls_result = seg_metrics['Class_IoU']
    for k, v in cls_result.items():
        print(trainId2label[k].name, ':', v, '\n')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config/config-contrastive.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-s','--split', default='val', help='which split for evaluation')
    parser.add_argument('--n_sup', default= -1)
    parser.add_argument('--seed', default= 12345)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
