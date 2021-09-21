import sys
sys.path.append('/home/ma-user/work/pytorch_segmentation-rebuttal')
from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import torch
from PIL import Image

ignore_label = 255
from dataloaders.transforms import build_transform, build_weak_strong_transform, \
    build_autoaug_transform, build_siam_transform

ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


class CityScapesDataset(BaseDataSet):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1, end_percent=0.4,
                 color_jitter=1, random_grayscale=0.2, gaussian_blur=11, to_bgr=False, rotate=False,
                 high=512, scale_list=[1.0], remove_calibration=False, siam_weak=False, siam_strong=False, **kwargs):
        self.num_classes = 19
        self.mode = mode
        self.palette = palette.CityScpates_palette
        self.id_to_trainId = ID_TO_TRAINID
        self.n_sup = n_sup
        self.split_seed = split_seed

        self.start_percent = 0.2
        self.end_percent = end_percent
        self.use_cb_threshold = False
        self.curr_epoch = 0
        self.scale_list = scale_list
        if rotate == True:
            rotation_degree = 10
        else:
            rotation_degree = 0
        self.rotation_degree = rotation_degree
        self.color_jitter = color_jitter
        self.random_grayscale = random_grayscale
        self.gaussian_blur = gaussian_blur
        self.to_bgr = to_bgr
        self.split = kwargs['split']
        self.high = high
        self.crop_size = kwargs['crop_size']
        self.siam_weak_transform = siam_weak
        self.siam_strong_transform = siam_strong
        self._set_transform()
        self.remove_calibration = remove_calibration

        super(CityScapesDataset, self).__init__(to_bgr=to_bgr, rotate=rotate, **kwargs)

    def _set_transform(self):
        if self.split == 'train':
            self.transform = build_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale, self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr, flip=True, high=self.high, scales=self.scale_list)
        else:
            self.transform = build_transform(
                rotation=0,
                color_jitter=0,
                random_grayscale=0,
                gaussian_blur=0,

                to_bgr=self.to_bgr, flip=False, high=self.high)

    def map_id_and_file(self):
        self.id_to_file = {}
        self.file_to_id = {}
        for i, name in enumerate(self.files):
            self.id_to_file[i] = name
            self.file_to_id[name] = i

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def init_cb_threshold(self, prob=0.9):
        self.cls_conf_threshold = [prob for _ in range(19)]  # [prob] * 19

    def update_ema_cb_threshold(self, prob, ema=0.95, gamma=0.0):
        probability, predict = torch.max(prob.data, 1)
        probability = probability.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        curr_prob_list = [[] for _ in range(19)]  #
        curr_cls_conf_threshold = [0] * 19
        for cls in range(19):
            if (cls == predict).any():
                curr_prob_list[cls].extend(probability[cls == predict].tolist())
        curr_total_num = [len(cls_prob) for cls_prob in curr_prob_list]
        curr_prob_list = [sorted(cls_prob, reverse=True) for cls_prob in curr_prob_list]
        for cls in range(19):
            if curr_total_num[cls] > 0:
                try:
                    if self.cls_conf_threshold[cls] == 0:
                        curr_cls_conf_threshold[cls] = curr_prob_list[cls] \
                            [int(curr_total_num[cls] * self.end_percent)]
                        self.cls_conf_threshold[cls] = curr_cls_conf_threshold[cls]
                    else:
                        curr_cls_conf_threshold[cls] = curr_prob_list[cls] \
                            [int(curr_total_num[cls] * self.end_percent * self.cls_conf_threshold[cls] ** gamma)]
                        self.cls_conf_threshold[cls] = self.cls_conf_threshold[cls] * ema + (1 - ema) * \
                                                       curr_cls_conf_threshold[cls]
                except:
                    print('bug in ema bc threshold func!')

    def get_cb_threshould_mask(self, prob):
        prob, pred = torch.max(prob, 1)
        conf_mask = torch.zeros_like(pred).float()
        binary_mask = torch.zeros_like(pred)
        for cls_id in range(19):
            conf_mask[pred == cls_id] = self.cls_conf_threshold[cls_id]
        binary_mask[prob >= conf_mask] = 1
        return binary_mask[:, None, :, :]

    def _set_files(self):
        assert (self.mode == 'fine' and self.split in ['train', 'val']) or \
               (self.mode == 'coarse' and self.split in ['train', 'train_extra', 'val'])

        SUFIX = '_gtFine_labelIds.png'
        if self.mode == 'coarse':
            label_path = os.path.join(self.root, 'gtCoarse', self.split)
        else:
            label_path = os.path.join(self.root, 'gtFine', self.split)
        image_path = os.path.join(self.root, 'leftImg8bit', self.split)
        image_paths, label_paths = [], []
        if self.n_sup == -1 or self.split != 'train':
            for city in os.listdir(image_path):
                image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.png'))))
                label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))
        else:
            assert self.n_sup in [4, 744, 372, 100, ]
            assert self.split_seed in [12345, 23456, 34567, 45678, 56789]
            SUP_SUBSET_FILE = 'splits/train_{}_seed{}.txt'.format(self.n_sup, self.split_seed)

            with open(SUP_SUBSET_FILE, 'r') as filehandle:
                for line in filehandle:
                    # remove linebreak which is the last character of the string
                    currentPlace = line[:-1]
                    image_paths.append(
                        os.path.join(self.root, 'leftImg8bit', currentPlace.replace('_x.png', '_leftImg8bit.png')))
                    label_paths.append(os.path.join(self.root, 'gtFine', currentPlace.replace('_x.png', SUFIX)))
        assert len(label_paths) == len(image_paths)
        self.files = list(zip(image_paths, label_paths))
        self.isPseudo = [0] * len(self.files)
        print('Total train sample in the dataset: %d' % len(self.files))

    def _augmentation(self, image, label):
        image, label = self.transform(image, label)
        return image, label

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        for k, v in self.id_to_trainId.items():
            label[label == k] = v

        if self.isPseudo[index] == 1:
            if self.use_cb_threshold:
                # use class balanced threshold
                pred_conf = np.asarray(Image.open(label_path.replace('pseudo.png', 'prob.png')), dtype=np.float16)
                conf_mask = np.zeros_like(pred_conf)
                for cls_id in range(19):
                    conf_mask[label == cls_id] = self.cls_conf_threshold[self.curr_epoch, cls_id]
                label[pred_conf < conf_mask] = ignore_label
        return image, label, image_id


class CityScapes(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 mode='fine', val=False,
                 shuffle=False, num_samples=1000, flip=False, rotate=False, blur=False, augment=False, val_split=None,
                 return_id=False,
                 distributed=False, world_size=1, rank=-1, pretrain_from='imagenet', split_seed=12345, n_sup=-1,
                 end_percent=0.4, color_jitter=1, random_grayscale=0.2, gaussian_blur=11, high=512, scale_list=[1.0],
                 ):
        if pretrain_from == 'imagenet':
            to_bgr = False
            self.MEAN = [0.485, 0.456, 0.406]
            self.STD = [0.229, 0.224, 0.225]
        else:
            to_bgr = True
            self.MEAN = [73.15835921, 82.90891754, 72.39239876]
            self.STD = [1., 1., 1.]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'to_bgr': to_bgr,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'split_seed': split_seed,
            'n_sup': n_sup,

            'end_percent': end_percent,
            "color_jitter": color_jitter,
            "random_grayscale": random_grayscale,
            "gaussian_blur": gaussian_blur,
            "high": high,
            "scale_list": scale_list,

        }

        self.dataset = CityScapesDataset(mode=mode, **kwargs)
        super(CityScapes, self).__init__(self.dataset, batch_size, shuffle, num_workers,
                                         val_split, distributed, world_size, rank, num_samples)
    # add 06/13 for rebuttal exp
    def init_cb_threshold(self, prob=0.9):
        return self.dataset.init_cb_threshold(prob)

    def get_cb_threshould_mask(self, prob):
        return self.dataset.get_cb_threshould_mask(prob)

    def update_ema_cb_threshold(self, prob, ema=0.99, gamma=0.0):
        self.dataset.update_ema_cb_threshold(prob, ema, gamma)

class PseudoCityScapesDataset(CityScapesDataset):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1,
                 end_percent=0.4, autoaug=False, labeled_part=False, high=512, scale_list=[1.0], **kwargs):
        self.num_classes = 19
        self.mode = mode
        self.palette = palette.CityScpates_palette
        self.id_to_trainId = ID_TO_TRAINID
        self.n_sup = n_sup
        self.split_seed = split_seed

        self.start_percent = 0.2
        self.end_percent = end_percent
        self.use_cb_threshold = False
        self.curr_epoch = 0
        self.cls_prob_list = [[] for _ in range(19)]
        self.autoaug = autoaug
        self.high = high
        self.crop_size = kwargs['crop_size']
        self.scale_list = scale_list
        self.labeled_part = labeled_part  # if true, load labeled set, if false load unlabeled set

        super(PseudoCityScapesDataset, self).__init__(mode=mode, n_sup=n_sup, split_seed=split_seed,
                                                      end_percent=end_percent, high=high, **kwargs)

    def _set_transform(self):
        '''
        strong contain jittering, grayscale and gaussian
        :return: self.transform = [share_trans, strong_trans, weak_trans]
        '''

        if not self.autoaug:
            self.transform = build_weak_strong_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale,
                self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                high=self.high,
                scales=self.scale_list
            )
        elif self.siam_strong_transform:
            print('using siam-strong-transform')
            self.transform = build_siam_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale,
                self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                high=self.high,
                scales=self.scale_list,
                mode='strong'
            )
        elif self.siam_weak_transform:
            print('using siam-weak-transform')
            self.transform = build_siam_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale,
                self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                high=self.high,
                scales=self.scale_list,
                mode='weak'
            )
        else:
            print('using autoaug-transform')
            self.transform = build_autoaug_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale,
                self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                high=self.high,
                scales=self.scale_list
            )

    def init_cb_threshold(self, prob=0.9):
        self.cls_conf_threshold = [prob for _ in range(19)]  # [prob] * 19

    def update_ema_cb_threshold(self, prob, ema=0.95, gamma=0.0):

        probability, predict = torch.max(prob.data, 1)
        probability = probability.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        curr_prob_list = [[] for _ in range(19)]  #
        curr_cls_conf_threshold = [0] * 19
        for cls in range(19):
            if (cls == predict).any():
                curr_prob_list[cls].extend(probability[cls == predict].tolist())
        curr_total_num = [len(cls_prob) for cls_prob in curr_prob_list]
        curr_prob_list = [sorted(cls_prob, reverse=True) for cls_prob in curr_prob_list]
        for cls in range(19):
            if curr_total_num[cls] > 0:
                try:

                    if self.cls_conf_threshold[cls] == 0:
                        curr_cls_conf_threshold[cls] = curr_prob_list[cls] \
                            [int(curr_total_num[cls] * self.end_percent)]
                        self.cls_conf_threshold[cls] = curr_cls_conf_threshold[cls]
                    else:
                        curr_cls_conf_threshold[cls] = curr_prob_list[cls] \
                            [int(curr_total_num[cls] * self.end_percent * self.cls_conf_threshold[cls] ** gamma)]
                        self.cls_conf_threshold[cls] = self.cls_conf_threshold[cls] * ema + (1 - ema) * \
                                                       curr_cls_conf_threshold[cls]
                except:
                    print('bug in ema bc threshold func!')

    def get_cb_threshould_mask(self, prob):
        prob, pred = torch.max(prob, 1)
        conf_mask = torch.zeros_like(pred).float()
        binary_mask = torch.zeros_like(pred)
        for cls_id in range(19):
            conf_mask[pred == cls_id] = self.cls_conf_threshold[cls_id]
        binary_mask[prob >= conf_mask] = 1
        return binary_mask[:, None, :, :]

    def _set_files(self):
        assert (self.mode == 'fine' and self.split in ['train', 'val']) or \
               (self.mode == 'coarse' and self.split in ['train', 'train_extra', 'val'])
        SUFIX = '_gtFine_labelIds.png'

        if self.mode == 'coarse':
            label_path = os.path.join(self.root, 'gtCoarse', self.split)
        else:
            label_path = os.path.join(self.root, 'gtFine', self.split)
        image_path = os.path.join(self.root, 'leftImg8bit', self.split)
        all_image_paths = []
        all_label_paths = []
        label_paths = []
        labeled_image_paths = []
        for city in os.listdir(image_path):
            all_image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.png'))))
            all_label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))
        if self.n_sup == -1:
            sorted(all_image_paths)
            sorted(all_label_paths)
            self.files = list(zip(all_image_paths, all_label_paths))
            print('Total labeled sample in the pseudodataset: %d' % len(self.files))
            return

        assert self.n_sup in [4, 744, 372, 100]
        assert self.split_seed in [12345, 23456, 34567, 45678, 56789]
        SUP_SUBSET_FILE = 'splits/train_{}_seed{}.txt'.format(self.n_sup, self.split_seed)
        with open(SUP_SUBSET_FILE, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                # add item to the list
                labeled_image_paths.append(
                    os.path.join(self.root, 'leftImg8bit', currentPlace.replace('_x.png', '_leftImg8bit.png')))
                label_paths.append(
                    os.path.join(self.root, 'gtFine', currentPlace.replace('_x.png', '_gtFine_labelIds.png'))
                    )
        if not self.labeled_part:
            unlabeled_img_path = list(set(all_image_paths) - set(labeled_image_paths))
            unlabeled_label_path = list(set(all_label_paths) - set(label_paths))

            unlabeled_img_path = sorted(unlabeled_img_path)
            unlabeled_label_path = sorted(unlabeled_label_path)
            self.files = list(zip(unlabeled_img_path, unlabeled_label_path))  # unlabeled_img_path

            print('Total pseudo sample for generation in the dataset: %d' % len(self.files))
        else:
            labeled_image_paths = sorted(labeled_image_paths)
            label_paths = sorted(label_paths)
            self.files = list(zip(labeled_image_paths, label_paths))
            print('Total labeled sample in the dataset: %d' % len(self.files))

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label, image_id

    def _augmentation(self, image, label):
        image, label = self.transform[0](image, label)
        weak, _ = self.transform[2](image.copy(), None)
        strong, _ = self.transform[1](image.copy(), None)
        return weak, strong, label

    def get_name_from_id(self, index):
        filename = self.files[index]
        basename = filename.split('/')[-1]
        return basename

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        aug_data = self._augmentation(image, label)
        weak, strong, label = aug_data
        weak = Image.fromarray(np.uint8(weak))
        strong = Image.fromarray(np.uint8(strong))
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        for k, v in self.id_to_trainId.items():
            label[label == k] = v
        return self.normalize(self.to_tensor(weak)), self.normalize(self.to_tensor(strong)), label, image_id


class PseudoCityScapes(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 mode='fine', val=False,
                 shuffle=False, num_samples=1000, flip=False, rotate=False, blur=False, augment=False, val_split=None,
                 return_id=False,
                 distributed=False, world_size=1, rank=-1, pretrain_from='imagenet', split_seed=12345,
                 n_sup=-1, end_percent=0.4, color_jitter=1, random_grayscale=0.2, gaussian_blur=11,
                 generation=False, autoaug=False, labeled_part=False, high=512, scale_list=[1.0],
                 siam_weak=False, siam_strong=False):
        if pretrain_from == 'imagenet':
            to_bgr = False
            self.MEAN = [0.485, 0.456, 0.406]
            self.STD = [0.229, 0.224, 0.225]
        else:
            to_bgr = True
            self.MEAN = [73.15835921, 82.90891754, 72.39239876]
            self.STD = [1., 1., 1.]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'to_bgr': to_bgr,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'split_seed': split_seed,
            'n_sup': n_sup,

            'end_percent': end_percent,
            "color_jitter": color_jitter,
            "random_grayscale": random_grayscale,
            "gaussian_blur": gaussian_blur,
            "autoaug": autoaug,
            "labeled_part": labeled_part,
            "high": high,
            "scale_list": scale_list,

            "siam_weak": siam_weak,
            "siam_strong": siam_strong
        }

        self.dataset = PseudoCityScapesDataset(mode=mode, **kwargs)

        super(PseudoCityScapes, self).__init__(self.dataset, batch_size, shuffle, num_workers,
                                               val_split, distributed, world_size, rank, num_samples)


class AugLabeledDataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_weak = None
        self.next_strong = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def set_epoch(self, epoch):
        self.loader.sampler.set_epoch(epoch)

    def preload(self):
        try:
            self.next_weak, self.next_strong, self.next_target, self.next_id = next(self.loaditer)
        except StopIteration:
            self.next_weak = None
            self.next_strong = None
            self.next_id = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_weak = self.next_weak.cuda(device=self.device, non_blocking=True)
            self.next_strong = self.next_strong.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def init_cb_threshold(self, prob=0.9):
        return self.loader.dataset.init_cb_threshold(prob)

    def get_cb_threshould_mask(self, prob):
        return self.loader.get_cb_threshould_mask(prob)

    def update_ema_cb_threshold(self, prob, ema=0.99, gamma=0.0):
        self.loader.dataset.update_ema_cb_threshold(prob, ema, gamma)

    def __iter__(self):
        self.count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_weak is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            weak = self.next_weak
            strong = self.next_strong
            target = self.next_target
            id = self.next_id
            self.preload()
            self.count += 1
            yield weak, strong, target, id
            if type(self.stop_after) is int and (self.count > self.stop_after):
                break

    def reset(self):
        if type(self.stop_after) is int:
            self.count = 0


class PseudoDataWithLabelPrefetcher(AugLabeledDataPrefetcher):
    def __init__(self, loader, device, stop_after=None):
        super(PseudoDataWithLabelPrefetcher, self).__init__(loader, device, stop_after)


if __name__ == '__main__':
    pass
