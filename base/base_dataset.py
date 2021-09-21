import cv2
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                 crop_size=321, scale=True, flip=True, rotate=False, blur=False,
                 return_id=False, to_bgr=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.to_bgr = to_bgr
        self.crop_size = crop_size
        if isinstance(self.crop_size, list):
            self.crop_h, self.crop_w = self.crop_size
        else:
            self.crop_h = self.crop_w = self.crop_size
        self.base_size = base_size
        if self.augment:
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id
        self.image_padding = (np.array(mean) * 255.).tolist()
        cv2.setNumThreads(0)
        # self.map_id_and_file()

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def _set_transform(self):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        h, w = label.shape
        if self.base_size:
            longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
            int(1.0 * longside * h / w + 0.5), longside)
        # do not use crop in val
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.to_bgr:
            image = image[:, :, ::-1]
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int32)
        return image, label

    def _augmentation(self, image, label):
        if self.to_bgr:
            image = image[:, :, ::-1]
        h, w, _ = image.shape
        # if image shape != label shape resize image to label shape
        h_l, w_l = label.shape
        if h != h_l or w != w_l: image = cv2.resize(image, (w_l, h_l), interpolation=cv2.INTER_LINEAR)
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
            int(1.0 * longside * h / w + 0.5), longside)

            # WE change INTER_LINEAR TO  Image.BICUBIC IN VOC, for BDD/CITYSCAPE, We use cv2.INTER_LINEAR
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR)  # , borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h),
                                   flags=cv2.INTER_NEAREST)  # ,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_h - h, 0)
            pad_w = max(self.crop_w - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT, }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=255, **pad_kwargs)

            # Cropping
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_h)
            start_w = random.randint(0, w - self.crop_w)
            end_h = start_h + self.crop_h
            end_w = start_w + self.crop_w
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                     borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        # pdb.set_trace()
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

    def map_id_and_file(self):
        pass

    def get_cb_threshould_mask(self, prob):
        raise NotImplementedError

    def update_pred(self, prob):
        raise NotImplementedError

    def calculate_cb_threshold(self):
        raise NotImplementedError
