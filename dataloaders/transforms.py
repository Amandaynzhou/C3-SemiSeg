
import sys
sys.path.append('/home/ma-user/work/pytorch_segmentation-master')
import torch
from PIL import Image
import torchvision.transforms as transforms
import numbers
import torchvision.transforms.functional as F
import cv2
import pdb

#### AutoAug:
""" AutoAugment and RandAugment
Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
Papers: https://arxiv.org/abs/1805.09501, https://arxiv.org/abs/1906.11172, and https://arxiv.org/abs/1909.13719
Hacked together by Ross Wightman
"""
import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance
import PIL
import numpy as np


_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (0, 0, 0)

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def identity(img, **__):
    return img


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _MAX_LEVEL) * 1.8 + 0.1,


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, _hparams):
    # range [-0.45, 0.45]
    level = (level / _MAX_LEVEL) * 0.45
    level = _randomly_negate(level)
    return level,


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    return int((level / _MAX_LEVEL) * 4) + 4,


def _posterize_research_level_to_arg(level, _hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image'
    return 4 - int((level / _MAX_LEVEL) * 4),


def _posterize_tpu_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    return int((level / _MAX_LEVEL) * 4),


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    return int((level / _MAX_LEVEL) * 256),


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _MAX_LEVEL) * 110),


LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Identity': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'PosterizeResearch': _posterize_research_level_to_arg,
    'PosterizeTpu': _posterize_tpu_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Identity': identity,
    # 'Rotate': rotate,
    'PosterizeOriginal': posterize,
    'PosterizeResearch': posterize,
    'PosterizeTpu': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    # 'ShearX': shear_x,
    # 'ShearY': shear_y,
    # 'TranslateX': translate_x_abs,
    # 'TranslateY': translate_y_abs,
    # 'TranslateXRel': translate_x_rel,
    # 'TranslateYRel': translate_y_rel,
}


class AutoAugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude)) # clip to valid range
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert', # todo: remove?
    # 'Rotate',
    'PosterizeTpu',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    # 'ShearX',
    # 'ShearY',
    # 'TranslateXRel',
    # 'TranslateYRel',
    #'Cutout'  # FIXME I implement this as random erasing separately
]

_RAND_TRANSFORMS_CMC = [
    'AutoContrast',
    'Identity',
    'Rotate',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # FIXME I implement this as random erasing separately
]


# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'PosterizeTpu': 0,
    'Invert': 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    """rand augment ops for RGB images"""
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AutoAugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


def rand_augment_ops_cmc(magnitude=10, hparams=None, transforms=None):
    """rand augment ops for CMC images (removing color ops)"""
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS_CMC
    return [AutoAugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(config_str, hparams, use_cmc=False):
    """
    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :param use_cmc: Flag indicates removing augmentation for coloring ops.
    :return: A PyTorch compatible Transform
    """
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    if use_cmc:
        ra_ops = rand_augment_ops_cmc(magnitude=magnitude, hparams=hparams)
    else:
        ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)

######
def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class RandomVerticalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            if mask is None:
                return  img.transpose(Image.FLIP_TOP_BOTTOM), mask
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

class RandomHFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            if mask is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomCropNumpy(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))#256 512
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, mask):
        h,w,_ = img.shape
        pad_h = max(self.size[0] - h, 0)
        pad_w = max(self.size[1] - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, value=0, **pad_kwargs)
            if mask is not None:
                mask = cv2.copyMakeBorder(mask, value=0, **pad_kwargs)

        # Cropping
        h, w, _ = img.shape
        start_h = random.randint(0, h - self.size[0])
        start_w = random.randint(0, w - self.size[1])
        end_h = start_h + self.size[0]
        end_w = start_w + self.size[1]
        img = img[start_h:end_h, start_w:end_w]
        if mask is not None:
            mask = mask[start_h:end_h, start_w:end_w]
        return img, mask

class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.fill_mask = 255
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            if mask is not None:
                mask = F.pad(mask, self.padding, self.fill_mask, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            if mask is not None:
                mask =  F.pad(mask, (self.size[1] - mask.size[0], 0), self.fill_mask, self.padding_mode)
         # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            if mask is not None:
                mask = F.pad(mask, (0, self.size[0] - mask.size[1]), self.fill_mask, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        if mask is not None:
            return F.crop(img, i, j, h, w), F.crop(mask, i, j, h, w)
        else:
            return F.crop(img, i, j, h, w),mask
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class RandRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask = None):
        angle = random.randint(-self.degree, self.degree)
        h, w,_  = img.shape
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)  # , borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)  # ,  borderMode=cv2.BORDER_REFLECT)
        return img, mask

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, mask):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor, mask

class Normalize(object):
    def __init__(self, mean, std):
        self.transform = transforms.Normalize(mean, std)

    def __call__(self, img, mask):
        img = img
        return self.transform(img), mask

class MaskToTensor(object):
    def __call__(self, img, blockout_predefined_area=False):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class PairToTensor(object):
    def __init__(self):
        pass
    def __call__(self, img, mask):
        if mask is None:
            return  F.to_tensor(img), mask
        return F.to_tensor(img), torch.from_numpy(np.array(mask, dtype=np.int32)).long()

class MapLabel(object):
    def __init__(self, labelmap):
        self.label_map = labelmap

    def __call__(self, img, mask):
        if mask is None:
            return img, mask
        for k, v in self.label_map.items():
            mask[ mask == k] = v
        return img, mask

class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        if self.target_h == -1:
            # do not resize
            return img, mask
        w, h = img.size
        target_w = int(w / h * self.target_h)
        if mask is not None:
            mask = mask.resize((target_w, self.target_h), Image.NEAREST)
        return img.resize((target_w, self.target_h), self.interpolation),\
               mask

class FlipChannels(object):
    """
    Flip around the x-axis
    """
    def __call__(self, img, mask):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8)), mask

class ColorJit(object):
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., prob = 0.8 ):
        self.transform =transforms.RandomApply([ transforms.ColorJitter(brightness, contrast, saturation, hue)
                                                ], prob)
    def __call__(self, img, mask):
        img = self.transform(img)
        return img, mask

class ToBGR(object):
    def __call__(self, img, mask):
        img = img[:,:,::-1].copy()
        return img, mask

class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, img, mask):
        img = np.asarray(img)
        if mask is not None:
            mask =np.asarray(mask)
        return img, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            try:
                img, mask = t(img, mask)
            except:
                # autoAug
                img = t(img)
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToPIL(object):
    def __call__(self, image, mask):
        image = F.to_pil_image(image.astype(np.uint8))
        if mask is not None:
            mask = F.to_pil_image(mask)
        return image, mask

class GrayScale(object):
    def __init__(self, p):
        self.p = p
        self.trans = transforms.RandomGrayscale(p)

    def __call__(self, img, mask):
        return self.trans(img), mask

class GaussianBlur(object):
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample, mask):
        # blur the image with a 50% chance
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample, mask

class RandomScale(object):
    def __init__(self, scales = [1.0]):
        self.scale = scales

    def __call__(self, sample, mask = None):
        if len(self.scale)==1 and self.scale[0] ==1.0:
            return sample, mask

        else:
            scale = random.choice(self.scale)
            sh = int(sample.size[1] * scale)
            sw = int(sample.size[0] * scale)

            sample = sample.resize((sw, sh), PIL.Image.BILINEAR)
            if mask is not None:
                mask = mask.resize((sw, sh), PIL.Image.NEAREST)
            return sample, mask


class RandomRangeScale(object):
    def __init__(self, scales = [1.0]):
        self.scale = scales

    def __call__(self, sample, mask = None):
        if len(self.scale)==1 and self.scale[0] ==1.0:
            return sample, mask

        else:
            scale = random.uniform(self.scale[0], self.scale[1])
            # scale = random.choice(self.scale)
            sh = int(sample.size[1] * scale)
            sw = int(sample.size[0] * scale)

            sample = sample.resize((sw, sh), PIL.Image.BILINEAR)
            if mask is not None:
                mask = mask.resize((sw, sh), PIL.Image.NEAREST)
            return sample, mask


def build_transform(rotation= 0, color_jitter=1, random_grayscale= 0.2,
                    gaussian_blur=11, high = 512, crop = (256,512), to_bgr = False,
                    flip = True, scales = [1.0], pad_if_need = False):
    trans_list = [
        ToPIL(),
        ResizeHeight(high),
        RandomScale(scales),
        RandomCrop(crop,pad_if_needed=pad_if_need),
        ]
    if flip:
        trans_list.append(RandomHFlip())
    if color_jitter>0:
        trans_list.append(ColorJit(color_jitter*0.8,color_jitter*0.8,color_jitter*0.8,color_jitter*0.2))
    if random_grayscale>0:
        trans_list.append(GrayScale(random_grayscale))
    if gaussian_blur>0:
        if gaussian_blur%2==0:
            gaussian_blur+=1 # make it odd
        trans_list.insert(0, GaussianBlur(gaussian_blur))
    if to_bgr:
        trans_list.insert(0, ToBGR())
    trans_list.append(ToNumpy())
    if rotation>0:
        trans_list.append(RandRotation(rotation))
    trans_list = Compose(trans_list)
    return trans_list


def build_weak_strong_transform(rotation = 10, color_jitter=1, random_grayscale= 0.2,
                                gaussian_blur=11, high = 512, crop = (256,512),
                                to_bgr = False, flip = True,scales = [1.0],pad_if_need=False):
    '''

    useage:  img,mask = share_transforms(img,mask)
             weak,mask = weak_transforms(img.copy(), mask.copy())
             strong, _ = strong_transforms(img.copy(), None)
             :return strong, weak, mask
    '''

    share_transforms = [ToPIL(),
        ResizeHeight(high),
        RandomScale(scales),
        RandomCrop(crop,pad_if_needed=pad_if_need),]
    if to_bgr:
        share_transforms.insert(0, ToBGR())
    if flip:
        share_transforms.append(RandomHFlip())
    if rotation>0:
        share_transforms.insert(0,RandRotation(rotation))
    share_transforms = Compose(share_transforms)

    strong_transforms = [ColorJit(color_jitter*0.5,color_jitter*0.5,color_jitter*0.5,color_jitter*0.25, prob=1.)]
    if random_grayscale>0:
        strong_transforms.append(GrayScale(random_grayscale))
    strong_transforms.append(ToNumpy())
    strong_transforms = Compose(strong_transforms)
    weak_transforms = [ToNumpy()]
    weak_transforms = Compose(weak_transforms)
    return share_transforms, strong_transforms, weak_transforms

def build_siam_transform(rotation = 10, color_jitter=1, random_grayscale= 0.2,
                                gaussian_blur=11, high = 512, crop = (256,512),
                                to_bgr = False, flip = True,scales = [1.0],pad_if_need=False, mode = 'strong'):
    '''

    useage:  img,mask = share_transforms(img,mask)
             weak,mask = weak_transforms(img.copy(), mask.copy())
             strong, _ = strong_transforms(img.copy(), None)
             :return strong, weak, mask
    '''
    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )
    share_transforms = [ToPIL(),
        ResizeHeight(high),
        RandomScale(scales),
        RandomCrop(crop,pad_if_needed=pad_if_need),]
    if to_bgr:
        share_transforms.insert(0, ToBGR())
    if flip:
        share_transforms.append(RandomHFlip())
    if rotation>0:
        share_transforms.insert(0,RandRotation(rotation))
    share_transforms = Compose(share_transforms)
    if mode =='strong':
        strong_transforms = [ColorJit(color_jitter*0.5,color_jitter*0.5,color_jitter*0.5,color_jitter*0.25, prob=1.)]
        strong_transforms = strong_transforms + \
                            [rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params, )]
        if random_grayscale>0:
            strong_transforms.append(GrayScale(random_grayscale))
        strong_transforms.append(ToNumpy())
        strong_transforms = Compose(strong_transforms)
        # this *weak* transforms is not weak!
        weak_transforms = [ColorJit(color_jitter*0.5,color_jitter*0.5,color_jitter*0.5,color_jitter*0.25, prob=1.)]
        weak_transforms =  weak_transforms + \
                            [rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params, )]
        if random_grayscale>0:
            weak_transforms.append(GrayScale(random_grayscale))
        weak_transforms.append(ToNumpy())
        weak_transforms = Compose(weak_transforms)
    else:
        weak_transforms = [ToNumpy()]
        weak_transforms = Compose(weak_transforms)
        strong_transforms = [ToNumpy()]
        strong_transforms = Compose(strong_transforms)
    return share_transforms, strong_transforms, weak_transforms


def build_autoaug_transform(rotation = 10, color_jitter=1, random_grayscale= 0.2,
                            gaussian_blur=11, high = 512, crop = (256,512), to_bgr = False,
                            flip=True, scales=[1.0],pad_if_need=False ):
    '''

    useage:  img,mask = share_transforms(img,mask)
             weak,mask = weak_transforms(img.copy(), mask.copy())
             strong, _ = strong_transforms(img.copy(), None)
             :return strong, weak, mask
    '''
    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )
    share_transforms = [ToPIL(),
        ResizeHeight(high),
        RandomScale(scales),
        RandomCrop(crop,pad_if_needed=pad_if_need),]
    if to_bgr:
        share_transforms.insert(0, ToBGR())
    if flip:
        share_transforms.append(RandomHFlip())
    if rotation>0:
        share_transforms.insert(0,RandRotation(rotation))
    share_transforms = Compose(share_transforms)
    strong_transforms = [ColorJit(color_jitter*0.5,color_jitter*0.5,color_jitter*0.5,color_jitter*0.25, prob=1.)]
    strong_transforms = strong_transforms +\
                        [ rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params,)]
    # strong_transforms = [ColorJit(color_jitter*0.5,color_jitter*0.5,color_jitter*0.5,color_jitter*0.25, prob=1.)]
    if random_grayscale>0:
        strong_transforms.append(GrayScale(random_grayscale))
    strong_transforms.append(ToNumpy())
    strong_transforms = Compose(strong_transforms)
    weak_transforms = [ToNumpy()]
    weak_transforms = Compose(weak_transforms)
    return share_transforms, strong_transforms, weak_transforms

## VOC

def build_voc_pseudo_transform(rotation = 10, color_jitter=1, random_grayscale= 0.2,
                            gaussian_blur=11, high = -1, crop = (320,320), to_bgr = False,
                            flip=True, scales=[0.5, 2.0],pad_if_need=False ):
    '''

    useage:  img,mask = share_transforms(img,mask)
             weak,mask = weak_transforms(img.copy(), mask.copy())
             strong, _ = strong_transforms(img.copy(), None)
             :return strong, weak, mask
    '''
    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )
    share_transforms = [ToPIL(),
        ResizeHeight(high),
        RandomRangeScale(scales),
        RandomCrop(crop,pad_if_needed=pad_if_need),]
    if to_bgr:
        share_transforms.insert(0, ToBGR())
    if flip:
        share_transforms.append(RandomHFlip())
    if rotation>0:
        share_transforms.insert(0,RandRotation(rotation))
    share_transforms = Compose(share_transforms)
    strong_transforms = [ColorJit(color_jitter*0.5,color_jitter*0.5,color_jitter*0.5,color_jitter*0.25, prob=1.)]
    strong_transforms = strong_transforms +\
                        [ rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params,)]
    # strong_transforms = [ColorJit(color_jitter*0.5,color_jitter*0.5,color_jitter*0.5,color_jitter*0.25, prob=1.)]
    if random_grayscale>0:
        strong_transforms.append(GrayScale(random_grayscale))
    strong_transforms.append(ToNumpy())
    strong_transforms = Compose(strong_transforms)
    weak_transforms = [ToNumpy()]
    weak_transforms = Compose(weak_transforms)
    return share_transforms, strong_transforms, weak_transforms


def build_voc_transform(rotation= 0, color_jitter=1, random_grayscale= 0.2,
                    gaussian_blur=11, high = 512, crop = (320,320), to_bgr = False,
                    flip = True, scales = [0.5,2.0], pad_if_need = False):
    trans_list = [
        ToPIL(),
        ResizeHeight(high),
        RandomRangeScale(scales),
        RandomCrop(crop,pad_if_needed=pad_if_need),
        ]
    if flip:
        trans_list.append(RandomHFlip())
    if color_jitter>0:
        trans_list.append(ColorJit(color_jitter*0.8,color_jitter*0.8,color_jitter*0.8,color_jitter*0.2))
    if random_grayscale>0:
        trans_list.append(GrayScale(random_grayscale))
    if gaussian_blur>0:
        if gaussian_blur%2==0:
            gaussian_blur+=1 # make it odd
        trans_list.insert(0, GaussianBlur(gaussian_blur))
    if to_bgr:
        trans_list.insert(0, ToBGR())
    trans_list.append(ToNumpy())
    if rotation>0:
        trans_list.append(RandRotation(rotation))
    trans_list = Compose(trans_list)
    return trans_list



##https://github.com/Britefury/cutmix-semisup-seg/blob/44e81b3ae862d2da7d1c4df77fb274f8f1f0a861/mask_gen.py#L46
class MaskGenerator (object):
    """
    Mask Generator
    """

    def generate_params(self, n_masks, mask_shape, rng=None):
        raise NotImplementedError('Abstract')

    def append_to_batch(self, *batch):
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        raise NotImplementedError('Abstract')

class BoxMaskGenerator(MaskGenerator):
    def __init__(self, prop_range=0.5, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True, invert=False):

        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        return t_params

if __name__ == '__main__':
    img = Image.open('../dataset/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png').convert('RGB')
    # img2 = Image.open('../dataset/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png').convert('RGB')

    img= np.asarray(img, dtype=np.uint8)
    # img =  np.asarray(img2, dtype=np.uint8)
    trans = build_autoaug_transform(random_grayscale=0.0, gaussian_blur=0)
    # trans = build_transform()
    image, _ =trans[0](img, None)
    weak, _ = trans[2](image.copy(), None)
    strong, _ =trans[1](image.copy(), None)
    cv2.imwrite('weak.png', weak)
    cv2.imwrite('strong.png', strong)
    # res = trans(img, img2)
    # import pdb

    pdb.set_trace()
    print('finish')