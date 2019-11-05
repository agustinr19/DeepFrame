import types

import torch
import numpy as np

from PIL import Image, ImageOps, ImageEnhance
import scipy.ndimage.interpolation as itpl
import scipy.misc as misc


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def adjust_brightness(img, brightness_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img

def adjust_contrast(img, contrast_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img

def adjust_saturation(img, saturation_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img

def adjust_hue(img, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue factor {} is not in [-0.5, 0.5]'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()
    np_h = np.array(h, dtype=np.uint8)
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Lambda(object):

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class Rotate(object):
    
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return itpl.rotate(img, self.angle, reshape=False, prefilter=False, order=0)

class HorizontalFlip(object):
    
    def __init__(self, should_flip):
        self.should_flip = should_flip

    def __call__(self, img):
        if not self.should_flip:
            return img
        return np.fliplr(img)

class RandomCrop(object):

    def __init__(self, h_offset, v_offset, output_size):
        self.h_offset = h_offset
        self.v_offset = v_offset
        self.output_size = output_size

    def __call__(self, img):
        # print("IMG SHAPE {}".format(img.shape))
        # print("h {} v {} output size {}".format(self.h_offset, self.v_offset, self.output_size))

        return img[self.h_offset: self.h_offset + self.output_size[0], self.v_offset: self.v_offset + self.output_size[1]]

class CenterCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        if len(img.shape) == 3:
            w, h, _ = img.shape
        else:
            w, h = img.shape 

        w_os, h_os = self.output_size
        i = int(round(w - self.output_size[0]) / 2)
        j = int(round(h - self.output_size[1]) / 2)
        return img[i:i + w_os,j:j + h_os]

class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation='nearest'):
        assert isinstance(size, int) or isinstance(size, float) or \
               (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        if img.ndim == 3:
            return misc.imresize(img, self.size, self.interpolation)
        elif img.ndim == 2:
            return misc.imresize(img, self.size, self.interpolation, 'F')
        else:
            RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))



class ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transforms = Compose(transforms)

        return transforms

    def __call__(self, img):
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        pil = Image.fromarray(img)
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return np.array(transform(pil))

class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    """

    def __call__(self, img):
        """Convert a ``numpy.ndarray`` to tensor.
        Args:
            img (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(img, np.ndarray):
            # handle numpy array
            if img.ndim == 3:
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
            elif img.ndim == 2:
                img = torch.from_numpy(img.copy())
            else:
                raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

            # backward compatibility
            # return img.float().div(255)
            return img.float()

