import datetime
import sys
import time

import cv2
import kornia
import numpy as np
import torch
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
from kornia.geometry import HomographyWarper
from matplotlib import pyplot as plt
from skimage import io
from torch import nn
from torchvision import transforms
from visdom import Visdom

def to_gray(image, inv_color=True, contrast_enhance=True, threshold=0, to_3_ch_gray=False):
    if image.dtype is not np.uint8:
        if np.max(image) <= 1:
            image = (image*255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    if image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    if inv_color:
        image_gray = cv2.bitwise_not(image_gray)

    if contrast_enhance:
        image_gray = cv2.equalizeHist(image_gray)

    image_gray[image_gray < threshold] = 0

    if to_3_ch_gray:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    return image_gray


def apply_mask(image, mask, bg_color=[0, 0, 0]):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if mask[x, y] == 0:
                image[x, y] = bg_color

    return image

def pixmap2np(pixmap: QPixmap):
    num_channels = pixmap.depth()//8
    image = pixmap.toImage()
    ib = image.bits()
    ib.setsize(image.height() * image.width() * num_channels)
    image_np = np.frombuffer(ib, np.uint8).reshape((pixmap.height(), pixmap.width(), num_channels))
    return image_np


def np2pixmap(image):
    if len(image.shape) == 2:
        depth = 1
        image_format = QImage.Format_Grayscale8
    else:
        if image.shape[2] == 3:
            depth = 3
            image_format = QImage.Format_RGB888
        else:
            depth = 4
            image_format = QImage.Format_RGBA8888

    if depth == 1:
        temp = QImage(image.copy(), image.shape[1], image.shape[0], image.shape[1], image_format)
    else:
        # image = np.transpose(image, (1, 0, 2)).copy()
        temp = QImage(image.copy(), image.shape[1], image.shape[0], image.shape[1]*depth, image_format)
    return QPixmap(temp)


def image2tensor(image, size=256):
    if type(size) == int:
        size = (size, size)

    if len(image.shape) == 2:
        ## single channel image
        transform = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor()])
    pilimage = Image.fromarray(image)
    if pilimage.mode == "RGBA":
        pilimage = pilimage.convert("RGB")

    pilimage = transform(pilimage)
    return pilimage.unsqueeze(0)


def tensor2image(tensor, alter_range=True):
    image = tensor[0].detach().cpu().float().numpy()
    if np.min(image) < -0.5:
        image = (image + 1.0)/2

    image = np.transpose(image, (1, 2, 0))

    if image.shape[2] == 1:
        image = image.squeeze(axis=2)

    return image


def warp_image(image, homo_matrix, size=256, inv_warp=False):
    if size is None:
        size = image.shape
    if type(size) == int:
        size = (size, size)

    homo_matrix_tensor = torch.Tensor(homo_matrix)
    homo_matrix_tensor = homo_matrix_tensor.unsqueeze(0)
    if inv_warp:
        homo_matrix_tensor = torch.inverse(homo_matrix_tensor)
    image_tensor = image2tensor(image)
    warp = HomographyWarper(size[0], size[1])
    image_warped = warp(image_tensor, homo_matrix_tensor)
    image_warped = tensor2image(image_warped)
    image_warped = (image_warped*255).astype(np.uint8)
    return image_warped


def set_bg_colour_histo(patch, threshold=0.8, bg_colour=None):
    if bg_colour is None:
        bg_colour = [0, 0, 0]

    if patch.shape[2] > 3:
        patch_alpha = patch[:, :, 3].astype(np.float32)
        patch_alpha /= 255.0
        patch = patch[:, :, :3]
    else:
        patch_alpha = None

    gray = ((patch[:, :, 0] * 0.2989 + patch[:, :, 1] * 0.5870 + patch[:, :, 2] * 0.1140)/255).astype(np.float32)
    mask = 1 - (gray > threshold).astype(np.uint8)
    mask_bg_colour = np.dstack(((1 - mask) * bg_colour[0], (1 - mask) * bg_colour[1], (1 - mask) * bg_colour[2]))
    patch = patch / 255.0 * np.dstack((mask, mask, mask)) + mask_bg_colour

    if patch_alpha is not None:
        patch = np.concatenate((patch, np.reshape(patch_alpha, (patch_alpha.shape[0], patch_alpha.shape[1], 1))), axis=2)
    return (patch*255).astype(np.uint8), mask


class Logger():
    def __init__(self, vis_name="main"):
        self.viz = Visdom(env=vis_name)
        self.image_windows = {}

    def log(self, images=None, n_images_disp=4):
        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.images(self._tensor2image(tensor.data, n_images_disp),
                                                                 nrow=n_images_disp, opts={'title': image_name})
            else:
                self.viz.images(tensor2image(tensor.data, n_images_disp), win=self.image_windows[image_name],
                                nrow=n_images_disp,
                                opts={'title': image_name})

    def _tensor2image(self, tensor, n_images_disp=4):
        images = []
        for i in range(n_images_disp):
            image = 127.5 * (tensor[i].cpu().float().numpy() + 1.0)
            if image.shape[0] == 1:
                image = np.tile(image, (3, 1, 1))

            images.append(image.astype(np.uint8))

        return images


if __name__ == "__main__":




    pass