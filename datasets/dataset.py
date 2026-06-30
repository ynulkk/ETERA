import torch
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

class EEG_Dataset(Dataset):
    """
    Args:
        images_path: Path of the EEG signal file.
        image_height (int): Height of spatial map in the EEG signal
        image_width (int): Width of spatial map in the EEG signal.
        transform:
    """

    def __init__(self, images_path: list, image_height: int, image_width: int, num_classes=3, feature='DE_PSD', map_type='SST', transform=None, return_raw=False, cfbm_smoothing=None):
        self.images_path = images_path
        self.image_height = image_height
        self.image_width = image_width
        self.feature = feature
        self.map_type = map_type
        self.transform = transform
        self.num_classes = num_classes
        self.return_raw = return_raw
        self.cfbm_smoothing = cfbm_smoothing or {}
        if map_type == 'SST':
            self.data_1D_to_2D = self._data_1D_to_2D_

    def __len__(self):
        return len(self.images_path)

    def _data_1D_to_2D_(self, data_1D, X=9, Y=9, size=None):
        """ Z-Score """
        data_1D = (data_1D - np.mean(data_1D)) / np.std(data_1D)

        data_2D = np.zeros([X, Y])
        data_2D[0, 3:6] = data_1D[0:3]
        data_2D[1, 3], data_2D[1, 5] = data_1D[3], data_1D[4]
        for i in range(5):
            data_2D[i + 2, :] = data_1D[5 + i * 9:5 + (i + 1) * 9]
        data_2D[7, 1:8] = data_1D[50:57]
        data_2D[8, 2:7] = data_1D[57:62]

        if not size:
            size = (self.image_height, self.image_width)

        
        data = np.array(Image.fromarray(data_2D).resize(size, resample=Image.BICUBIC))
        #data = (data - np.mean(data)) / np.std(data)
        return data

    def _data_1D_to_2D_4DaNN(self, data_1D, X=19, Y=19, size=None):
        #data_1D = (data_1D - np.mean(data_1D)) / np.std(data_1D)

        data_2D = np.zeros([X, Y])

        data_2D[0, 7], data_2D[0, 9], data_2D[0, 11] = data_1D[0], data_1D[1], data_1D[2]
        data_2D[2, 5], data_2D[2, 13] = data_1D[3], data_1D[4]
        for i in range(5):
            for j in range(9):
                data_2D[2 * i + 4, 1 + 2 * j] = data_1D[5 + i * 9 + j]

        for i in range(7):
            data_2D[14, 3 + 2 * i] = data_1D[50 + i]

        #for i in range(5):
        #    data_2D[16, 5 + 2 * i] = data_1D[57 + i]

        data_2D[16, 5], data_2D[16, 13] = data_1D[57], data_1D[61]
        data_2D[18, 7], data_2D[18, 9], data_2D[18, 11] = data_1D[58], data_1D[59], data_1D[60]

        if not size:
            size = (self.image_height, self.image_width)

        data = np.array(Image.fromarray(data_2D).resize(size, resample=Image.BICUBIC))
        data = (data - np.mean(data)) / np.std(data)
        return data

    def _smooth_spatial_2d(self, img, alpha):
        if alpha <= 0:
            return img
        center = img
        count = np.ones_like(center)
        acc = center.copy()
        acc[..., 1:, :] += center[..., :-1, :]
        count[..., 1:, :] += 1
        acc[..., :-1, :] += center[..., 1:, :]
        count[..., :-1, :] += 1
        acc[..., :, 1:] += center[..., :, :-1]
        count[..., :, 1:] += 1
        acc[..., :, :-1] += center[..., :, 1:]
        count[..., :, :-1] += 1
        neigh = acc / np.maximum(count, 1)
        return (1.0 - alpha) * center + alpha * neigh

    def _smooth_frequency_groups(self, img, alpha):
        if alpha <= 0 or img.shape[1] < 3:
            return img
        out = img.copy()
        groups = [(0, img.shape[1] // 2), (img.shape[1] // 2, img.shape[1])] if img.shape[1] % 2 == 0 else [(0, img.shape[1])]
        for start, end in groups:
            if end - start < 3:
                continue
            band = img[:, start:end, :, :]
            out[:, start + 1:end - 1, :, :] = (
                (1.0 - alpha) * band[:, 1:-1, :, :]
                + 0.5 * alpha * (band[:, :-2, :, :] + band[:, 2:, :, :])
            )
        return out

    def _apply_cfbm_smoothing(self, img):
        cfg = self.cfbm_smoothing or {}
        if not cfg.get('enabled', False):
            return img
        img = img.astype(np.float32, copy=False)
        freq_alpha = float(cfg.get('freq_alpha', 0.0))
        spatial_alpha = float(cfg.get('spatial_alpha', 0.0))
        norm_after = bool(cfg.get('norm_after', False))
        img = self._smooth_frequency_groups(img, freq_alpha)
        img = self._smooth_spatial_2d(img, spatial_alpha)
        if norm_after:
            mean = np.mean(img, axis=(-2, -1), keepdims=True)
            std = np.std(img, axis=(-2, -1), keepdims=True)
            img = (img - mean) / np.maximum(std, 1e-6)
        return img

    def __getitem__(self, item):
        """ data = [image_frames, image_channels, num_channels] """
        data = np.load(self.images_path[item])
        t, c, d = data.shape[0], data.shape[1], data.shape[2]
        raw = data.copy()

        label = os.path.splitext(self.images_path[item])[0].split('_')[-1]
        label = torch.as_tensor(int(label))
        labels = F.one_hot(label, num_classes=self.num_classes)

        data = data.reshape([-1, d])
        if self.map_type == 'SST':
            img = np.array([self._data_1D_to_2D_(x) for x in data])
        else:
            img = np.array([self._data_1D_to_2D_4DaNN(x) for x in data])
        img = img.reshape(t, c, self.image_height, self.image_width)
        if self.feature == 'DE':
            img = img[:, 0::2, :, :]
            raw = raw[:, 0::2, :]
        elif self.feature == 'PSD':
            img = img[:, 1::2, :, :]
            raw = raw[:, 1::2, :]
        elif self.feature == "Conv_Fusion":
            img = img
        else:
            indices = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
            #indices = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
            #indices = [2, 4, 6, 8, 10, 3, 5, 7, 9, 11]
            img = img[:, indices, :]
            raw = raw[:, indices, :]

        img = self._apply_cfbm_smoothing(img)

        if self.return_raw:
            return (img, raw), labels
        return img, labels

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        # images = {tuple:batch}
        # images[0] = ndarray:[image_frames, image_channels, image_height, image_width]

        images = np.array(images)
        images = torch.as_tensor(images)

        labels = np.array(labels)
        labels = torch.as_tensor(labels)
        return images, labels
