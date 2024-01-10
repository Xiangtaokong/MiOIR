import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import numpy as np
import torch


@DATASET_REGISTRY.register()
class MiODateset(data.Dataset):
    """
    Dataset used for Real-ESRGAN model.
    """

    def __init__(self, opt):
        super(MiODateset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        with open(self.opt['meta_info']) as fin:
            # paths = [line.strip() for line in fin]
            # self.paths_gt = paths
            # self.paths_lq = [path.replace('GT','LQ') for path in paths]
            # #self.paths_lq = [path.replace('GT', 'LQ_noise') for path in paths]

            paths = [line.strip() for line in fin]
            self.paths_lq = paths
            self.paths_gt = []
            for path in paths:
                if 'snow_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('snow_', 'GT_'), 'snow'])
                elif 'rain_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('rain_', 'GT_'), 'rain'])
                elif 'haze_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('haze_', 'GT_'), 'haze'])
                elif 'blur_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('blur_', 'GT_'), 'blur'])
                elif 'noise_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('noise_', 'GT_'), 'noise'])
                elif 'jpeg_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('jpeg_', 'GT_'), 'jpeg'])
                elif 'dark_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('dark_', 'GT_'), 'dark'])
                elif 'sr_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('sr_', 'GT_'), 'sr'])
                else:
                    print(ddd)
                    self.paths_gt.append([path, 'clean'])




    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths_gt[index][0]
        img_type = self.paths_gt[index][1]
        lq_path = self.paths_lq[index]


        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes_gt = self.file_client.get(gt_path, 'gt')
                img_bytes_lq = self.file_client.get(lq_path, 'lq')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes_gt, float32=True)
        img_lq = imfrombytes(img_bytes_lq, float32=True)

        # -------------------- augmentation for training: flip, rotation -------------------- #
        gt_size = self.opt['gt_size']
        scale = self.opt['scale']
        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]


        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path,'img_type_word':img_type}

    def __len__(self):
        return len(self.paths_gt)

@DATASET_REGISTRY.register()
class MiO_EP_Dateset(data.Dataset):
    """
    Dataset used for Real-ESRGAN model.
    """

    def __init__(self, opt):
        super(MiO_EP_Dateset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        type_npl = np.load('/home/notebook/data/personal/S9053103/CL_code/type7.npy').astype(np.float32)
        self.typep = torch.from_numpy(type_npl).clone() #bnjrhd 012345
        print(self.typep)

        with open(self.opt['meta_info']) as fin:
            # paths = [line.strip() for line in fin]
            # self.paths_gt = paths
            # self.paths_lq = [path.replace('GT','LQ') for path in paths]
            # #self.paths_lq = [path.replace('GT', 'LQ_noise') for path in paths]

            paths = [line.strip() for line in fin]
            self.paths_lq = paths
            self.paths_gt = []
            for path in paths:
                if 'snow_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('snow_', 'GTv_'), 'snow'])
                elif 'rain_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('rain_', 'GT_'), 'rain'])
                elif 'haze_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('haze_', 'GT_'), 'haze'])
                elif 'blur_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('blur_', 'GT_'), 'blur'])
                elif 'noise_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('noise_', 'GT_'), 'noise'])
                elif 'jpeg_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('jpeg_', 'GT_'), 'jpeg'])
                elif 'dark_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('dark_', 'GT_'), 'dark'])
                elif 'sr_' in path:
                    self.paths_gt.append([path.replace('LQ_sub', 'GT_sub').replace('sr_', 'GT_'), 'sr'])
                else:
                    print(ddd)
                    self.paths_gt.append([path, 'clean'])


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths_gt[index][0]
        lq_path = self.paths_lq[index]
        img_type=self.paths_gt[index][1]
        if 'snow' in img_type:
            print(ddd)
        elif 'rain' in img_type:
            img_typee=self.typep[3]
        elif 'haze' in img_type:
            img_typee=self.typep[4]
        elif 'blur' in img_type:
            img_typee = self.typep[0]
        elif 'noise' in img_type:
            img_typee = self.typep[1]
        elif 'jpeg' in img_type:
            img_typee=self.typep[2]
        elif 'dark' in img_type:
            img_typee=self.typep[5]
        elif 'sr' in img_type:
            img_typee=self.typep[6]


        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes_gt = self.file_client.get(gt_path, 'gt')
                img_bytes_lq = self.file_client.get(lq_path, 'lq')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes_gt, float32=True)
        img_lq = imfrombytes(img_bytes_lq, float32=True)

        # -------------------- augmentation for training: flip, rotation -------------------- #
        gt_size = self.opt['gt_size']
        scale = self.opt['scale']
        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]


        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path,'img_type':img_typee,'img_type_word':img_type}

    def __len__(self):
        return len(self.paths_gt)


@DATASET_REGISTRY.register()
class ClassifierDataset(data.Dataset):
    """
    Dataset used for Real-ESRGAN model.
    """

    def __init__(self, opt):
        super(ClassifierDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        with open(self.opt['meta_info']) as fin:
            # paths = [line.strip() for line in fin]
            # self.paths_gt = paths
            # self.paths_lq = [path.replace('GT','LQ') for path in paths]
            # #self.paths_lq = [path.replace('GT', 'LQ_noise') for path in paths]

            paths = [line.strip() for line in fin]
            self.paths_lq = paths

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.

        lq_path = self.paths_lq[index]


        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes_lq = self.file_client.get(lq_path, 'lq')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        img_lq = imfrombytes(img_bytes_lq, float32=True)

        # -------------------- augmentation for training: flip, rotation -------------------- #
        gt_size = self.opt['gt_size']
        scale = self.opt['scale']
        # random crop
        _, img_lq = paired_random_crop(img_lq, img_lq, gt_size, scale)
        _, img_lq = augment([img_lq, img_lq], self.opt['use_flip'], self.opt['use_rot'])


        # BGR to RGB, HWC to CHW, numpy to tensor
        if 'snow_' in lq_path:
            print(ddd)
        elif 'rain_' in lq_path:
            label=torch.Tensor([0,0,0,1,0,0,0])
        elif 'haze_' in lq_path:
            label=torch.Tensor([0,0,0,0,1,0,0])
        elif 'blur_' in lq_path:
            label=torch.Tensor([1,0,0,0,0,0,0])
        elif 'noise_' in lq_path:
            label=torch.Tensor([0,1,0,0,0,0,0])
        elif 'jpeg_' in lq_path:
            label=torch.Tensor([0,0,1,0,0,0,0])
        elif 'dark_' in lq_path:
            label=torch.Tensor([0,0,0,0,0,1,0])
        elif 'sr_' in lq_path:
            label=torch.Tensor([0,0,0,0,0,0,1])
        else:
            print(ddd)
        
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]


        return {'lq': img_lq, 'label': label, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths_lq)



