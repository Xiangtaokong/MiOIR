import torch
from torch import nn as nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import functional as F
import math
from copy import deepcopy
import os

from basicsr.archs import build_network
from basicsr.losses import build_loss,l1_loss_type
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch
import numpy as np

##############################################################################################

# Main models are the IRModels, following models are extended from the IR model with only the test() modified (Including SwinIR, Restormer, Uformer and PromptIR).

# _EP gets the explicit degradation type by the data path. _EP_class uses classifier to obtain the explicit prompt.
# We use the _EP_class model for over paper. (In fact, because the classification accuracy is so high, the difference between them is very small)

##############################################################################################


@MODEL_REGISTRY.register()
class IRModel(BaseModel):
    """Base IR model, it will recored 7 type of losses."""

    def __init__(self, opt):
        super(IRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        self.loss_blur=torch.Tensor([100])[0].to(self.device)
        self.loss_noise=torch.Tensor([100])[0].to(self.device)
        self.loss_jpeg=torch.Tensor([100])[0].to(self.device)
        self.loss_rain=torch.Tensor([100])[0].to(self.device)
        self.loss_haze=torch.Tensor([100])[0].to(self.device)
        self.loss_dark=torch.Tensor([100])[0].to(self.device)
        self.loss_sr=torch.Tensor([100])[0].to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pixtype = l1_loss_type().to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        if self.is_train:
            self.typeword=data['img_type_word']
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            if current_iter %100 ==0:

                loss_list=self.cri_pixtype(self.output, self.gt,self.typeword)

                if loss_list[0] != 0:
                    self.loss_blur = loss_list[0]

                if loss_list[1] != 0:
                    self.loss_noise = loss_list[1]

                if loss_list[2] != 0:
                    self.loss_jpeg = loss_list[2]

                if loss_list[3] != 0:
                    self.loss_rain = loss_list[3]

                if loss_list[4] != 0:
                    self.loss_haze = loss_list[4]

                if loss_list[5] != 0:
                    self.loss_dark = loss_list[5]
                
                if loss_list[6] != 0:
                    self.loss_sr = loss_list[6]
                

                loss_dict['loss_blur'] = self.loss_blur
                loss_dict['loss_noise'] = self.loss_noise
                loss_dict['loss_jpeg'] = self.loss_jpeg
                loss_dict['loss_rain'] = self.loss_rain
                loss_dict['loss_haze'] = self.loss_haze
                loss_dict['loss_dark'] = self.loss_dark
                loss_dict['loss_sr'] = self.loss_sr
                

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def tile_test(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """

        _, _, H_ori, W_ori = self.lq.shape

        window_size = 128
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.tile_size = 128
        self.tile_pad = 0
        self.scale = 1

        self.img = img
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                # try:
                with torch.no_grad():
                    output_tile = self.net_g([input_tile,self.img_typee.to(self.device)])
                # except RuntimeError as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

        self.output = self.output[:, :, :H_ori, :W_ori]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train=False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            #self.tile_test()
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
                # if '001' in save_img_path or '0801' in save_img_path or 'haze' in save_img_path:
                #     imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train=True

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class IRModel_EP(BaseModel):
    """Base explicit prompt IR model."""

    def __init__(self, opt):
        super(IRModel_EP, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        type_npl = np.load(os.path.join(os.getcwd(),'data/type7.npy')).astype(np.float32)
        self.typep = torch.from_numpy(type_npl).clone()  # bnjrhd 012345
        self.img_typee=0

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        self.loss_blur=torch.Tensor([100])[0].to(self.device)
        self.loss_noise=torch.Tensor([100])[0].to(self.device)
        self.loss_jpeg=torch.Tensor([100])[0].to(self.device)
        self.loss_rain=torch.Tensor([100])[0].to(self.device)
        self.loss_haze=torch.Tensor([100])[0].to(self.device)
        self.loss_dark=torch.Tensor([100])[0].to(self.device)
        self.loss_sr=torch.Tensor([100])[0].to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pixtype = l1_loss_type().to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        if self.is_train:
            self.typeword=data['img_type_word']
            self.lq = [data['lq'].to(self.device),data['img_type'].to(self.device)]
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
        else:
            self.lq = data['lq'].to(self.device)
            self.lq_path = data['lq_path']
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)


        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            if current_iter %100 ==0:

                loss_list=self.cri_pixtype(self.output, self.gt,self.typeword)

                if loss_list[0] != 0:
                    self.loss_blur = loss_list[0]

                if loss_list[1] != 0:
                    self.loss_noise = loss_list[1]

                if loss_list[2] != 0:
                    self.loss_jpeg = loss_list[2]

                if loss_list[3] != 0:
                    self.loss_rain = loss_list[3]

                if loss_list[4] != 0:
                    self.loss_haze = loss_list[4]

                if loss_list[5] != 0:
                    self.loss_dark = loss_list[5]
                
                if loss_list[6] != 0:
                    self.loss_sr = loss_list[6]
                
                

                loss_dict['loss_blur'] = self.loss_blur
                loss_dict['loss_noise'] = self.loss_noise
                loss_dict['loss_jpeg'] = self.loss_jpeg
                loss_dict['loss_rain'] = self.loss_rain
                loss_dict['loss_haze'] = self.loss_haze
                loss_dict['loss_dark'] = self.loss_dark
                loss_dict['loss_sr'] = self.loss_sr
                


        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     with torch.no_grad():
        #         self.output = self.net_g_ema(self.lq)
        # else:
        
        if 'snow' in self.lq_path:
            print(ddd)
        elif '_r.' in self.lq_path[0]:
            self.img_typee=self.typep[3]
        elif '_h.' in self.lq_path[0]:
            self.img_typee=self.typep[4]
        elif '_b.' in self.lq_path[0]:
            self.img_typee = self.typep[0]
        elif '_n.' in self.lq_path[0]:
            self.img_typee = self.typep[1]
        elif '_j.' in self.lq_path[0]:
            self.img_typee=self.typep[2]
        elif '_d.' in self.lq_path[0]:
            self.img_typee=self.typep[5]
        elif '_s.' in self.lq_path[0]:
            self.img_typee=self.typep[6]


        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([self.lq,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()

    def tile_test(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """

        _, _, H_ori, W_ori = self.lq.shape

        window_size = 128
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.tile_size = 128
        self.tile_pad = 0
        self.scale = 1

        self.img = img
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                # try:
                with torch.no_grad():
                    output_tile = self.net_g([input_tile,self.img_typee.to(self.device)])
                # except RuntimeError as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

        self.output = self.output[:, :, :H_ori, :W_ori]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train=False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        self.img_typee=0

        if 'snow' in dataset_name:
            print(ddd)
        elif 'rain' in dataset_name:
            self.img_typee=self.typep[3]
        elif 'haze' in dataset_name:
            self.img_typee=self.typep[4]
        elif 'blur' in dataset_name:
            self.img_typee = self.typep[0]
        elif 'noise' in dataset_name:
            self.img_typee = self.typep[1]
        elif 'jpeg' in dataset_name:
            self.img_typee=self.typep[2]
        elif 'dark' in dataset_name:
            self.img_typee=self.typep[5]
        elif 'sr' in dataset_name:
            self.img_typee=self.typep[6]
        #self.img_typee = self.typep[3]

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            #self.tile_test()
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
                # if '001' in save_img_path or '0801' in save_img_path or 'haze' in save_img_path:
                #     imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train=True
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class IRModel_EP_class(BaseModel):
    """Base explicit prompt IR model with classifier."""

    def __init__(self, opt):
        super(IRModel_EP_class, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.net_c = build_network(opt['network_c'])
        self.net_c = self.model_to_device(self.net_c)
        self.print_network(self.net_c)

        type_npl = np.load(os.path.join(os.getcwd(),'data/type7.npy')).astype(np.float32)
        self.typep = torch.from_numpy(type_npl).clone()  # bnjrhd 012345
        self.img_typee=0

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        
        load_path_c = self.opt['path_c'].get('pretrain_network_g', None)
        if load_path_c is not None:
            param_key = self.opt['path_c'].get('param_key_g', 'params')
            self.load_network(self.net_c, load_path_c, self.opt['path_c'].get('strict_load_g', True), 'param_key_g')

        if self.is_train:
            self.init_training_settings()

        self.loss_blur=torch.Tensor([100])[0].to(self.device)
        self.loss_noise=torch.Tensor([100])[0].to(self.device)
        self.loss_jpeg=torch.Tensor([100])[0].to(self.device)
        self.loss_rain=torch.Tensor([100])[0].to(self.device)
        self.loss_haze=torch.Tensor([100])[0].to(self.device)
        self.loss_dark=torch.Tensor([100])[0].to(self.device)
        self.loss_sr=torch.Tensor([100])[0].to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pixtype = l1_loss_type().to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        if self.is_train:
            self.typeword=data['img_type_word']
            self.lq = [data['lq'].to(self.device),data['img_type'].to(self.device)]
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
        else:
            self.lq = data['lq'].to(self.device)
            self.lq_path = data['lq_path']
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # self.count+=1
        #
        # if self.count==156:
        #
        #     imwrite(tensor2img(self.gt.cpu()), '/home/xtkong/GLV/GLV-training/experiments/gt.png')
        #     imwrite(tensor2img(self.lq.cpu()), '/home/xtkong/GLV/GLV-training/experiments/lq.png')
        #     print(ddd)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            if current_iter %100 ==0:

                loss_list=self.cri_pixtype(self.output, self.gt,self.typeword)

                if loss_list[0] != 0:
                    self.loss_blur = loss_list[0]

                if loss_list[1] != 0:
                    self.loss_noise = loss_list[1]

                if loss_list[2] != 0:
                    self.loss_jpeg = loss_list[2]

                if loss_list[3] != 0:
                    self.loss_rain = loss_list[3]

                if loss_list[4] != 0:
                    self.loss_haze = loss_list[4]

                if loss_list[5] != 0:
                    self.loss_dark = loss_list[5]
                
                if loss_list[6] != 0:
                    self.loss_sr = loss_list[6]
                
                

                loss_dict['loss_blur'] = self.loss_blur
                loss_dict['loss_noise'] = self.loss_noise
                loss_dict['loss_jpeg'] = self.loss_jpeg
                loss_dict['loss_rain'] = self.loss_rain
                loss_dict['loss_haze'] = self.loss_haze
                loss_dict['loss_dark'] = self.loss_dark
                loss_dict['loss_sr'] = self.loss_sr
                


        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     with torch.no_grad():
        #         self.output = self.net_g_ema(self.lq)
        # else:
        self.net_g.eval()
        with torch.no_grad():
            self.output_c = self.net_c(self.lq).detach().cpu()
            mask = (self.output_c == self.output_c.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
            # result=torch.mul(mask,result)

            if 'snow' in self.lq_path:
                print(ddd)
            elif mask[0][3]: #r
                self.img_typee=self.typep[3]
            elif mask[0][4]: #h
                self.img_typee=self.typep[4]
            elif mask[0][0]: #b
                self.img_typee = self.typep[0]
            elif mask[0][1]: #n
                self.img_typee = self.typep[1]
            elif mask[0][2]: #j
                self.img_typee=self.typep[2]
            elif mask[0][5]: #d
                self.img_typee=self.typep[5]
            elif mask[0][6]: #s
                self.img_typee=self.typep[6]
            else:
                print(ddd)

            self.output = self.net_g([self.lq,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()

    def tile_test(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """

        _, _, H_ori, W_ori = self.lq.shape

        window_size = 128
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.tile_size = 128
        self.tile_pad = 0
        self.scale = 1

        self.img = img
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                # try:
                with torch.no_grad():
                    output_tile = self.net_g([input_tile,self.img_typee.to(self.device)])
                # except RuntimeError as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

        self.output = self.output[:, :, :H_ori, :W_ori]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train=False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        self.img_typee=0


        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            #self.tile_test()
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
                # if '001' in save_img_path or '0801' in save_img_path or 'haze' in save_img_path:
                #     imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train=True
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class IRModel_AP(BaseModel):
    """Base adaptive prompt IR model.."""

    def __init__(self, opt):
        super(IRModel_AP, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        self.loss_blur=torch.Tensor([100])[0].to(self.device)
        self.loss_noise=torch.Tensor([100])[0].to(self.device)
        self.loss_jpeg=torch.Tensor([100])[0].to(self.device)
        self.loss_rain=torch.Tensor([100])[0].to(self.device)
        self.loss_haze=torch.Tensor([100])[0].to(self.device)
        self.loss_dark=torch.Tensor([100])[0].to(self.device)
        self.loss_sr=torch.Tensor([100])[0].to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pixtype = l1_loss_type().to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        if self.is_train:
            self.typeword=data['img_type_word']
            self.lq = [data['lq'].to(self.device),data['lq'].to(self.device)]
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # self.count+=1
        #
        # if self.count==156:
        #
        #     imwrite(tensor2img(self.gt.cpu()), '/home/xtkong/GLV/GLV-training/experiments/gt.png')
        #     imwrite(tensor2img(self.lq.cpu()), '/home/xtkong/GLV/GLV-training/experiments/lq.png')
        #     print(ddd)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            if current_iter %100 ==0:

                loss_list=self.cri_pixtype(self.output, self.gt,self.typeword)

                if loss_list[0] != 0:
                    self.loss_blur = loss_list[0]

                if loss_list[1] != 0:
                    self.loss_noise = loss_list[1]

                if loss_list[2] != 0:
                    self.loss_jpeg = loss_list[2]

                if loss_list[3] != 0:
                    self.loss_rain = loss_list[3]

                if loss_list[4] != 0:
                    self.loss_haze = loss_list[4]

                if loss_list[5] != 0:
                    self.loss_dark = loss_list[5]
                
                if loss_list[6] != 0:
                    self.loss_sr = loss_list[6]
                
                

                loss_dict['loss_blur'] = self.loss_blur
                loss_dict['loss_noise'] = self.loss_noise
                loss_dict['loss_jpeg'] = self.loss_jpeg
                loss_dict['loss_rain'] = self.loss_rain
                loss_dict['loss_haze'] = self.loss_haze
                loss_dict['loss_dark'] = self.loss_dark
                loss_dict['loss_sr'] = self.loss_sr
            


        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     with torch.no_grad():
        #         self.output = self.net_g_ema(self.lq)
        # else:
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([self.lq,self.lq])
        self.net_g.train()

    def tile_test(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """

        _, _, H_ori, W_ori = self.lq.shape

        window_size = 128
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.tile_size = 128
        self.tile_pad = 0
        self.scale = 1

        self.img = img
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                # try:
                with torch.no_grad():
                    output_tile = self.net_g([input_tile,input_tile])
                # except RuntimeError as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

        self.output = self.output[:, :, :H_ori, :W_ori]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train=False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')


        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            #self.tile_test()
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
                # if '001' in save_img_path or '0801' in save_img_path or 'haze' in save_img_path:
                #     imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train=True
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
            # if 'cond' in k:
            #     load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=False)

##############################################################################################

# The following models are extended from the IR model with only the test() modified.
# Including SwinIR, Restormer, Uformer and PromptIR
# The PromptIR-EP or -AP are the Restormer_EP or -AP, because PromptIR is modified from Restormer.

##############################################################################################

# SwinIR:

@MODEL_REGISTRY.register()
class SwinIR(IRModel):
      
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(img)
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class SwinIR_EP(IRModel_EP):

    def test(self):
        if 'snow' in self.lq_path:
            print(ddd)
        elif '_r.' in self.lq_path[0]:
            self.img_typee=self.typep[3]
        elif '_h.' in self.lq_path[0]:
            self.img_typee=self.typep[4]
        elif '_b.' in self.lq_path[0]:
            self.img_typee = self.typep[0]
        elif '_n.' in self.lq_path[0]:
            self.img_typee = self.typep[1]
        elif '_j.' in self.lq_path[0]:
            self.img_typee=self.typep[2]
        elif '_d.' in self.lq_path[0]:
            self.img_typee=self.typep[5]
        elif '_s.' in self.lq_path[0]:
            self.img_typee=self.typep[6]

        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.net_g.eval()
        with torch.no_grad():
            #self.output = self.net_g(img)
            self.output = self.net_g([img,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class SwinIR_EP_class(IRModel_EP_class):

    def test(self):

        if 'snow' in self.lq_path:
            print(ddd)
        elif '_r.' in self.lq_path[0]:
            self.img_typee=self.typep[3]
        elif '_h.' in self.lq_path[0]:
            self.img_typee=self.typep[4]
        elif '_b.' in self.lq_path[0]:
            self.img_typee = self.typep[0]
        elif '_n.' in self.lq_path[0]:
            self.img_typee = self.typep[1]
        elif '_j.' in self.lq_path[0]:
            self.img_typee=self.typep[2]
        elif '_d.' in self.lq_path[0]:
            self.img_typee=self.typep[5]
        elif '_s.' in self.lq_path[0]:
            self.img_typee=self.typep[6]

        self.output_c = self.net_c(self.lq).detach().cpu()
        mask = (self.output_c == self.output_c.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
        # result=torch.mul(mask,result)

        if 'snow' in self.lq_path:
            print(ddd)
        elif mask[0][3]: #r
            self.img_typee=self.typep[3]
        elif mask[0][4]: #h
            self.img_typee=self.typep[4]
        elif mask[0][0]: #b
            self.img_typee = self.typep[0]
        elif mask[0][1]: #n
            self.img_typee = self.typep[1]
        elif mask[0][2]: #j
            self.img_typee=self.typep[2]
        elif mask[0][5]: #d
            self.img_typee=self.typep[5]
        elif mask[0][6]: #s
            self.img_typee=self.typep[6]
        else:
            print(ddd)

        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     with torch.no_grad():
        #         self.output = self.net_g_ema(img)
        # else:
        self.net_g.eval()
        with torch.no_grad():
            #self.output = self.net_g(img)
            self.output = self.net_g([img,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class SwinIR_AP(IRModel_AP):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([img,img])
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

##############################################################################################

# Restormer:

@MODEL_REGISTRY.register()
class Restormer(IRModel):
    def test(self):

        # pad to multiplication of window_size
        window_size = 64
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        B, C, H, W = img.shape

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(img)
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class Restormer_EP(IRModel_EP):
    def test(self):

        if 'snow' in self.lq_path:
            print(ddd)
        elif '_r.' in self.lq_path[0]:
            self.img_typee=self.typep[3]
        elif '_h.' in self.lq_path[0]:
            self.img_typee=self.typep[4]
        elif '_b.' in self.lq_path[0]:
            self.img_typee = self.typep[0]
        elif '_n.' in self.lq_path[0]:
            self.img_typee = self.typep[1]
        elif '_j.' in self.lq_path[0]:
            self.img_typee=self.typep[2]
        elif '_d.' in self.lq_path[0]:
            self.img_typee=self.typep[5]
        elif '_s.' in self.lq_path[0]:
            self.img_typee=self.typep[6]

        # pad to multiplication of window_size
        window_size = 64
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        B, C, H, W = img.shape

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([img,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()
        
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class Restormer_EP_class(IRModel_EP_class):

    def test(self):
        self.output_c = self.net_c(self.lq).detach().cpu()
        mask = (self.output_c == self.output_c.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
        # result=torch.mul(mask,result)

        if 'snow' in self.lq_path:
            print(ddd)
        elif mask[0][3]: #r
            self.img_typee=self.typep[3]
        elif mask[0][4]: #h
            self.img_typee=self.typep[4]
        elif mask[0][0]: #b
            self.img_typee = self.typep[0]
        elif mask[0][1]: #n
            self.img_typee = self.typep[1]
        elif mask[0][2]: #j
            self.img_typee=self.typep[2]
        elif mask[0][5]: #d
            self.img_typee=self.typep[5]
        elif mask[0][6]: #s
            self.img_typee=self.typep[6]
        else:
            print(ddd)


        # pad to multiplication of window_size
        window_size = 64
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        B, C, H, W = img.shape
        #img, mask = expand2square(img, factor=128)

        
        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     with torch.no_grad():
        #         self.output = self.net_g_ema(img)
        # else:
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([img,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()
        
        # self.output = torch.masked_select(self.output, mask.bool()).reshape(1, 3, H, W)

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class Restormer_AP(IRModel_AP):

    def test(self):
        # pad to multiplication of window_size
        window_size = 64
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        B, C, H, W = img.shape

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([img,img])
        self.net_g.train()
        


        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


##############################################################################################

# Uformer:


@MODEL_REGISTRY.register()
class Uformer(IRModel):
    def test(self):

        # pad to multiplication of window_size
        window_size = 128
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, h, w=img.shape

        mod_pad_h2=0
        mod_pad_w2=0
        if w > h:
            mod_pad_h2=w-h
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2), int(mod_pad_h2/2)), 'reflect')
        else:
            mod_pad_w2=h-w
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2),int(mod_pad_h2/2)), 'reflect')

        B, C, H, W = img.shape
        _,_,h,w=img.shape


        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(img)
        self.net_g.train()
        
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, int(mod_pad_h2/2) : h - int(mod_pad_h2/2) -mod_pad_h , int(mod_pad_w2/2) :w - int(mod_pad_w2/2) -mod_pad_w ]

@MODEL_REGISTRY.register()
class Uformer_EP(IRModel_EP):
    def test(self):

        if 'snow' in self.lq_path:
            print(ddd)
        elif '_r.' in self.lq_path[0]:
            self.img_typee=self.typep[3]
        elif '_h.' in self.lq_path[0]:
            self.img_typee=self.typep[4]
        elif '_b.' in self.lq_path[0]:
            self.img_typee = self.typep[0]
        elif '_n.' in self.lq_path[0]:
            self.img_typee = self.typep[1]
        elif '_j.' in self.lq_path[0]:
            self.img_typee=self.typep[2]
        elif '_d.' in self.lq_path[0]:
            self.img_typee=self.typep[5]
        elif '_s.' in self.lq_path[0]:
            self.img_typee=self.typep[6]

        # pad to multiplication of window_size
        window_size = 128
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, h, w=img.shape

        mod_pad_h2=0
        mod_pad_w2=0
        if w > h:
            mod_pad_h2=w-h
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2), int(mod_pad_h2/2)), 'reflect')
        else:
            mod_pad_w2=h-w
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2),int(mod_pad_h2/2)), 'reflect')

        B, C, H, W = img.shape
        _,_,h,w=img.shape

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([img,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()
        

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, int(mod_pad_h2/2) : h - int(mod_pad_h2/2) -mod_pad_h , int(mod_pad_w2/2) :w - int(mod_pad_w2/2) -mod_pad_w ]

@MODEL_REGISTRY.register()
class Uformer_EP_class(IRModel_EP_class):
    def test(self):
        self.output_c = self.net_c(self.lq).detach().cpu()
        mask = (self.output_c == self.output_c.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
        # result=torch.mul(mask,result)

        if 'snow' in self.lq_path:
            print(ddd)
        elif mask[0][3]: #r
            self.img_typee=self.typep[3]
        elif mask[0][4]: #h
            self.img_typee=self.typep[4]
        elif mask[0][0]: #b
            self.img_typee = self.typep[0]
        elif mask[0][1]: #n
            self.img_typee = self.typep[1]
        elif mask[0][2]: #j
            self.img_typee=self.typep[2]
        elif mask[0][5]: #d
            self.img_typee=self.typep[5]
        elif mask[0][6]: #s
            self.img_typee=self.typep[6]
        else:
            print(ddd)

        # pad to multiplication of window_size
        window_size = 128
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, h, w=img.shape

        mod_pad_h2=0
        mod_pad_w2=0
        if w > h:
            mod_pad_h2=w-h
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2), int(mod_pad_h2/2)), 'reflect')
        else:
            mod_pad_w2=h-w
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2),int(mod_pad_h2/2)), 'reflect')

        B, C, H, W = img.shape
        _,_,h,w=img.shape

        #img, mask = expand2square(img, factor=128)

        # if hasattr(self, 'net_g_ema'):
        #     self.net_g_ema.eval()
        #     with torch.no_grad():
        #         self.output = self.net_g_ema(img)
        # else:
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([img,torch.unsqueeze(self.img_typee.to(self.device),0)])
        self.net_g.train()
        
        # self.output = torch.masked_select(self.output, mask.bool()).reshape(1, 3, H, W)

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, int(mod_pad_h2/2) : h - int(mod_pad_h2/2) -mod_pad_h , int(mod_pad_w2/2) :w - int(mod_pad_w2/2) -mod_pad_w ]


@MODEL_REGISTRY.register()
class Uformer_AP(IRModel_AP):
    def test(self):

        # pad to multiplication of window_size
        window_size = 128
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, h, w=img.shape

        mod_pad_h2=0
        mod_pad_w2=0
        if w > h:
            mod_pad_h2=w-h
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2), int(mod_pad_h2/2)), 'reflect')
        else:
            mod_pad_w2=h-w
            img = F.pad(img, (int(mod_pad_w2/2), int(mod_pad_w2/2), int(mod_pad_h2/2),int(mod_pad_h2/2)), 'reflect')

        B, C, H, W = img.shape
        _,_,h,w=img.shape

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g([img,img])
        self.net_g.train()
        
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, int(mod_pad_h2/2) : h - int(mod_pad_h2/2) -mod_pad_h , int(mod_pad_w2/2) :w - int(mod_pad_w2/2) -mod_pad_w ]

##############################################################################################

# PromptIR:

@MODEL_REGISTRY.register()
class PromptIR(IRModel):

    def test(self):

        # pad to multiplication of window_size
        window_size = 64
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        B, C, H, W = img.shape

        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(img)
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

##############################################################################################

# ClassModel:

@MODEL_REGISTRY.register()
class ClassModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(ClassModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.celoss=nn.CrossEntropyLoss()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_path = data['lq_path']
        if self.is_train:
            self.label = data['label'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.celoss(self.output, self.label)
            l_total += l_pix
            loss_dict['l_ce'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.label)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):

        print(self.lq_path)

        if 'snow' in self.lq_path:
            print(ddd)
        elif '_r.' in self.lq_path[0] or 'rain' in self.lq_path[0]:
            self.label='r'
        elif '_h.' in self.lq_path[0] or 'haze' in self.lq_path[0]:
            self.label='h'
        elif '_b.' in self.lq_path[0] or 'blur' in self.lq_path[0]:
            self.label='b'
        elif '_n.' in self.lq_path[0] or 'noise' in self.lq_path[0]:
            self.label='n'
        elif '_j.' in self.lq_path[0] or 'jpeg' in self.lq_path[0]:
            self.label='j'
        elif '_d.' in self.lq_path[0] or 'dark' in self.lq_path[0]:
            self.label='d'
        elif '_s.' in self.lq_path[0] or 'sr' in self.lq_path[0]:
            self.label='s'
        else:
            self.label='None'
        
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):

        self.is_train=False

        dataset_name = dataloader.dataset.opt['name']
        pbar = tqdm(total=len(dataloader), unit='image')

        data_num=0
        acc_num=0

        for idx, val_data in enumerate(dataloader):
            data_num+=1
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            result = visuals['result']
            
            label = visuals['label']
            del self.label

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            mask = (result == result.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
            # result=torch.mul(mask,result)

            if 'snow' in self.lq_path:
                print(ddd)
            elif mask[0][3]:
                result='r'
            elif mask[0][4]:
                result='h'
            elif mask[0][0]:
                result='b'
            elif mask[0][1]:
                result='n'
            elif mask[0][2]:
                result='j'
            elif mask[0][5]:
                result='d'
            elif mask[0][6]:
                result='s'
            else:
                print(ddd)

            if result!=label:
                print('wrong')
                print(result)
                print(label)
            else:
                acc_num+=1
            
            log_str = f'img {img_name}'
        
            log_str += f'\t # result: {result}'
            logger = get_root_logger()
            logger.info(log_str)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()
        acc_rate=acc_num/data_num
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger,acc_rate)

        self.is_train=True

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger,acc_rate):
        log_str = f'Validation {dataset_name}\n'
        
        log_str += f'\t # acc: {acc_rate:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        
        out_dict['label'] = self.label

        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)



