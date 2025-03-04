import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss
import random
import numpy as np

logger = logging.getLogger('base')


class VideoSRBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoSRBaseModel, self).__init__(opt)
        
        self.lrs = []

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
            
        # print network
        self.net_opt = opt['network_G']
        self.net_base = self.net_opt['which_model_G']
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)
            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if 'time' in data.keys():
            self.times = [t_.to(self.device) for t_ in data['time']]
        else:
            self.times = None
        if 'scale' in data.keys():
            self.scale = data['scale']
        else:
            self.scale = 4
        if 'test' in data.keys():
            self.testmode = data['test']
        else:
            self.testmode = False
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        if 'flow' in data.keys():
            if 'psies' in data:
                self.flows = [data['flow'], data['psies'], data['flow_GT']]
            else:
                self.flows = [data['flow'], data['flow_GT']]
        else:
            self.flows = None

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        ratio = max(0, 1-(step/150000))
        use_GT = (random.random()) < ratio
        ratio = max(0, 1-((step%150000)/150000))
        self.optimizer_G.zero_grad()
        if self.times is None:
            self.fake_H = self.netG(self.var_L)
        elif self.net_base == 'LIIF':
            self.fake_H = self.netG(self.var_L, self.times, self.scale)
        elif self.net_base == 'MoTIF':
            self.fake_H, flow, flow_GT = self.netG(self.var_L, self.real_H, self.times, self.scale, use_GT = use_GT, flows = self.flows)
            flow_l = self.cri_pix(flow, flow_GT)
        elif self.net_base == 'bfstvsr':
            self.fake_H, flow, flow_GT = self.netG(self.var_L, self.real_H, self.times, self.scale, use_GT = use_GT, flows = self.flows)
            flow_l = self.cri_pix(flow, flow_GT)
        else:
            self.fake_H = self.netG(self.var_L, self.times, self.scale)
        self.real_H = self.real_H[:,1:-1] 
        # self.fake_H = self.netG(self.var_L)
        
        if self.times is None or self.net_base == 'TMNet':
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        else:
            l_pix = 0
            for idx in range(len(self.times)):
                l_pix += self.l_pix_w * self.cri_pix(self.fake_H[idx], self.real_H[:, idx])
        
        if self.net_base == 'LIIF':
            l_pix*= (4./(self.fake_H[0].shape[3]/self.var_L.shape[3]))**2
        else:
            l_pix*= (4./(self.fake_H.shape[3]/self.var_L.shape[3]))**2
        
        if  ('MoTIF' in self.net_base or 'bfstvsr' in self.net_base):
            (l_pix + 0.1*flow_l*(ratio)).backward()
        else:
            l_pix.backward()
        # print("Training Loss: ", l_pix.item())
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.lrs.append(l_pix.item())
        self.log_dict['l_pixs'] = np.mean(self.lrs)
        if  ('MoTIF' in self.net_base or 'bfstvsr' in self.net_base):
            self.log_dict['flow_l'] = flow_l.item()
            self.log_dict['GT'] = use_GT
        torch.cuda.empty_cache()

    def test(self, output=False):
        self.netG.eval()
        with torch.no_grad():
            if self.net_base == 'EDVR':
                #self.fake_H = self.netG(torch.cat((self.var_L[:,1:], self.var_L), 1))
                self.fake_H = self.netG(self.var_L).unsqueeze(1).repeat(1,7,1,1,1)
                #self.fake_H = self.fake_H.unsqueeze(1).repeat(1,2,1,1,1)
            elif self.times is None:
                self.fake_H = self.netG(self.var_L)
            elif self.net_base == 'LIIF':
                # self.scale=1
                self.fake_H = self.netG(self.var_L, self.times, self.scale, test = False)
            elif self.net_base == 'bfstvsr':
                self.fake_H, flow, flow_GT = self.netG(self.var_L, self.real_H, self.times[:3], self.scale, use_GT = False, iter = 4)
                if len(self.times) != 3:
                    for l in range(3, len(self.times), 3):
                        tmp, flow, flow_GT = self.netG(self.var_L, None, self.times[l:l+3], self.scale, use_GT = False, iter = 4)
                        self.fake_H = torch.cat((self.fake_H,tmp), 0)
                self.flow = flow
                self.flow_GT = flow_GT

        self.netG.train()
        if output == True:
            return self.fake_H

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['restore'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                            self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G'] 
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
