'''
Vimeo7 dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
# try:
#     import mc  # import memcached
# except ImportError:
#     pass

logger = logging.getLogger('base')

class Vimeo7Dataset(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(Vimeo7Dataset, self).__init__()
        self.opt = opt
        self.opt['use_time'] = True
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        self.half_N_frames = opt['N_frames'] // 2
        self.LR_N_frames = 1 + self.half_N_frames
        assert self.LR_N_frames > 1, 'Error: Not enough LR frames to interpolate'
        self.LR_index_list = [0, 6]
        # for i in range(self.LR_N_frames):
        #     self.LR_index_list.append(i*2)

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        self.LR_resolution = tuple(opt['LR_resolution'])
        self.HR_resolution = tuple(opt['HR_resolution'])
        #### directly load image keys
        if opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            cache_keys = opt['cache_keys']
        else:
            cache_keys = 'Vimeo7_train_keys.pkl'
        logger.info('Using cache keys - {}.'.format(cache_keys))
        self.paths_GT = list(pickle.load(open(cache_keys, 'rb'))['keys'])
     
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()

        scale = self.opt['scale']
        # print(scale)
        N_frames = self.opt['N_frames']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index] # key = self.paths_GT[index]
        name_a = key[:len(key) - (1 + len(key.split('_')[-1]))] # name_a, name_b = key.split('_')
        name_b = key.split('_')[-1]

        neighbor_list = [1,1,2,3,4,5,6,7,7]

        self.LQ_frames_list = [1,7]
        # for i in self.LR_index_list:
        #     self.LQ_frames_list.append(neighbor_list[i])
        
        #### get the GT image (as the center frame)
        img_GT_l = []
        for v in neighbor_list:
            if self.data_type == 'mc':
                img_GT = self._read_img_mc_BGR(self.GT_root, name_a, name_b, '{}.png'.format(v))
                img_GT = img_GT.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_GT = util.read_img(self.GT_env, key + '_{}'.format(v), self.HR_resolution)
            else:               
                img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b, 'im{}.png'.format(v)))
            img_GT_l.append(img_GT)
       #### get LQ images
        LQ_size_tuple = self.LR_resolution if self.LR_input else self.HR_resolution
        img_LQ_l = []
        for v in self.LQ_frames_list:
            if self.data_type == 'mc':
                img_LQ = self._read_img_mc(
                    osp.join(self.LQ_root, name_a, name_b, '/{}.png'.format(v)))
                img_LQ = img_LQ.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)
            else:
                img_LQ = util.read_img(None, osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LQ_l.append(img_LQ)

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GTs = np.stack(img_GT_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()


        if self.opt['use_time'] == True:
            time_list = []
            for i in neighbor_list[1:-1]:
                time_list.append(torch.Tensor([(i-1) / (len(neighbor_list[1:-1])-1)]))
            # time_Tensors = torch.cat(time_list)

            return {'LQs': img_LQs[[0, -1], :, :, :], 'GT': img_GTs, 'key': key, 'time': time_list}
        else:
            return {'LQs': img_LQs, 'GT': img_GTs, 'key': key}

    def __len__(self):
        return len(self.paths_GT)
