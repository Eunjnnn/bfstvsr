'''
The code is modified from the implementation of Zooming Slow-Mo:
https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020
'''
import os

import numpy as np
import torch
torch.set_num_threads(2)
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import option
from utilss import util
from data import create_dataloader, create_dataset
from models import create_model
from pdb import set_trace as bp
import torchvision


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main(opt):
    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            print(opt['path']['experiments_root'])
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            pass
            '''
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
            '''
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    import numpy as np
    losses = []
    psnrs = []
    ssims = []
    psnrs_anchor = []
    psnrs_inter = []
    psnrs_center = []
    psnrs_all = []
    ssim_all = []
    flows = []
    flows_0 = []
    for epoch in range(1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for iter_id, train_data in enumerate(train_loader):
            current_step += 1
            #### update learning rate
            #model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            imgs_in = train_data['LQs']
            scale = 4
            b,n,c,h,w = imgs_in.size()
            # print(b,n,c,h,w)
            h_n = int(scale*np.ceil(h/scale))
            w_n = int(scale*np.ceil(w/scale))
            imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
            imgs_temp[:,:,:,0:h,0:w] = imgs_in
            train_data['LQs'] = imgs_temp
            if "scale" not in train_data.keys():
                H = 4*h
                W = 4*w
            else:
                H = train_data['GT'].shape[3]
                W = train_data['GT'].shape[4]
                train_data['scale'] = [[h_n*opt['scale']], [w_n*opt['scale']]]
            
            # print(H, W)
            model.feed_data(train_data)
            model.test()
            
            n = model.real_H.shape[1] - 2
            real_H = model.real_H[:,1:-1].reshape(b*n,3,H,W)
            if opt['network_G']['which_model_G'] == 'LIIF':
                fake_H = torch.stack(model.fake_H,1)
                fake_H = fake_H[:, :, :, 0:H, 0:W].reshape(b*n,3,H,W)
            elif opt['network_G']['which_model_G'] == 'Ours':
                fake_H = model.fake_H
                fake_H = fake_H[:, :, :, 0:H, 0:W].reshape(b*n,3,H,W)
            else:
                fake_H = model.fake_H
                fake_H = fake_H[:, :, :, 0:H, 0:W].reshape(b*n,3,H,W)
            
            print(real_H.shape, fake_H.shape)
            loss = abs(real_H-fake_H).mean().item()
            losses.append(loss)

            real_H *= 255.
            fake_H *= 255.
            real_H = (real_H[:,0] * 65.481 + real_H[:,1] * 128.553 + real_H[:,2] * 24.966)/255. + 16.
            fake_H = (fake_H[:,0] * 65.481 + fake_H[:,1] * 128.553 + fake_H[:,2] * 24.966)/255. + 16.
            real_H /= 255.
            fake_H /= 255.
            
            '''psnr_anchor = util.calculate_psnr(real_H[0].detach().cpu().numpy(), fake_H[0].detach().cpu().numpy())
            psnr_inter = util.calculate_psnr(real_H[1].detach().cpu().numpy(), fake_H[1].detach().cpu().numpy())+ util.calculate_psnr(real_H[2].detach().cpu().numpy(), fake_H[2].detach().cpu().numpy())
            psnr_center = psnr_inter'''
            
            mse = (real_H - fake_H) ** 2
            mse = torch.mean(mse.contiguous().view(b*n, -1), dim=1)
            psnr_anchor = (10 * torch.log10(1. ** 2 / mse[0:1]).mean().item())#/2 + (10 * torch.log10(1. ** 2 / mse[-1:]).mean().item())/2
            psnr_inter = 10 * torch.log10(1. ** 2 / mse[1:-1]).mean().item()
            psnr_center = 10 * torch.log10(1. ** 2 / mse[len(mse)//2]).mean().item()
            psnr = (psnr_anchor*1 + psnr_inter*(n-2))/(n-1)
            psnrs_anchor.append(psnr_anchor)
            psnrs_inter.append(psnr_inter)
            psnrs_center.append(psnr_center)
            psnrs.append(psnr)
            print((10 * torch.log10(1. ** 2 / mse))) 
            
            psnr_all = 10 * torch.log10(1. ** 2 / mse).cpu().numpy()
            psnrs_all.append(psnr_all)
            
            try:
                flows.append((abs(model.flow - model.flow_GT)).mean().item())
                flows_0.append((abs(model.flow)).mean().item())
            except:
                pass
            
            ssim = []
            for idx in range(n):
                s = util.calculate_ssim(real_H[idx:idx+1].permute(1,2,0).cpu().detach().numpy()*255., fake_H[idx:idx+1].permute(1,2,0).cpu().detach().numpy()*255.)
                ssim.append(s)
            ssims.append(np.mean(ssim[:-1]))
            ssim_all.append(ssim)
            
            # print(loss, np.mean(losses), psnr, np.mean(psnrs))
            model.log_dict['l'] = loss
            model.log_dict['ls'] = np.mean(losses)
            model.log_dict['p'] = psnr
            model.log_dict['ps'] = np.mean(psnrs)
            model.log_dict['p_a'] = psnr_anchor
            model.log_dict['ps_a'] = np.mean(psnrs_anchor)
            model.log_dict['p_i'] = psnr_inter
            model.log_dict['ps_i'] = np.mean(psnrs_inter)
            model.log_dict['p_c'] = psnr_center
            model.log_dict['ps_c'] = np.mean(psnrs_center)
            model.log_dict['ssim'] = ssims[-1]
            model.log_dict['ssims'] = np.mean(ssims)
            model.log_dict['flows'] = np.mean(flows)
            model.log_dict['flows_0'] = np.mean(flows_0)
            

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')>'
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            '''#### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)'''
        np.save("/psnrs/"+opt["name"]+".npy", psnrs_all)
        np.save("/psnrs/"+opt["name"]+"_ssim.npy", ssim_all)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
        del logger


if __name__ == '__main__':
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='/options/test/test_bfstvsr.yml', help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    main(opt)
    exit()
    s_scales = [4]
    t_scales = [6]
    for s in s_scales:
        for t in t_scales:
            if s == 4 and  t == 6:
                continue
            opt['scale'] = s
            opt['datasets']['train']["scale"] = s
            opt['datasets']['train']['time'] = t
            opt["name"] = "BFSTVSR"+str(s)+"x_t"+str(t)+"_12"
            main(opt)