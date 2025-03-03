import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.MoTIF as MoTIF
import models.modules.ZSM as ZSM
import models.modules.EDVR as EDVR
import models.modules.bfstvsr as bfstvsr


####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LIIF':
        netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                            groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                            back_RBs=opt_net['back_RBs'])   
    elif which_model == 'MoTIF':
        netG = MoTIF.LunaTokis()
    elif which_model == 'bfstvsr':
        netG = bfstvsr.LunaTokis() 

    elif which_model == 'ZSM':
        netG = ZSM.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                            groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                            back_RBs=opt_net['back_RBs']) 
    elif which_model == 'EDVR':
        netG = EDVR.EDVR() 
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
