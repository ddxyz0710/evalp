import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from time import time
import json
import pickle as p
import sys

import torch.distributed as dist
from torch.multiprocessing import Process


from PIL import Image


from model import AutoEncoder
import utils
import datasets
from train_nvae import test, init_processes
from discriminator_model_conv import Discriminator, MiniBatchIterator
from pathlib import Path
from models import RealNVP, RealNVPLoss 
import torch.autograd as autograd
from torch.autograd import Variable
from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3
from flow_models import load_flows
import torch.utils.data as data

logging = None

# TODO merge with code below, print to file and plot pdf
stats_headings = [ #['epoch',          '{:>14}',  '{:>14d}'],
                ['err(E)',         '{:>14}',  '{:>14.3f}'],
                ['en_pos(E)',         '{:>14}',  '{:>14.3f}'],
                ['en_neg(E)',         '{:>14}',  '{:>14.3f}'],
                ['grad_penalty(E)',         '{:>14}',  '{:>14.3f}'],
                ['norm(grad(E))',  '{:>14}',  '{:>14.3f}'],
                ['norm(weight(E))',  '{:>14}',  '{:>14.3f}'],
                ['err(F)',         '{:>14}',  '{:>14.3f}'],
                ['log_prob(F)',         '{:>14}',  '{:>14.3f}'],
                ['neg_log_p0(F)',         '{:>14}',  '{:>14.3f}'],
                ['kld(F)',         '{:>14}',  '{:>14.3f}'],
                ['en(F)',         '{:>14}',  '{:>14.3f}'], #log_probF 
                ['norm(grad(F))',  '{:>14}',  '{:>14.3f}'],
                ['norm(weight(F))',  '{:>14}',  '{:>14.3f}'],
            #   ['inception',            '{:>14}',  '{:>14.3f}'],
            #   ['inception_std', '{:>14}', '{:>14.3f}'],
                ['fid(gen)', '{:>14}', '{:>14.3f}'],

            ]
def copy_source(file, output_dir):
    from shutil import copyfile
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))

class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.T_max = 150

        apply_sn = lambda x: x #sn if args.e_sn else lambda x: x

        f = nn.LeakyReLU() #get_activation(args.e_activation, args)

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(args.bottleneck_factor, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.nez))
        )

    def forward(self, z):
        # return self.ebm(z.squeeze()).view(-1, self.args.nez, 1, 1)
        return self.ebm(z.squeeze()).view(-1, self.args.nez)

class _netEConv(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.T_max = 200
        self.concat = args.concat

        apply_sn = lambda x: x #sn if args.e_sn else lambda x: x

        f = nn.LeakyReLU() #get_activation(args.e_activation, args)

        self.ebm = nn.Sequential(
            nn.Conv2d(args.in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(args.ftr_map_size), # avgpool over whole spatial dim
            nn.Flatten(),

            # apply_sn(nn.Linear(128, args.ndf)),
            # f,

            # apply_sn(nn.Linear(args.ndf, args.ndf)),
            # f,

            # apply_sn(nn.Linear(args.ndf, args.nez))
            apply_sn(nn.Linear(128, args.nez))
        )

    def forward(self, z):
        # return self.ebm(z.squeeze()).view(-1, self.args.nez, 1, 1)
        return self.ebm(z.squeeze()).view(-1, self.args.nez)

        # Taken from ncp-vae code 
    def get_importance_weights(self, z, s, dist, t=1.0):
        '''
            returns iw_weight, w(z) = - f_\theta(z)
            Note: In NCP-VAE, w(z) =  f_\theta(z) ; difference of sign
        '''
        num_chunks = 100
        self.eval()
        with torch.no_grad():
            sir_mutinomial_dim = z.size(1)
            s = s.unsqueeze(1).expand(-1,sir_mutinomial_dim, -1,-1,-1 )
            collect_log_iw = []
            assert sir_mutinomial_dim % num_chunks == 0, 'sir_mult_dim not integer multiple of num_chunks!!!'
            if self.concat:
                for z_chunk, s_chunk in zip(torch.chunk(z, num_chunks, dim=1), torch.chunk(s, num_chunks, dim=1)):
                    z_s = torch.cat([z_chunk, s_chunk], dim=2)
                    batch_size, chunk_sir_mutinomial_dim, *rd = z_s.size()
                    z_s = z_s.reshape(batch_size * chunk_sir_mutinomial_dim, *rd)
                    log_iw = -self.forward(z_s)  # since p_\theta = exp(-f_\tehta) / Z
                    collect_log_iw.append(log_iw.view(batch_size, chunk_sir_mutinomial_dim))
            else:
                for z_chunk in torch.chunk(z, num_chunks, dim=1):
                    z_s = z_chunk
                    batch_size, chunk_sir_mutinomial_dim, *rd = z_s.size()
                    z_s = z_s.reshape(batch_size * chunk_sir_mutinomial_dim, *rd)
                    log_iw = -self.forward(z_s) 
                    collect_log_iw.append(log_iw.view(batch_size, chunk_sir_mutinomial_dim))

            log_iw_final = torch.cat(collect_log_iw, dim=1)
        return log_iw_final


def set_bn(model, bn_eval_mode, disc_list, sir_multinomial_dim, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        for i in range(iter):
            if i % 10 == 0:
                print('setting BN statistics iter %d out of %d' % (i+1, iter))
            model.train()
            for d in disc_list:
                d.train()
            # model.sample(num_samples, t)
            model.sample_with_importance_weights(disc_list=disc_list, 
                                    num_samples=num_samples, 
                                    sir_multinomial_dim=sir_multinomial_dim, 
                                    t=t)
        for d in disc_list:
            d.eval()
        model.eval()


def get_feature_map_size(group_idx, num_gps_list):
    if len(num_gps_list)==1:
        if group_idx < num_gps_list[0]:
            return 16 #8 #4
    if len(num_gps_list)==2:
        if group_idx < num_gps_list[0]:
            return 8 #8
        if num_gps_list[0] <= group_idx < sum(num_gps_list[0:2]):
            return 16
    if len(num_gps_list)==3:
        if group_idx < num_gps_list[0]:
            return 8 
        if num_gps_list[0] <= group_idx < sum(num_gps_list[0:2]):
            return 16
        if sum(num_gps_list[0:2]) <= group_idx < sum(num_gps_list[0:3]):
            return 32
    else:
        if group_idx < num_gps_list[0]:
            return 8 
        if num_gps_list[0] <= group_idx < sum(num_gps_list[0:2]):
            return 16
        if sum(num_gps_list[0:2]) <= group_idx < sum(num_gps_list[0:3]):
            return 32
        if sum(num_gps_list[0:3]) <= group_idx < sum(num_gps_list[0:4]):
            return 64
        if sum(num_gps_list[0:4]) <= group_idx < sum(num_gps_list[0:5]):
            return 128
        
from matplotlib import pyplot as plt        
def plot_stats(stat_1, stat_1_i, output_dir):
    p_i = 1
    p_n = len(stat_1)

    f = plt.figure(figsize=(20, p_n * 5))

    def plot(stats, stats_i):
        nonlocal p_i
        for j, (k, v) in enumerate(stats.items()):
            plt.subplot(p_n, 1, p_i)
            plt.plot(stats_i, v)
            plt.ylabel(k)
            p_i += 1

    plot(stat_1, stat_1_i)

    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    plt.close(f)

def save_model(opt_dict, epoch, output_dir, idfier=None):
    for key in opt_dict:
        save_dict = {
            'epoch': epoch,
            'model_state': opt_dict[key][0].state_dict(),
            'optimizer_state': opt_dict[key][1].state_dict()
        }
        if idfier is not None:
            torch.save(save_dict, '%s/%s_%s.pth' % (output_dir, key, idfier))
        else:
            torch.save(save_dict, '%s/%s.pth' % (output_dir, key))

def main(eval_args):
    
    global logging
    logging = utils.Logger(0, eval_args.save, fname='log.txt')
    logging.info(eval_args)

    # load a checkpoint
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    # checkpoint = torch.load(eval_args.data_dir + eval_args.checkpoint, map_location='cpu')
    checkpoint = torch.load(eval_args.root_dir + eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    logging.info('loaded model at epoch %d', checkpoint['epoch'])
    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    model = model.cuda()
    print('num conv layers:', len(model.all_conv_layers))

    
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))

    # if eval_args.eval_mode == 'evaluate':
    if eval_args.is_local:
        if args.dataset == 'mnist':
            args.data = eval_args.data_dir + '/MNIST/'
        elif args.dataset == 'cifar10':
            args.data = eval_args.data_dir + '/cifar10/'                
        elif args.dataset == 'imagenet_32':
            args.data = eval_args.data_dir + '/data/datasets/imagenet-oord/imagenet-oord-lmdb_32/'
        elif args.dataset == 'celeba_64':
            args.data = eval_args.data_dir #+ '/celeba64_lmdb/' #'/data/celeba_64/'
            print(eval_args.data_dir) #+ '/celeba64_lmdb/')
        elif args.dataset == 'celeba_256':
            args.data = eval_args.data_dir + '/celeba_256/'
        else:
            raise ValueError('Unknown Dataset')    
        
    # load train valid queue
    args.batch_size = eval_args.train_batch_size
    eval_args.batch_size = eval_args.train_batch_size
    eval_args.dataset = args.dataset

    # NOTE: MOdify/ change the appropriate dataset inside get_loaders fn incase 
    # type of data used is changed. 
    # When using saved feats, ImageArrayDataset should be used. Else, LMDBDataset (for celeba) or 
    # torch.data.dataset should be used for cifar10.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args, 'eval')
    print(len(train_queue))
    print(len(valid_queue))

    if eval_args.eval_on_train:
        logging.info('Using the training data for eval.')
        valid_queue = train_queue

    concat = False
    # discNet = Discriminator(ftr_map_size=get_feature_map_size(eval_args.group_idx, eval_args.num_groups_per_scale_list,
    #                                                           ), args=args, concat=False).cuda()    
    # ensures that weight initializations are all the same
    ftr_map_size = get_feature_map_size(eval_args.group_idx, eval_args.num_groups_per_scale_list)                                                       
    eval_args.bottleneck_factor = int(args.num_latent_per_group * ftr_map_size ** 2)
    eval_args.ftr_map_size = ftr_map_size
    eval_args.ndf = 200
    eval_args.nez = 1
    in_channel = args.num_latent_per_group
    eval_args.in_channel = in_channel
    eval_args.ftr_map_size = ftr_map_size
    eval_args.concat = concat

    #discNet = _netE(eval_args).cuda()
    discNet = _netEConv(eval_args).cuda()
    print(discNet)

    discriminator_path = eval_args.save #+ f'/midchn{eval_args.sampler_mid_channels}-nblk{eval_args.sampler_num_blocks}'
    # eval_args.save = discriminator_path
    Path(discriminator_path).mkdir(parents=True, exist_ok=True)
    copy_source(__file__, discriminator_path)  # save the current file to the output dir

    samplerNet = RealNVP(num_scales=2, in_channels=in_channel, mid_channels=eval_args.sampler_mid_channels, 
                         num_blocks=eval_args.sampler_num_blocks).cuda() #num_blocks=5

    sampler_param_count = sum(p.numel() for p in samplerNet.parameters() if p.requires_grad)
    print(samplerNet)
    print('num samplerNet paramters: ', sampler_param_count)

    optimD = torch.optim.Adam(discNet.parameters(), lr=eval_args.lrD, weight_decay=1e-6, betas=(0.5, 0.999))
    optimF = torch.optim.Adam(samplerNet.parameters(), lr=eval_args.lrF, weight_decay=1e-6, betas=(0.5, 0.999))
 
    # Initialize stats
    stat_1 = {k[0]:[] for k in stats_headings}
    stat_1_i = []
    stat_1['lrD'] = []
    stat_1['lrF'] = []
    fid = 0.0
    best_fid = 1e5
    best_recon_fid = 1e5
    fid_dict={'gen_fid':1e5, 'recon_fid':1e5}
    inception = 0.0
    inception_std = 0.0
    mse_val = 0.0
    best_mse_val = 1e5
    auroc = 0
    # fixed_noise = torch.randn(100, in_channel * ftr_map_size * ftr_map_size).cuda()
    fixed_noise = torch.randn(100, in_channel , ftr_map_size , ftr_map_size).cuda()

    #extract_feats(args, discNet, model, valid_queue, group_idx=eval_args.group_idx, loss_path=discriminator_path)
    for epch in range(discNet.T_max):
        epoch_stats = train(eval_args, epch, discNet, samplerNet, model, 
          valid_queue, 
          optimD,
          optimF, 
          group_idx=0, 
          loss_path=None, 
          concat=concat) # since sampler's in_channel == in_channel in N(0,1) in dec
        
        fid_temp = test(eval_args, epch, discNet, samplerNet, model, 
          valid_queue, 
          fixed_noise,
          group_idx=0, 
          loss_path=None, 
          concat=concat)
        
        # Scheduler step 
        # lrD_schedule.step()
        
        if fid_temp is not None:
            fid = fid_temp
            if fid < best_fid:
                best_fid = fid
                opt_dict = {'Dnet': (discNet, optimD),'Fnet': (samplerNet, optimF)}
                save_model(opt_dict, epch+1, eval_args.save, idfier='best_fid')
                logging.info(f"Saved best fid model at epoch {epch+1}")
        epoch_stats['fid(gen)'] = fid

        # plot stats and save checkpoint
        for key in epoch_stats.keys():
            stat_1[key].append(epoch_stats[key])
        #TODO: update with scheduler
        stat_1['lrD'].append(5e-4)
        stat_1['lrF'].append(5e-4)

        stat_1_i.append(epch)
        plot_stats(stat_1, stat_1_i, eval_args.save)
        opt_dict = {'Dnet': (discNet, optimD),'Fnet': (samplerNet, optimF)}
        save_model(opt_dict, epch+1, eval_args.save)

        logging.info(f"Epoch: {epch+1}")
        out_str = "\t".join(["{}:{:.4f}".format(key, epoch_stats[key]) for key in epoch_stats.keys()])
        logging.info(out_str)

        #TODO: save checkpoint and plot stats

def train(args, epoch,
          Dnet, Fnet, model, 
          valid_queue, 
          optimD,
          optimF, 
          group_idx=0, 
          loss_path=None, 
          concat=True):
    
    def getGradNorm(net):
        pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
        gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
        return pNorm, gradNorm
    
    def diag_standard_normal_NLL(z):
        # get the negative log-likelihood of standard normal distribution
        if z.ndim == 4:
            nll = 0.5 * torch.sum((torch.mul(z, z)).squeeze(), dim=[1,2,3])
        else:
            nll = 0.5 * torch.sum((torch.mul(z, z)).squeeze(), dim=1)
        return nll.squeeze()
    
    def compute_gradient_penalty(netE, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        # alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1))).cuda()
        if real_samples.ndim == 4:
            alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1,1,1))).cuda()
        else:
            alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1))).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True) 
        en_interpolates = netE(interpolates) # B, nez
        fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        # Get gradient w.r.t. interpolates 
        # outputs and grad_outputs should have the same size
        gradients = autograd.grad(
            outputs=en_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    global logging
    # model.eval()
    print('Len valid queue: ', len(valid_queue))
    loss_per_epoch = []
    count=0
    Dnet.train()
    Fnet.train()
    model.eval()


    in_channel = args.in_channel # True only if concat is False

    loss_val = 0.
    n = 0
    stats_values = {k[0]: 0 for k in stats_headings}
    num_batch = len(valid_queue)


    for step, x in enumerate(valid_queue):
        # if step % 10 == 0: print(step)
        x = x[0] if len(x) > 1 else x
        x = x.cuda()
            # x = utils.pre_process(x, self.args.num_x_bits)
        # x = utils.pre_process(x, 8) #hardcoded for celebA-256 
        z_pos = x
        pnoise = torch.randn(z_pos.size()).cuda()

        # assert z_pos.size() == pnoise.size()
        # flatten 
        bsz = z_pos.size(0)


        model.zero_grad()
        Dnet.zero_grad()
        

        z_neg, _ = Fnet(pnoise, reverse=False)
        # z_neg, _ = Fnet(pnoise, mode='direct')
        en_pos = torch.mean(Dnet(z_pos.detach()))
        en_neg  = torch.mean(Dnet(z_neg.detach()))
        gradient_penalty = compute_gradient_penalty(Dnet, z_pos.data, z_neg.data)
        
        # NOTE: In WGAN-gp paper disc_cost = disc_fake - disc_real + args.lambda_gp * gradient_penalty 
        errD =  en_pos - en_neg + args.lambda_gp * gradient_penalty 
        errD.backward()
        optimD.step()

        netEpNorm, netEgradNorm = getGradNorm(Dnet)

        if (step+1) % 10 == 0: # 5
            Fnet.zero_grad()
            Dnet.zero_grad()
            pnoise = torch.randn(z_pos.size()).cuda()
            #normal_dist = Normal(mu=torch.zeros_like(z_pos), log_sigma=torch.zeros_like(z_pos))

            z_neg, logdets = Fnet(pnoise, reverse=False)
            # z_neg, logdets = Fnet(pnoise, mode='direct')
            logdets = torch.zeros(z_neg.shape[0]).cuda() #!
            neg_log_prob_z = diag_standard_normal_NLL(pnoise)
            log_probF = torch.mean(-neg_log_prob_z - logdets)
            neg_log_p0_z_neg = torch.mean(diag_standard_normal_NLL(z_neg))
            reverse_klF = log_probF + neg_log_p0_z_neg
            en_F = torch.mean(Dnet(z_neg)) 
            errF = en_F +  0. * reverse_klF  # TODO: Are these two losses balanced?
            
            errF.backward()
            optimF.step()
            netFpNorm, netFgradNorm = getGradNorm(Fnet)
            if (step+1) % 20 == 0:
                logging.info('[%3d/%3d][%3d/%3d] errF: %6.2f, neg errE: %6.2f, en(F): %6.2f, log_prob(F): %6.2f, neg_logp0(F): %6.2f, kld(F): %6.2f, norm weight(F): %6.2f, norm grad(F): %6.2f, en(pos): %6.2f, en(neg): %6.2f, grad_penalty(E): %6.2f, norm weight(E): %6.2f, norm grad(E): %6.2f,'
                    % (epoch+1, Dnet.T_max, step+1, len(valid_queue),
                        errF.data.item(), -errD.data.item(), en_F.data.item(),
                        log_probF.data.item(), neg_log_p0_z_neg.data.item(),reverse_klF.data.item(), 
                        netFpNorm.data.item(), netFgradNorm.data.item(),
                        en_pos.data.item(), en_neg.data.item(), gradient_penalty.data.item(), netEpNorm.data.item(), netEgradNorm.data.item(),
                        ))

            stats_values['err(E)'] += errD.data.item() / num_batch
            stats_values['en_pos(E)'] += en_pos.data.item() / num_batch
            stats_values['en_neg(E)'] += en_neg.data.item() / num_batch
            stats_values['grad_penalty(E)'] += gradient_penalty.data.item() / num_batch
            stats_values['norm(grad(E))'] += netEgradNorm.data.item() / num_batch
            stats_values['norm(weight(E))'] += netEpNorm.data.item() / num_batch
            stats_values['err(F)'] += errF.data.item() / num_batch
            stats_values['log_prob(F)'] += log_probF.data.item() / num_batch
            stats_values['neg_log_p0(F)'] += neg_log_p0_z_neg.data.item() / num_batch
            stats_values['kld(F)'] += reverse_klF.data.item() / num_batch
            stats_values['en(F)'] += en_F.data.item() / num_batch
            stats_values['norm(grad(F))'] += netFgradNorm.data.item() / num_batch
            stats_values['norm(weight(F))'] += netFpNorm.data.item() / num_batch
            sys.stdout.flush()

        # break
    
    return stats_values


def test(args, epoch,
          Dnet, Fnet, model, 
          valid_queue, 
          fixed_noise,
          group_idx=0, 
          loss_path=None, 
          concat=True):

    Dnet.eval()
    Fnet.eval()
    model.eval()


    in_channel = args.in_channel

    evalp = EVALP(Fnet, list(fixed_noise.size()[1:]), out_size=[20, 8, 8])
    save_syn_dir = os.path.join(args.save, 'syn')
    os.makedirs(save_syn_dir, exist_ok=True)

    # args.sampling_temp = 1.0
    if (epoch) % 1==0 or epoch + 1 ==Dnet.T_max:
        with torch.no_grad():
            logits = model.sample_with_evalp(evalp, fixed_noise.shape[0], args.sampling_temp, noise=fixed_noise) #TODO: temp
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
            
            n = int(np.floor(np.sqrt(fixed_noise.shape[0])))
            output_tiled = utils.tile_image(output_img, n).cpu().numpy().transpose(1, 2, 0)
            output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
            output_tiled = np.squeeze(output_tiled)
            file_path = os.path.join(save_syn_dir, f'gen_samples_fixed_noise_epoch{epoch}_temp{args.sampling_temp}.png')
            im = Image.fromarray(output_tiled)
            im.save(file_path)

            # random image
            logits = model.sample_with_evalp(evalp, fixed_noise.shape[0], args.sampling_temp, noise=None) 
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
            
            n = int(np.floor(np.sqrt(output_img.shape[0])))
            output_tiled = utils.tile_image(output_img, n).cpu().numpy().transpose(1, 2, 0)
            output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
            output_tiled = np.squeeze(output_tiled)
            file_path = os.path.join(save_syn_dir, f'gen_samples_epoch{epoch}_temp{args.sampling_temp}.png')
            im = Image.fromarray(output_tiled)
            im.save(file_path)
    #Fid
    #TODO: Settings should match with evaluate_nvae.py
    fid = None
    if (epoch) % 5 == 0 or (epoch + 1) == Dnet.T_max:
        bn_eval_mode = not args.readjust_bn
        # set_bn(model, bn_eval_mode, num_samples=2, t=1.0, iter=500) #! This shouldn't
        #! be used during training. Also, needs to be checked on how to use. 
        if (epoch + 1) == Dnet.T_max:
            args.num_fid_samples = 50000
        fid = test_vae_fid_with_evalp(model, evalp, args, args.num_fid_samples)
        # fid = 100.0 #test_vae_fid_with_evalp(model, evalp, args, args.num_fid_samples)
        return fid
    return fid



class EVALP():
    def __init__(self, Fnet, size, out_size=None):
        ''' size shouldn;t include batch_size
        '''
        self.Fnet = Fnet
        self.size = size
        self.out_size = out_size
        if self.out_size is None:
            self.out_size = self.size 

    def sample(self, num_samples=1, noise=None):
        if noise is None:
            noise = torch.randn([num_samples] + self.size).cuda()
        # if self.is_sampler_1d and noise.ndim > 2:
        #         noise = noise.reshape(noise.shape[0], -1)
        num_samples = noise.shape[0]
        # return self.Fnet(noise, mode='direct')[0].view([num_samples]+self.out_size)
        return self.Fnet(noise, reverse=False)[0].view([num_samples] + self.out_size)

    def log_prob(self, x):
        return NotImplementedError

def create_generator_vae_with_evalp_based_snis(model, Dnet, evalp, batch_size, sir_multinomial_dim, num_total_samples):
    # batch_size = 20 #evalp.size[0]
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.importance_sample_with_evalp(Dnet, evalp, batch_size, sir_multinomial_dim, 1.0) # 2nd arg is the temperature
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
            # output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
            assert output_img.max() < 1.0001
            assert output_img.min() > -0.0001
        yield output_img.float()



def create_generator_vae_with_evalp(model, evalp, batch_size, num_total_samples):
    # batch_size = batch_size #evalp.size[0]
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample_with_evalp(evalp, batch_size, 1.0) # 2nd arg is the temperature
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
            # output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
            assert output_img.max() < 1.0001
            assert output_img.min() > -0.0001
        yield output_img.float()

def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0) # 2nd arg is the temperature
            output = model.decoder_output(logits)
            
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
   
        yield output_img.float()

def test_vae_fid_with_evalp(model, evalp, args, total_fid_samples):
    dims = 2048
    device = 'cuda'
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae_with_evalp(model, evalp, args.batch_size, num_sample_per_gpu)
    # g = create_generator_vae(model, 100  ,num_sample_per_gpu) #!!
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid


def test_vae_fid_with_evalp_based_snis(model, Dnet, evalp, args, total_fid_samples):
    dims = 2048
    device = 'cuda'
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae_with_evalp_based_snis(model, Dnet, evalp, args.batch_size, args.sir_multinomial_dim, num_sample_per_gpu)
    # g = create_generator_vae(model, 100  ,num_sample_per_gpu) #!!
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid







    # path_fake_data = os.path.join(args.save, f'pz_samples_{group_idx}.npy')
    # np.save(path_fake_data, fake_samples_all)



if __name__ == '__main__':



    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--checkpoint', type=str, default='/tmp/nasvae/expr',
                        help='location of the checkpoint')
    parser.add_argument('--save', type=str, default='/tmp/nasvae/expr',
                        help='location of the checkpoint')
    parser.add_argument('--eval_mode', type=str, default='sample', choices=['sample', 'evaluate'],
                        help='evaluation mode. you can choose between sample or evaluate.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--is_local', action='store_true', default=False,
                        help='Settings this to true will load data on the local machine.')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--num_iw_samples', type=int, default=1000,
                        help='The number of IW samples used in test_ll mode.')
    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--port_number', type=str, default='6020',
                        help='port number')        
    parser.add_argument('--expt_name', type=str, default='base',
                        help='port number')
    parser.add_argument('--ld_lr', type=float, default=1e-3,
                        help='step size for langevin dynamics')

    parser.add_argument('--group_idx', type=int, default=0,
                        help='which level of the heiratchy used for training the discriminator')   
    parser.add_argument('--ftr_map_size', type=int, default=8,
                        help=' size of conv feature map')
    parser.add_argument('--train_discriminator', action='store_true', default=False,
                        help='default gets all sample data.')
    parser.add_argument('--temperature_list', nargs='+', type=float, dest='temperature_list', default=[],
                        help=' temperature values for which we want ')
    parser.add_argument('--num_samples', type=int, default=16,
                        help=' test time batch size')
    parser.add_argument('--num_ld_updates', type=int, default=10, help='number of steps of langevin dynamics')    
    parser.add_argument('--num_stein_updates', type=int, default=10, help='number of steps of SVGD')    
    parser.add_argument('--svgd_lr', type=int, default=10, help=' SVGD learning rate')  
    parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size for discriminator')  

    parser.add_argument('--hmc_nsamples_from_chain_end', type=int, default=1, help=' samples taken after hmc finishes')  
    parser.add_argument('--hmc_warmup_steps', type=int, default=100, help=' warmup for hmc')  
    
    parser.add_argument('--HMC_steps', type=int, default=10, help='number of HMC steps')  
    parser.add_argument('--leapfrog_steps', type=int, default=10, help='number of leapfrog steps')  
    parser.add_argument('--HMC_step_size', type=float, default=0.001, help='HMC step size')  

    parser.add_argument('--num_groups_per_scale_list', nargs='+', type=int, dest='num_groups_per_scale_list', default=[],
                        help=' number of groups in each scale')    
    parser.add_argument('--rescale_shift', type=bool, default=False,
                        help=' number of groups per scale')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to data workspace')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='local mount, code resides here')
    parser.add_argument('--disc_dir', type=str, default=None,
                        help='directory for storing discriminator')
    parser.add_argument('--config_file', default=None,
                        help='optional yaml file with configuration parameters')
    
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='weight for grad penalty')
    parser.add_argument('--kl_weight', type=float, default=1e-8,
                        help='weight for sampler KL loss')
    parser.add_argument('--fid_dir', type=str, default='fid-stats',
                        help='Location of stored fid statistics')
    parser.add_argument('--num_fid_samples', type=int, default=50000,
                        help='Num of samples for fid calculcation')
    parser.add_argument('--sampling_temp', type=float, default=1.0,
                        help='Num of samples for fid calculcation')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='needed during fid')
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='r')
    parser.add_argument('--sampler_num_blocks', type=int, default=8,
                        help='r')
    parser.add_argument('--sampler_mid_channels', type=int, default=64,
                        help='r')
    parser.add_argument('--lrD', type=float, default=1e-4,
                        help='LR for discriminator')
    parser.add_argument('--lrF', type=float, default=1e-4,
                        help='LR for Sampler/Generator net')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay for optims')
    parser.add_argument('--sir_multinomial_dim', type=int, default=100,
                        help='dimnsion of categorical for doing SIR sampling') 
    
    args = parser.parse_args()
    utils.create_exp_dir(args.save)
    if args.config_file:
        print('Found Config File..')
        import yaml
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    print(vars(args))
    size = args.world_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            p = Process(target=init_processes, args=(rank, size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)
