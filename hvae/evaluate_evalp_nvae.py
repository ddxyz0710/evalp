

import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time

from torch.multiprocessing import Process
from torch.cuda.amp import autocast

from model import AutoEncoder
import utils
import datasets
from train_nvae import test, init_processes, test_vae_fid
from PIL import Image
from main_evalp_cifar_small_models_savedfeat import test_vae_fid_with_evalp, get_feature_map_size, _netE, _netEConv, EVALP, test_vae_fid_with_evalp_based_snis
from models import RealNVP, RealNVPLoss, RealNVP_Big
import torchvision.datasets as dset
from sklearn.neighbors import KDTree
from scipy import misc
import imageio
import torchvision.utils as vutils
from sklearn.decomposition import PCA
from tqdm import tqdm 


def get_KNN(sampled_images, images, save_dir, num_neighbours=10):

    to_range_0_1 = lambda x: (x + 1.) / 2.
    num_samples = sampled_images.shape[0]
    tree = KDTree(np.reshape(images, [images.shape[0], -1]), leaf_size=2)
    for i in range(num_samples):
        _, neighbour_inds = tree.query(np.reshape(sampled_images[i].flatten(), (1, -1)), k=num_neighbours)
        selected_images = images[neighbour_inds[0]]
        im2=torch.cat((torch.from_numpy(sampled_images[i]).unsqueeze(0), torch.from_numpy(selected_images)),dim=0)
        vutils.save_image(im2, os.path.join(save_dir, "torch"+ str(i)+".png"), normalize=True, scale_each=True, nrow=1)


        for j in range(len(selected_images)):
            selected_images[j] = to_range_0_1(selected_images[j])
        img_together = np.squeeze(np.hstack([sampled_images[i],np.hstack(selected_images)
                                            ]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imageio.imwrite(os.path.join(save_dir, str(i)+".png"), np.moveaxis(img_together, 0, -1))
        # imageio.imwrite(os.path.join(save_dir, str(i)+".pdf"), np.moveaxis(img_together, 0, -1))
        img = Image.fromarray(np.uint8(255 * np.moveaxis(img_together, 0, -1)))
        img.save(os.path.join(save_dir, "pil"+str(i)+".png"))
        img.save(os.path.join(save_dir, "pil"+str(i)+".pdf"))

def set_bn_with_evalp(model, evalp, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
        
    else:
        model.train()
        with autocast():
            for i in range(iter):
                if i % 10 == 0:
                    print('setting BN statistics iter %d out of %d' % (i+1, iter))
                model.sample_with_evalp(evalp, num_samples, t)
        model.eval()


def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()
        with autocast():
            for i in range(iter):
                if i % 10 == 0:
                    print('setting BN statistics iter %d out of %d' % (i+1, iter))
                model.sample(num_samples, t)
        model.eval()

def main(eval_args):
    # ensures that weight initializations are all the same
    if eval_args.readjust_bn:
        logging = utils.Logger(0, eval_args.save, fname='eval_log_bnadjusted.txt')
    else:
        logging = utils.Logger(0, eval_args.save, fname='eval_log.txt')
    logging.info("\n")  # Leaves a gap between different evals

    # load a checkpoint
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        logging.info('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    if eval_args.batch_size > 0:
        args.batch_size = eval_args.batch_size

    logging.info('loaded the model at epoch %d', checkpoint['epoch'])
    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
    # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
    # did not have this variable.
    #model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True) #! Try loading in strict mode
    model = model.cuda()

    # logging.info('args = %s', args)
    logging.info('num conv layers: %d', len(model.all_conv_layers))
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))

    # Sampler Net
    concat = False
    if args.dataset == 'celeba_64':
        ftr_map_size = 8
    elif args.dataset == 'cifar10':
        ftr_map_size = 16
    in_channel = args.num_latent_per_group
    eval_args.in_channel = in_channel
    eval_args.ftr_map_size = ftr_map_size
    eval_args.concat = concat
    eval_args.dataset = args.dataset
                                                    
    eval_args.bottleneck_factor = int(args.num_latent_per_group * ftr_map_size ** 2)
    eval_args.ftr_map_size = ftr_map_size
    eval_args.ndf = 200
    eval_args.nez = 1

    f_checkpoint = torch.load(eval_args.f_checkpoint, map_location='cpu')
    logging.info(f"Loading samplerNet at epoch {f_checkpoint['epoch']}")
    logging.info(eval_args.f_checkpoint)
    if eval_args.dataset == 'cifar10':
        Fnet = RealNVP(num_scales=2, in_channels=in_channel, mid_channels=eval_args.sampler_mid_channels, 
                            num_blocks=eval_args.sampler_num_blocks) #num_blocks=5
    elif eval_args.dataset == 'celeba_64':
        Fnet = RealNVP_Big(num_scales=2, in_channels=in_channel, mid_channels=eval_args.sampler_mid_channels, 
                            num_blocks=eval_args.sampler_num_blocks) #num_blocks=5
    else:
        raise ValueError("Only celeba_64 and cifar10 supported")
    
    Fnet.load_state_dict(f_checkpoint['model_state'])
    Fnet.cuda()

    # discNet
    e_checkpoint = torch.load(eval_args.e_checkpoint, map_location='cpu')
    Dnet = _netEConv(eval_args)
    logging.info(f"Loading discNet at epoch {e_checkpoint['epoch']}")
    logging.info(eval_args.e_checkpoint)
    Dnet.load_state_dict(e_checkpoint['model_state'])
    Dnet.cuda()

    # Define evalp
    evalp_in_size = [in_channel, ftr_map_size, ftr_map_size]
    evalp_out_size = evalp_in_size
    evalp = EVALP(Fnet, evalp_in_size)




    if eval_args.eval_mode == 'evaluate':
        # load train valid queue
        args.data = eval_args.data
        train_queue, valid_queue, num_classes = datasets.get_loaders(args)

        if eval_args.eval_on_train:
            logging.info('Using the training data for eval.')
            valid_queue = train_queue

        # get number of bits
        num_output = utils.num_output(args.dataset)
        bpd_coeff = 1. / np.log(2.) / num_output

        valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=eval_args.num_iw_samples, args=args, logging=logging)
        logging.info('final valid nelbo %f', valid_nelbo)
        logging.info('final valid neg log p %f', valid_neg_log_p)
        logging.info('final valid nelbo in bpd %f', valid_nelbo * bpd_coeff)
        logging.info('final valid neg log p in bpd %f', valid_neg_log_p * bpd_coeff)
    
    elif eval_args.eval_mode == 'evaluate_fid':
        bn_eval_mode = not eval_args.readjust_bn
        
        logging.info(f"Readusting batch norm={eval_args.readjust_bn}")
        set_bn_with_evalp(model, evalp, bn_eval_mode, num_samples=eval_args.batch_size, t=1.0, iter=500) 
        args.fid_dir = eval_args.fid_dir
        eval_args.num_process_per_node, eval_args.num_proc_node = eval_args.world_size, 1   # evaluate only one 1 node
        # fid = test_vae_fid(model, args, total_fid_samples=50000)
        
        fid = test_vae_fid_with_evalp(model, evalp, eval_args, total_fid_samples=50000)
        logging.info('fid with EVaLP is %f' % fid)

        # EvALp + SNIS
        fid = test_vae_fid_with_evalp_based_snis(model, Dnet, evalp, eval_args, total_fid_samples=50000)
        logging.info('fid with EVaLP + SNIS is %f' % fid)

    elif eval_args.eval_mode == 'knn':
        bn_eval_mode = not eval_args.readjust_bn
        logging.info(f"Readusting batch norm={eval_args.readjust_bn}")
        total_samples = 60
        num_samples = 60
        set_bn_with_evalp(model, evalp, bn_eval_mode, num_samples=num_samples, t=eval_args.temp, iter=200) 
        
        # num_iter = int(np.ceil(total_samples / num_samples))

        # store train dataset
        if eval_args.dataset == 'celeba_64':
            img_path = 'data/celeba/img_align_celeba'
            image_size = 64
        
        save_syn_dir = os.path.join(eval_args.save, 'syn', 'nn_evalp')
        os.makedirs(save_syn_dir, exist_ok=True)

        train_images = []
        # data = dset.celeba.CelebA(root=img_path, split='train', target_type='attr', transform=None, download=True)
        train_files = os.listdir(img_path)
        for fl in tqdm(train_files):
            path = f"{img_path}/{fl}"
            img = Image.open(path)
            im = img.resize((image_size,image_size))
            # im = np.array(im) / 255.
            im = np.array(im)
            train_images.append(im)
        
        train_images = np.stack(train_images) * 1. 
        n_train_images = len(train_images)

        torch.cuda.synchronize()
        with autocast():
            logits = model.sample_with_evalp(evalp, num_samples, eval_args.temp)

        output = model.decoder_output(logits)
        output_images = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
            else output.sample()
        torch.cuda.synchronize()
        end = time()
        output_images = output_images.detach().cpu().numpy() * 255.
        output_images = np.moveaxis(output_images, 1, -1) # [B, h, w, 3]

        combined_img_data = np.concatenate((train_images, output_images), axis=0)
        pca = PCA(n_components=100)
        pca_combined_img = pca.fit_transform(combined_img_data.reshape(combined_img_data.shape[0], -1))
        train_pca_data = pca_combined_img[:n_train_images]
        output_pca_data  = pca_combined_img[n_train_images:]
        
        img_together = np.empty((image_size, (10+1) * image_size, 3),np.float32)
        
        tree = KDTree(np.reshape(train_pca_data, [train_pca_data.shape[0], -1]), leaf_size=3)
        n_nn = min(10, len(output_images))
        for qid in range(n_nn):
            _, neighbour_inds = tree.query(np.reshape(output_pca_data[qid], (1, -1)), k=10)
            img_with_nn = np.hstack([output_images[qid], np.hstack(train_images[neighbour_inds[0]])])

            img_together = np.concatenate((img_together, img_with_nn), axis=0)

        im = Image.fromarray(np.uint8(img_together.astype(int)))
        k = np.random.randint(10000) # save different image everytime
        im.save(f"{save_syn_dir}/nn_{k}.png")




        

    
    else:
        bn_eval_mode = not eval_args.readjust_bn
        total_samples = 500 // eval_args.world_size          # num images per gpu
        num_samples = 100                                      # sampling batch size
        num_iter = int(np.ceil(total_samples / num_samples))   # num iterations per gpu


        with torch.no_grad():
            n = int(np.floor(np.sqrt(num_samples)))
            # set_bn_with_evalp(model, evalp, bn_eval_mode, num_samples=16, t=eval_args.temp, iter=200)
            set_bn(model, bn_eval_mode, num_samples=16, t=eval_args.temp, iter=500)
            save_syn_dir = os.path.join(eval_args.save, 'syn', 'evalp')
            os.makedirs(save_syn_dir, exist_ok=True)
            for ind in range(num_iter):     # sampling is repeated.
                torch.cuda.synchronize()
                start = time()
                with autocast():
                    logits = model.sample_with_evalp(evalp, num_samples, eval_args.temp)
                output = model.decoder_output(logits)
                output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                    else output.sample()
                torch.cuda.synchronize()
                end = time()
                logging.info('sampling time per batch: %0.3f sec', (end - start))

                visualize = False
                if visualize:
                    output_tiled = utils.tile_image(output_img, n).cpu().numpy().transpose(1, 2, 0)
                    output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
                    output_tiled = np.squeeze(output_tiled)

                    plt.imshow(output_tiled)
                    plt.show()
                else:
                    output_tiled = utils.tile_image(output_img, n).cpu().numpy().transpose(1, 2, 0)
                    output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
                    output_tiled = np.squeeze(output_tiled)

                    # file_path = os.path.join(save_syn_dir, 'gpu_%d_samples_%d_temp_%f.npz' % (eval_args.local_rank, ind, eval_args.temp))
                    # np.savez_compressed(file_path, samples=output_img.cpu().numpy())

                    file_path = os.path.join(save_syn_dir, 'gpu_%d_samples_%d_temp_%f.png' % (eval_args.local_rank, ind, eval_args.temp))
                    im = Image.fromarray(output_tiled)
                    im.save(file_path)
                    #logging.info('Saved at: {}'.format(file_path))'
            
            save_syn_dir = os.path.join(eval_args.save, 'syn', 'evalp_plus_snis')
            os.makedirs(save_syn_dir, exist_ok=True)

            for ind in range(num_iter):     # sampling is repeated.
                torch.cuda.synchronize()
                start = time()
                with autocast():
                    logits = model.importance_sample_with_evalp(Dnet, evalp, num_samples, eval_args.sir_multinomial_dim, eval_args.temp)
                output = model.decoder_output(logits)
                output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                    else output.sample()
                torch.cuda.synchronize()
                end = time()
                logging.info('sampling time per batch: %0.3f sec', (end - start))

                visualize = False
                if visualize:
                    output_tiled = utils.tile_image(output_img, n).cpu().numpy().transpose(1, 2, 0)
                    output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
                    output_tiled = np.squeeze(output_tiled)

                    plt.imshow(output_tiled)
                    plt.show()
                else:
                    output_tiled = utils.tile_image(output_img, n).cpu().numpy().transpose(1, 2, 0)
                    output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
                    output_tiled = np.squeeze(output_tiled)

                    file_path = os.path.join(save_syn_dir, 'gpu_%d_samples_%d_temp_%f.npz' % (eval_args.local_rank, ind, eval_args.temp))
                    np.savez_compressed(file_path, samples=output_img.cpu().numpy())

                    file_path = os.path.join(save_syn_dir, 'gpu_%d_samples_%d_temp_%f.png' % (eval_args.local_rank, ind, eval_args.temp))
                    im = Image.fromarray(output_tiled)
                    im.save(file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--checkpoint', type=str, default='/tmp/expr/checkpoint.pt',
                        help='location of the checkpoint')
    parser.add_argument('--save', type=str, default='/tmp/expr',
                        help='location of the checkpoint')
    parser.add_argument('--eval_mode', type=str, default='sample', choices=['sample', 'evaluate', 'evaluate_fid','knn'],
                        help='evaluation mode. you can choose between sample or evaluate.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--temp', type=float, default=0.7,
                        help='The temperature used for sampling.')
    parser.add_argument('--num_iw_samples', type=int, default=1000,
                        help='The number of IW samples used in test_ll mode.')
    parser.add_argument('--fid_dir', type=str, default='fid-stats',
                        help='path to directory where fid related files are stored')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    # added
    parser.add_argument('--f_checkpoint', type=str, required=True,
                        help='sampler chkpt')
    parser.add_argument('--e_checkpoint', type=str, required=True,
                        help='discNet chkpt')
    parser.add_argument('--sampler_mid_channels', type=int, required=True,
                        help='sampler mid channel dim')
    parser.add_argument('--sampler_num_blocks', type=int, required=True,
                        help='sampler mid channel dim')
    parser.add_argument('--sir_multinomial_dim', type=int, default=200,
                        help='sampler mid channel dim')


    args = parser.parse_args()
    utils.create_exp_dir(args.save)

    size = args.world_size

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