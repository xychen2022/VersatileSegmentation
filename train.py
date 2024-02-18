#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18, 2023

@author: xychen
"""

import os
import random
import datetime
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import numpy as np
from dataset import CustomDataSet
from test_on_epoch_end import test
from networks.vit_seg_configs import get_vit_3d_config
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.loss import BalancedCELoss, DiceLoss

from tensorboardX import SummaryWriter

import timeit
start = timeit.default_timer()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    
    parser = argparse.ArgumentParser(description="Versatile Model toward Universal Segmentation")
    
    parser.add_argument("--data_dir", type=str, default='/gpfs/fs001/cbica/home/chexiao/MultiOrgan/awesome/MM')
    parser.add_argument("--train_mode", type=str, default='sparse_only', help="Valid value include 'partial_only', 'sparse_only' and 'mixed'")
    parser.add_argument("--use_simulated_sparse_data", type=str2bool, default=True, help="Ineffective when train_mode == 'partial_only'. True if the sparsely annotated data is simulated with partially or even fully annotated data")
    parser.add_argument("--keep_one_every_this_num_of_slices", type=int, default=5, help="Effective only when use_simulated_sparse_data == True. Control the sparsity of annotations. The default leads to 20% annotated slices for training")
    parser.add_argument("--views", nargs="+", type=str, default=['axial', 'sagittal'], help="The default should be a subset of ['axial', 'sagittal', 'coronal'] when use_simulated_sparse_data = True")
    
    parser.add_argument("--datasets_for_training", nargs="+", type=str, default=["amos", "btcv", "flare22", "abdomenct1k", "totalsegmentator", "nihpancreas", "word", "urogram122"], help="List of training datasets. This is used for printing the datasets and the number of images in each.")
    parser.add_argument("--modalities_for_training", nargs="+", type=str, default=["ct", "t1w"], help="List of modalities included in training datasets")
    
    parser.add_argument("--datasets_for_testing", nargs="+", type=str, default=["amos", "btcv", "flare22", "abdomenct1k", "totalsegmentator", "nihpancreas", "word", "urogram122"], help="List of testing datasets. By default, all datasets are involved in testing")
    parser.add_argument("--modalities_for_testing", nargs="+", type=str, default=["ct", "t1w"], help="List of modalities included in testing datasets")
    
    parser.add_argument("--test_regularly_in_training", type=str2bool, default=True, help="True if you want to monitor the training process by testing the model regularly")
    parser.add_argument("--test_or_save_every_this_num_epochs", type=int, default=10)
    
    parser.add_argument("--partial_tree_representation", type=str, default='tree_repr_3sets_mm_partial.pkl', help="Tree representation partially annotated data for training (the default follows a class->modality->dataset hierarchy). Used for sampling in training")
    parser.add_argument("--sparse_tree_representation", type=str, default='tree_repr_5sets_mm_sparse.pkl', help="Tree representation for sparsely annotated data for training (the default follows a class->modality->dataset hierarchy). Used for sampling in training")
    parser.add_argument("--snapshot_dir", type=str, default='./snapshots/')
    
    parser.add_argument("--batch_size", type=int, default=8, help="As implied by the name") # 8
    parser.add_argument("--val_batch_size", type=int, default=2, help="As implied by the name")
    parser.add_argument("--num_workers", type=int, default=8, help="Num of workers to load data") # 8
    
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False, help="Set to False when training from scratch. Otherwise, set to True")
    parser.add_argument("--start_epoch", type=int, default=0, help="The index of epoch from which training is resumed")
    parser.add_argument("--reload_path", type=str, default='snapshots/best_model.pth', help="Path to checkpoint. Should be consistent with start_epoch")
    
    parser.add_argument("--common_spacing", nargs="+", type=float, default=[2.0, 2.0, 2.0], help="The voxel spacing for normalization")
    parser.add_argument("--input_size", nargs="+", type=int, default=[112, 112, 112], help="Patch size") # [112, 112, 112], [96, 96, 96]
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    
    parser.add_argument("--num_epochs", type=int, default=200, help="As implied by the name")
    parser.add_argument("--samples_with_replacement_per_modality", type=int, default=4000, help="As implied by the name")
    parser.add_argument("--total_classes", type=int, default=17, help="As implied by the name")
    parser.add_argument("--weight_entropy_minimization", type=int, default=3)
    
    parser.add_argument("--clip_lower", type=int, default=-400, help="Used for intensity clipping of CT images only")
    parser.add_argument("--clip_upper", type=int, default=400, help="Used for intensity clipping of CT images only")
    
    parser.add_argument("--use_amp", type=str2bool, default=False) # Not used
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--ignore_label", type=int, default=255) # Not used
    parser.add_argument("--random_seed", type=int, default=1234)
    
    return parser

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def seed_worker(seed):
    np.random.seed(seed)
    random.seed(seed)
    
def main():
    
    parser = get_arguments()
    args = parser.parse_args()
    
    assert args.train_mode in ['partial_only', 'sparse_only', 'mixed'], "args.train_mode has the WRONG value!"
    if args.train_mode in ['sparse_only', 'mixed'] and args.use_simulated_sparse_data == True:
        for view in args.views:
            assert view in ['axial', 'sagittal', 'coronal'], "args.views has the WRONG element in it!"
    
    val_subjects = []
    images_for_val = os.listdir(args.data_dir + "/imagesVa")
    for val_subject in images_for_val:
        modality = val_subject.split('_')[1]
        dataset = val_subject.split('_')[0]
        
        if dataset in args.datasets_for_testing and modality in args.modalities_for_testing:
            val_subjects.append(val_subject)
    
    num_of_gpus = torch.cuda.device_count()
    
    if args.batch_size < num_of_gpus:
        args.batch_size = num_of_gpus
    else:
        if not args.batch_size % num_of_gpus == 0:
            args.batch_size = args.batch_size // num_of_gpus * num_of_gpus
    
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=60 * len(val_subjects)))
    dist.barrier()
    
    rank = dist.get_rank()
    pid = os.getpid()
    device_id = rank % torch.cuda.device_count()
    
    print(f"Current pid: {pid}")
    print(f"Current rank: {rank}")
    print(f"Current device_id: {device_id}")
    
    if device_id == 0:
        print("num_of_val_images = ", len(val_subjects))
        print("num_of_gpus = ", num_of_gpus)
    
    if rank == 0:
        writer = SummaryWriter(args.snapshot_dir)
    
    cudnn.benchmark = True
    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create model
    vit_seg = get_vit_3d_config()
    model = ViT_seg(vit_seg, img_size=args.input_size, num_classes=args.total_classes, in_channels=1)
    model.to(device_id)
    
    # Load checkpoint...
    if args.reload_from_checkpoint:
        map_location = {'cuda:0': f"cuda:{device_id}"}
        state_dict = torch.load(args.reload_path, map_location=map_location)
        assert args.start_epoch == state_dict['epoch']
        print('loading from checkpoint at {}-th epoch'.format(args.start_epoch))
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        assert args.start_epoch == 0, "start_epoch should be 0 when training from scratch"
    
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    
    optimizer = torch.optim.AdamW(ddp_model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    
    loss_seg_DICE = DiceLoss(n_classes=args.total_classes)
    loss_seg_CE = BalancedCELoss(n_classes=args.total_classes, beta=1.0, gamma=2.0, multiplier_for_unlabeled_data=args.weight_entropy_minimization)
    
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    trainset = CustomDataSet(args, samples_with_replacement_per_modality=args.samples_with_replacement_per_modality, device_id=device_id)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=trainset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size//num_of_gpus, sampler=train_sampler, num_workers=args.num_workers, worker_init_fn=lambda k : seed_worker(rank * 1000 + k * 100 + seed), persistent_workers=True)
    
    current_best_mean_dice = 0
    for epoch in range(args.start_epoch, args.num_epochs):
        
        adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)
        train_sampler.set_epoch(epoch)
        
        ddp_model.train()
        for iter, batch in enumerate(trainloader):
            
            images = batch[0].to(device_id)
            labels = batch[1].to(device_id)
            annotated_categories = batch[2].to(device_id)
            annotated_categories_z_axis = batch[3].to(device_id)
            annotated_categories_y_axis = batch[4].to(device_id)
            annotated_categories_x_axis = batch[5].to(device_id)
            masks = batch[6].to(device_id)
            is_sparse = batch[7].to(device_id)
            
            preds = ddp_model(images)
            
            max_along_axis = torch.max(preds, dim=1, keepdim=True).values
            exp_logits = torch.exp(preds-max_along_axis)
            probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
            
            loss_wce, regularizer = loss_seg_CE(probs, labels, annotated_categories, annotated_categories_z_axis, annotated_categories_y_axis, annotated_categories_x_axis, masks, is_sparse=is_sparse)
            loss_dice = loss_seg_DICE(probs, labels, annotated_categories, annotated_categories_z_axis, annotated_categories_y_axis, annotated_categories_x_axis, masks, is_sparse=is_sparse)
            
            loss = 1.0 * loss_wce + 1.0 * loss_dice + 1.0 * regularizer
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if device_id == 0:
                print(
                    'Epoch {}: {}/{}, lr = {:.4}, loss = {:.4}, loss_wce = {:.4}, loss_reg = {:.4}, loss_dice = {:.4}'.format( \
                        epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], loss.item(), loss_wce.item(), regularizer.item(), loss_dice.item()))
        
        dist.barrier()
        
        if device_id == 0 and (epoch+1) % args.test_or_save_every_this_num_epochs == 0:
            
            if args.test_regularly_in_training:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
                
                # Validation #
                ddp_model.eval()
                with torch.no_grad():
                    current_best_mean_dice = test(args, ddp_model, val_subjects, args.snapshot_dir, epoch=epoch+1, current_best=current_best_mean_dice)
            else:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': ddp_model.module.state_dict()
                    }, args.snapshot_dir+'/model_ckpt_epoch{0}.pth'.format(epoch+1))
        
        dist.barrier()
    
    dist.destroy_process_group()
    
    if rank == 0:
        writer.close()

if __name__ == '__main__':
    main()
