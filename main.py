import argparse
import open_clip.loss
import os
import open_clip
import torch
import numpy as np
import random

from torch.utils.data import DataLoader
from train import train_one_epoch, evaluate, itm_eval
from loguru import logger
from optimizers import creat_optimizer, create_scheduler
from data.re_dataset import re_eval_dataset, re_train_dataset

def run(args):

    ##### BUILD CLIP MODEL #####
    CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        cache_dir='cache/weights/open_clip'
    )

    ##### LOAD CHECKPOINT ######
    if args.checkpoint != '-1':
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        CLIP_model.load_state_dict(checkpoint)
        logger.info(f'Loaded from {args.checkpoint}')
    
    criterion = open_clip.loss.ClipLoss()
    optimizer = creat_optimizer(CLIP_model, args)
    scheduler = create_scheduler(optimizer, args)
    ##### LOAD DATASET ######
    if args.dataset == 'rsitmd':
        train_set = re_train_dataset(ann_file=['./data/finetune_json/rsitmd_train.json'], transform=preprocess_train, image_root='./dataset/rsitmd')
        test_set = re_eval_dataset(ann_file='./data/finetune_json/rsitmd_test.json', transform=preprocess_val, image_root='./dataset/rsitmd')
    else:
        raise NotImplementedError
    
    train_loader = DataLoader(train_set, batch_size=args.bs, pin_memory=True, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=args.test_bs, pin_memory=True, shuffle=False, num_workers=12)  

    logger.info('Start Training')
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Training Epoch: {epoch}")
        
        train_one_epoch(CLIP_model, train_loader, optimizer, criterion, epoch, args.device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # For ReduceLROnPlateau, we need to provide a metric to base the decision on
            score_test_i2t, score_test_t2i = evaluate(CLIP_model, test_loader, args.device, args)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            # Use r_mean as the metric for ReduceLROnPlateau
            scheduler.step(test_result['r_mean'])
        else:
            # For other schedulers, just step
            scheduler.step()
              
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Current learning rate: {current_lr}')

        score_test_i2t, score_test_t2i = evaluate(CLIP_model, test_loader, args.device, args)
        test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)

        for key, value in test_result.items():
            logger.info(f'{key}: {value}')
        
        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': CLIP_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'test_results': test_result
            }, save_path)
            
            logger.info(f'Saved checkpoint to {save_path}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='GeoRSCLIP_FT')
    parser.add_argument('--dataset', type=str, default='rsitmd', )
    parser.add_argument('--pretrained', choices=['openai', 'geoclip','-1'], default='openai')
    parser.add_argument(
        "--model-name", type=str,default='ViT-B-32',
        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
        help="Name of backbone. In open_clip.list_models() or hugging face transformers",
    )
    
    ### LR——SCHREDULE
    parser.add_argument('--lr_scheduler', type=str, default='cosine', 
                    choices=['cosine', 'step', 'plateau'])  
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epochs', type=int, default=5)

    ### Optimizer parameters
    parser.add_argument('--lr', default=4e-6, type=float, help="learning rate")

    parser.add_argument('--epochs', default=40, type=int, help="epochs")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--test_bs', default=256, type=int)
    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--checkpoint', default='./pretrained_dir/RS5M_ViT-B-32_RET-2.pt', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default=' ', type=str, help="load domain pre-trained params")
    parser.add_argument('--output_dir', type=str, default='./outputs/ft_clip', help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation on downstream tasks")
    parser.add_argument('--save_freq', type=int, default=25, help="save frequency")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, f'{args.task}_lr{args.lr}_bs{args.bs}_training.log')
    # Open file in write mode to clear content
    with open(log_path, 'w') as f:
        pass
    logger.add(log_path, 
               rotation="500 MB", 
               retention="10 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.warning(args)
    run(args)