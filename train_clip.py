import torch
import torch.nn.functional as F
import numpy as np
import os
from omegaconf import OmegaConf

# from utils.simple_tokenizer import SimpleTokenizer
from utils.custom_schedulers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from utils import set_seed, mkdir, setup_logger, load_config_file

from torch.optim import Adam, AdamW # both are same but AdamW has a default weight decay

from models import TextEncoder, Multimodalmodel, SCModel_Wrapper
from load_module import *
from data import create_datasets, get_dataloader

import argparse
from tqdm import *
from datetime import datetime
import math


def train(config, train_dataset, model):
    '''
    Trains the model.
    '''
    
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)    
    train_dataloader = get_dataloader(config, train_dataset, is_train=True)

    # total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps, weight_decay=config.optimizer.params.weight_decay)

    # Warmup iterations = 20% of total iterations
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= t_total)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(torch.device(config.device))
    model.train()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Number of GPUs = %d", config.n_gpu)

    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   config.train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if scheduler:
        logger.info("  warmup steps = %d", num_warmup_steps)


    global_step, global_loss, global_acc =0,  0.0, 0.0
    best_loss = math.inf
    model.zero_grad()

    for epoch in tqdm(range(int(config.num_train_epochs))):
        for step, batch in enumerate(train_dataloader):
            input_images, input_texts, text_len = batch

            input_images = input_images.to(torch.device(config.device))
            input_texts = input_texts.to(torch.device(config.device))
            text_len = text_len.to(torch.device(config.device))
            
            image_features, text_features = model(input_images, input_texts, text_len) # B,11,512  B,512

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if config.n_gpu == 1:
                logit_scale = model.logit_scale.exp()
            elif config.n_gpu > 1:
                logit_scale = model.module.logit_scale.exp()
            

            if epoch < 10:
                B,K,D = image_features.size()  # B,K,D
                image_features_flat = image_features.reshape(B * K, D)
                logits_per_slot = logit_scale * image_features_flat @ text_features.t() # B*K=704, 64
                logits_per_text = logit_scale * text_features @ image_features_flat.t() # 64, B*K=704
                # slots_labels = torch.arange(B).repeat_interleave(K).to(image_features.device)  # 704
                # text_labels = torch.arange(B).repeat_interleave(K).to(image_features.device)   # 704
                target_text = torch.zeros(B, B*K, device=image_features.device)
                for i in range(B):
                    target_text[i, i*K:(i+1)*K] = 1.0 / K
                target_slot = torch.zeros(B*K, B, device=image_features.device)
                for i in range(B):
                    target_slot[i*K:(i+1)*K, i] = 1.0
                loss_text = -(target_text * F.log_softmax(logits_per_text, dim=-1)).sum(dim=1).mean()
                loss_slot = -(target_slot * F.log_softmax(logits_per_slot, dim=-1)).sum(dim=1).mean()
                loss = (loss_text + loss_slot) / 2
            else:
                if config.slot_mode =='soft_align':
                    sim = (image_features * text_features.unsqueeze(1)).sum(-1)  # B,K
                    weights = F.softmax(sim, dim=1)  # B,K
                    image_features = (weights.unsqueeze(-1) * image_features).sum(1)   # B,D

                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text  = logit_scale * text_features @ image_features.t()
                    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
                    image_loss = F.cross_entropy(logits_per_image, labels)
                    text_loss  = F.cross_entropy(logits_per_text, labels)

                    loss = (image_loss + text_loss) / 2
                elif config.slot_mode == 'topk':
                    # import pdb; pdb.set_trace()
                    B,K,D = image_features.size()  # B,K,D
                    sim = (image_features * text_features.unsqueeze(1)).sum(-1)  # B,K
                    k = 3
                    topk_values, topk_idx = sim.topk(k, dim=1)  # B,k
                    topk_list = []
                    for i in range(B):
                        idx = topk_idx[i]  # k
                        topk_list.append(image_features[i, idx, :])
                    topk_slots = torch.stack(topk_list, dim=0)  # B,k,D
                    # batch_idx = torch.arange(B).unsqueeze(-1).expand(-1, k) # B,k
                    # topk_slots = image_features[batch_idx, topk_idx]  # B,k,D
                    topk_slots = topk_slots.reshape(B * k, D)  # B*k,D
                    logits_per_slot = logit_scale * topk_slots @ text_features.t()  # B*k, 64
                    logits_per_text = logit_scale * text_features @ topk_slots.t()  # 64, B*k
                    target_text = torch.zeros(B, B*k, device=image_features.device)
                    for i in range(B):
                        target_text[i, i*k:(i+1)*k] = 1.0 / k
                    target_slot = torch.zeros(B*k, B, device=image_features.device)
                    for i in range(B):
                        target_slot[i*k:(i+1)*k, i] = 1.0
                    loss_text = -(target_text * F.log_softmax(logits_per_text, dim=-1)).sum(dim=1).mean()
                    loss_slot = -(target_slot * F.log_softmax(logits_per_slot, dim=-1)).sum(dim=1).mean()
                    loss = (loss_text + loss_slot) / 2

            if config.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            global_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step() # PYTORCH 1.x : call optimizer.step() first then scheduler.step()
                
                # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
                if config.n_gpu == 1:
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                elif config.n_gpu > 1:
                    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

                if scheduler:
                    scheduler.step() 
                    
                model.zero_grad()

                if global_step % config.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step)
                    )

                if (config.save_steps > 0 and global_step % config.save_steps == 0) or \
                        global_step == t_total:
                    # saving checkpoint
                    best = False
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best = True
                    save_checkpoint(config, epoch, global_step, model, optimizer, loss.item(), best) 
                    

    return global_step, global_loss / global_step


def save_checkpoint(config, epoch, global_step, model, optimizer, loss, best):
    checkpoint_dir = os.path.join(config.saved_checkpoints)
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, "best_ckpt.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Saved checkpoint at {checkpoint_path}")

def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='slotcontrast', type=str, choices=['slotcontrast'])
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--gpus", type=int, help="list of gpu ids to use. if None, use all available gpus.")
    parser.add_argument("--data_config", default=None, type=str, help="path of data config file")
    parser.add_argument("--train_config", default=None, type=str, help="path of trainer config file")
    parser.add_argument("--model_config", default=None, type=str, help="path of model config file")

    # if change img_dir and annotation_file in cml instead config file
    parser.add_argument("--train_img_dir", default=None, type=str, required=False, help="path of directory containing training images")
    parser.add_argument("--train_annotation_file", default=None, type=str, required=False, help="path of annotation file")
    parser.add_argument("--exp_name", default="debug", type=str, help="name of the experiment, used for saving checkpoints and logs")

    
    args = parser.parse_args()

    data_config = load_config_file(args.data_config)
    train_config = load_config_file(args.train_config)
    model_config = load_config_file(args.model_config)

    config = OmegaConf.merge(train_config, data_config)

    # config = OmegaConf.merge(OmegaConf.create(vars(args)), config)  
    # merging cli arguments, if data path given in cli args use those
    if args.train_img_dir : 
        config.train_img_dir = args.train_img_dir
    if args.train_annotation_file : 
        config.train_annotation_file = args.train_annotation_file
        

    global logger
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # creating directories for saving checkpoints and logs
    config.saved_checkpoints = os.path.join(config.saved_checkpoints, f"{current_time}_{args.exp_name}")
    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    
    logger = setup_logger("saycam_TRAIN", config.logs, 0, filename = f"training_logs_{current_time}.txt")

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gpus:
        config.n_gpu = args.gpus
    else:
        config.n_gpu = torch.cuda.device_count() # config.n_gpu 
    set_seed(seed=args.seed, n_gpu=config.n_gpu)

    # getting text tokenizer
    text_encoder = TextEncoder(model_config.text_encoder)
    
    if args.arch == 'slotcontrast':
        sc_args = {
            "config": model_config.slotcontrast.config,
            "continue_from": model_config.slotcontrast.continue_from,
            "config_overrides_file": None,
            "config_overrides": None,
            "pooling": "mean",
            "ret_type": "feat"
        }
        vision_encoder = SCModel_Wrapper(sc_args)
        set_parameter_requires_grad(vision_encoder)
        vision_encoder.feat_fc = torch.nn.Linear(in_features=768, out_features=512, bias=True)  # B,K,512
        vision_encoder.slot_fc = torch.nn.Linear(in_features=64, out_features=512, bias=True)  # B,K,512
    
    model = Multimodalmodel(vision_encoder, text_encoder)
        
    logger.info(f"Training/evaluation parameters {train_config}")

    # getting dataset for training
    # train_dataset = CLIP_COCO_dataset(config, tokenizer)
    train_dataset = create_datasets(config)['train']

    # Now training
    global_step, avg_loss = train(config, train_dataset, model)
    
    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)
    

if __name__ == "__main__":
    main()