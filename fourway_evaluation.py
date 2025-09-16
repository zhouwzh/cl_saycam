import os,sys
import argparse
import torch
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import *

from omegaconf import OmegaConf

from models import TextEncoder, Multimodalmodel, SCModel_Wrapper
from utils import set_seed, mkdir, setup_logger, load_config_file


parser = argparse.ArgumentParser()
parser.add_argument('--json_path',type=str,default="/home/wz3008/clip/config/eval_dev.json")
parser.add_argument('--img_root',type=str,default="/mnt/wwn-0x5000c500e421004a/yy2694/datasets/saycam_labeled")
parser.add_argument("--model_config", default="/home/wz3008/clip/config/model_config.yaml", type=str, help="path of model config file")
# parser.add_argument('--model',type=str,default='cvcl')
parser.add_argument('--checkpoint',type=str,default="/home/wz3008/clip/saved_checkpoints/2025-09-16-04-02-54_topk/best_ckpt.pth")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    with open(args.json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    data = obj["data"]

    model_config = load_config_file(args.model_config)
    text_encoder = TextEncoder(model_config.text_encoder)
    sc_args = {"config": model_config.slotcontrast.config,"continue_from": model_config.slotcontrast.continue_from,"config_overrides_file": None,"config_overrides": None,"pooling": "mean","ret_type": "feat"}
    vision_encoder = SCModel_Wrapper(sc_args)
    vision_encoder.feat_fc = torch.nn.Linear(in_features=768, out_features=512, bias=True)  # B,K,512
    vision_encoder.slot_fc = torch.nn.Linear(in_features=64, out_features=512, bias=True)  # B,K,512
    model = Multimodalmodel(vision_encoder, text_encoder)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)

    model.eval().to(device)       

    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    classes = ['ball', 'basket', 'car', 'cat', 'chair', 'computer', 'crib', 'door', 'floor', 'foot', 'ground', 'hand', 'kitchen', 'paper', 'puzzle', 'road', 'room', 'sand', 'stairs', 'table', 'toy', 'window']
    count = {c:0 for c in classes}
    acc = {c:0 for c in classes}
    to_tensor = transforms.ToTensor()
    new_classes = ['ball', 'basket', 'car', 'cat', 'chair', 'computer', 'crib', 'door', 'floor', 'foot', 'ground', 'hand', 'kitchen', 'paper', 'puzzle', 'road', 'room', 'sand', 'stairs', 'table', 'toy', 'window']
    new_class_to_idx = {'ball': 0, 'basket': 1, 'car': 2, 'cat': 3, 'chair': 4, 'computer': 5, 'crib': 6, 'door': 7, 'floor': 8, 'foot': 9, 'ground': 10, 'hand': 11, 'kitchen': 12, 'paper': 13, 'puzzle': 14, 'road': 15, 'room': 16, 'sand': 17, 'stairs': 18, 'table': 19, 'toy': 20, 'window': 21}
    # import pdb; pdb.set_trace()
    for i in tqdm(range(len(data))):
        target = data[i]['target_class']
        img_paths = data[i]['image_path']

        count[target] += 1

        images = []
        for path in img_paths:
            image = Image.open(args.img_root+"/"+path).convert("RGB")
            image = to_tensor(image)
            image = normalizer(image)
            images.append(image)
        # images = [transforms.ToTensor(exitImage.open(args.img_root+"/"+path).convert("RGB").resize((224,224))) for path in img_paths]
        images = torch.stack(images, dim=0).to(device)

        text_inputs, text_len = model.tokenize(target)
        text_inputs, text_len = text_inputs.to(device), text_len.to(device)
        image_features, text_features = model(images, text_inputs, text_len) # B,11,512  B,512

        sim = (image_features * text_features.unsqueeze(1)).sum(-1)
        sim = (image_features * text_features.unsqueeze(1)).sum(-1)
        weights = torch.nn.functional.softmax(sim, dim=1)
        image_features = (weights.unsqueeze(-1) * image_features).sum(1)
        
        logit_scale = model.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text  = logit_scale * text_features @ image_features.t()

        if logits_per_text.argmax(dim=1) == 0:
            acc[target] += 1
            
    
    avg_acc = 0
    for cls in classes:
         acc[cls] = acc[cls] / count[cls]
         avg_acc += acc[cls]
    print(f"avg_acc = {avg_acc * 100 / len(classes)}%")

        
if __name__ == "__main__":
    main()
