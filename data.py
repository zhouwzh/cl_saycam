import os.path as op
import random
import spacy
import torch
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from omegaconf import OmegaConf

from utils import read_json, read_vocab, load_data

from typing import Any, Tuple
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x    
    
class SAYCAMDataset(Dataset):
    """
    Dataset that returns paired image-utterances from baby S of the SAYCam Dataset.
    """
    def __init__(self, data, vocab, config, multiple_frames):
        super().__init__()
        self.img_dir = config.get("train_img_dir", "")
        self.data = data
        self.vocab = vocab
        image_size = config.get("image_size", 224)
        self.image_size = image_size
        self.multiple_frames = multiple_frames
        self.augment_frames = config.get("augment_frames", False)
        self.transform = None
        if self.augment_frames:
            # add same augmentations as emin used
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(0.2, 1.)),
                # transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalizer,
            ])
        self.nlp = spacy.load("en_core_web_sm")
    

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances)
        """
        # import pdb; pdb.set_trace()
        # get utterance and convert to indices
        utterance = self.data[idx]["utterance"]
        # utterance_words = utterance.split()
        # utterance_words = [SOS_TOKEN] + utterance_words + [EOS_TOKEN]
        # utterance_length = len(utterance_words)
        # utterance_idxs = torch.tensor([self.vocab.get(
        #     word, UNK_TOKEN_ID) for word in utterance_words], dtype=torch.long)

        # get image
        img_filenames = self.data[idx]["frame_filenames"]

        if self.multiple_frames:
            # sample a random image associated with this utterance
            img_filename = Path(self.img_dir, random.choice(img_filenames))
        else:
            # otherwise, sample the first frame
            img_filename = Path(self.img_dir, img_filenames[0])

        img = Image.open(img_filename).convert("RGB")

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        text_inputs, text_len = self.tokenize(utterance)

        return img, text_inputs, text_len
        # return img, utterance_idxs, utterance_length, [utterance]
    
        # img_path = op.join(self.img_dir, img_filename)
        # img = Image.open(img_path)
        # img_input = self.transform(img)
        # text_input = self.tokenize(text)

        # return img_input, text_inputs
    
    def tokenize(self, texts):
        """Tokenize texts to obtain tokens and token lengths"""
        max_seq_len = 25

        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        token_lengths = []

        for text in texts:
            doc = self.nlp(text.lower())
            word_tokens = [token.text for token in doc]
            
            if len(word_tokens) > max_seq_len - 2:
                word_tokens = word_tokens[:max_seq_len - 2]
            token_length = len(word_tokens) + 2  # for SOS and EOS
            tokens = [self.vocab["<sos>"]] + \
                [self.vocab.get(token, self.vocab["<unk>"]) for token in word_tokens] + \
                [self.vocab["<eos>"]] + [self.vocab["<pad>"]] * (max_seq_len - len(word_tokens) - 2)
            all_tokens.append(tokens)
            token_lengths.append(token_length)
        
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        token_lengths = torch.tensor(token_lengths, dtype=torch.long)
        return tokens, token_lengths


def create_datasets(config):
    shuffle_utterances = config.get("shuffle_utterances", False)
    multiple_frames = config.get("multiple_frames", False)
    train_metadata_filename = config.train_annotation_file
    val_metadata_filename = config.val_annotation_file
    test_metadata_filename = config.test_annotation_file
    vocab = read_vocab(config.get("vocab"))

    datasets = {}
    if shuffle_utterances:
        pass
    else:
        print("Training using matched utterances!")
        stage_splits = [
            ("train", train_metadata_filename, multiple_frames),
            ("val", val_metadata_filename, False),
            ("test", test_metadata_filename, False),
        ]
        for split, filename, multiple_frames in stage_splits:
            data = load_data(filename)
            dataset = SAYCAMDataset(
                data,
                vocab,
                config,
                multiple_frames=multiple_frames,
            )
            datasets[split] = dataset
    return datasets

def get_dataloader(config, dataset, is_train = True):
    
    if is_train:
        sampler = RandomSampler(dataset)
        batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    else:
        sampler = SequentialSampler(dataset)
        batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)

    dataloader = DataLoader(dataset, sampler=sampler, 
            batch_size=batch_size, num_workers=config.num_workers)

    return dataloader