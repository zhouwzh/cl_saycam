import json
import torch, torch.nn as nn
import math
import torch.nn.functional as F

from load_module import load_slotcontrast_model

from utils import read_vocab

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout, dim=1):
        if not (self.training and dropout):
            return x
        return x.new_empty(x.shape[:dim] + (1,) + x.shape[dim+1:]).bernoulli_(1 - dropout) / (1 - dropout) * x

    
class TextEncoder(nn.Module):
    """
    Text encoder
    """
    def __init__(self, config):
        super().__init__()
        self.text_encoder = config.get("text_encoder","embedding")
        self.embedding_type = config.get("embedding_type","flat")
        self.embedding_dim = config.get("embedding_dim",512)
        self.image_feature_map_dim = config.get("image_feature_map_dim")

        # always match embedding and hidden dim for consistency
        self.hidden_dim = self.embedding_dim
        self.input_dim = self.embedding_dim
        self.crange = config.get("crange", 1)
        self.dropout_i = config.get("dropout_i", 0.0)
        self.dropout_o = config.get("dropout_o", 0.0)
        self.pos_embed_type = config.get("pos_embed_type", 'no_pos_embed')

        # load vocab and create dict to map indices back to words
        self.vocab = read_vocab(config.get("vocab"))
        # self.word2idx = self.vocab
        self.idx2word = {idx: word for word, idx in self.vocab.items()}

        # build embedding layer
        self.embedding = nn.Embedding(len(self.vocab), self.embedding_dim,
                                      padding_idx=0)
        self.lockdrop = LockedDropout()
        self.output_dropout = nn.Dropout(self.dropout_o)
    
    def forward(self, x, x_len, image_features=None, image_feature_map=None):
        # x: (B, L)
        # x_len: (B,)
        attns = None

        embedding = self.embedding(x).squeeze(1)  # (B, L, E=512)

        if self.text_encoder == "embedding":
            raw_output = embedding  # (B, L, E)
            if self.embedding_type == "flat":  # flat embedding for embedding only model
                # calculate mean embedding per utterance
                ret = torch.sum(raw_output, dim=1) / x_len  # (B, E)

        output = self.lockdrop(raw_output, self.dropout_o)  # (B, L, E)

        if self.embedding_type == "flat":
            ret = self.output_dropout(ret)  # (B, E)
        elif self.embedding_type == "spatial":
            ret = output

        return ret, output, attns


class SCModel_Wrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = load_slotcontrast_model(args)
        self.feat_fc = nn.Identity()
        self.slot_fc = nn.Identity()
        self.pooling = args['pooling']
        self.ret_type = args.get('ret_type', 'feat')  # slot or feat

    def forward(self, x): # input:[B,3,224,224] output:[B, 512]
        feats, slots = None, None
        inputs = {
            'video': x.unsqueeze(1),
            'batch_padding_mask': torch.tensor(False).repeat(x.shape[0]).to(x.device)
        }
        output = self.backbone(inputs)
        backbone_features = output['encoder']['backbone_features']  # [B, T, 256, 768]  -> [B, T, 16,16, 768]
        slots = output['processor']['corrector']['slots'].squeeze(1)  # slots: [B, 1, 11, 64] -> [B, 11, 64]
        masks = output['processor']['corrector']['masks']   # [B,1,11,256]

        if self.ret_type == 'feat':
            attns = masks.transpose(-1,-2)
            B,T,P,K= attns.size()
            D = backbone_features.size(-1)

            H_enc = int(math.isqrt(P))
            attns = attns\
                .transpose(-1,-2)\
                .reshape(B, T, K, 1, H_enc, H_enc)\
                .squeeze(3).unsqueeze(-1)  # [B, T, K, H_enc, H_enc, 1]

            feats = backbone_features.view(B,T,H_enc, H_enc,D) \
                .unsqueeze(2).expand(-1, -1, K, -1, -1, -1)  # [B, T, K, H_enc, H_enc, D]
            weighted_feats = attns * feats + (1.0 - attns)  # [B, T, K, H_enc, H_enc, D]

            feats = weighted_feats.mean(dim=(1,3,4)) # [B,K,D]
            feats = self.feat_fc(feats)
        return feats, self.slot_fc(slots)

import numpy as np
import spacy

class Multimodalmodel(nn.Module):
    """
    Multimodal model
    """
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts, text_len):
        image_features, _ = self.vision_encoder(images)  # (B,K,D)
        text_features, text_outputs, _ = self.text_encoder(texts, text_len)  # (B, 512), (B, L, 512)
        return image_features, text_features
    
    def tokenize(self, texts):
        """Tokenize texts to obtain tokens and token lengths"""
        nlp = spacy.load("en_core_web_sm")
        vocab = self.text_encoder.vocab
        max_seq_len = 25

        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        token_lengths = []

        for text in texts:
            doc = nlp(text.lower())
            word_tokens = [token.text for token in doc]
            
            if len(word_tokens) > max_seq_len - 2:
                word_tokens = word_tokens[:max_seq_len - 2]
            token_length = len(word_tokens) + 2  # for SOS and EOS
            tokens = [vocab["<sos>"]] + \
                [vocab.get(token, vocab["<unk>"]) for token in word_tokens] + \
                [vocab["<eos>"]] + [vocab["<pad>"]] * (max_seq_len - len(word_tokens) - 2)
            all_tokens.append(tokens)
            token_lengths.append(token_length)
        
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        token_lengths = torch.tensor(token_lengths, dtype=torch.long)
        return tokens, token_lengths





        