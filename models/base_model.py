import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import pytorch_lightning as pl

from converters.Attn_convertor import Attnconverter
from models.PositionalEncoding import PositionalEncoding
import torch.nn.functional as F

from models.unet import UNet


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convertor = None

    def encode(self, labels):
        return self.converter.train_encode(labels)

    def postprocess(self, preds):
        probs = F.softmax(preds, dim=2)
        max_probs, indexes = probs.max(dim=2)

        preds_str = []
        preds_prob = []
        for i, pstr in enumerate(self.converter.decode(indexes)):
            str_len = len(pstr)
            if str_len == 0:
                prob = 0
            else:
                prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
            preds_prob.append(prob)
            preds_str.append(pstr)

        return preds_str, preds_prob




