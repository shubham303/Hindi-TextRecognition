import torch
import torch.nn.functional as F
import tqdm
from torchvision import transforms

import converters.Attn_Indic_converter
import options.train_options
from dataset.combined_dataset import ConcatDatasets
from dataset.lmdb_dataset import LmdbDataset
from metrics.accuracy import Accuracy
from models.model import STRHindi
import wandb
from models.model_unetbased import STRHindiUnet
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def load_model(model_name, num_characters):
    return STRHindiUnet(num_characters=num_characters)


def postprocess(preds, converter):
    probs = F.softmax(preds, dim=2)
    max_probs, indexes = probs.max(dim=2)

    preds_str = []
    preds_prob = []
    for i, pstr in enumerate(converter.decode(indexes)):
        str_len = len(pstr)
        if str_len == 0:
            prob = 0
        else:
            prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
        preds_prob.append(prob)
        preds_str.append(pstr)

    return preds_str, preds_prob


def train():
    args_parser = options.train_options.get_train_options()
    args = args_parser.parse_args()
    print(args)
    device = torch.device("cuda:0")
    converter = converters.Attn_convertor.Attnconverter(args.character, args.batch_max_length)
    model = load_model(args.model_name, num_characters=len(converter.character, characters=args.character)).to(device)

    # TODO define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    )

    datasets = []
    if args.mj_root != "":
        # number of characters is batch_max_len
        datasets.append(
            LmdbDataset(args.mj_root, character=args.character, batch_max_length=args.word_len, transform=transform,
                        max_tokens=args.batch_max_length // args.char_per_token))

    if args.st_root != "":
        datasets.append(
            LmdbDataset(args.st_root, character=args.character, batch_max_length=args.word_len, transform=transform,
                        max_tokens=args.batch_max_length // args.char_per_token))

    dataset = ConcatDatasets(datasets, [0.5, 0.5])
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    metric = Accuracy()

    trainer = pl.Trainer(max_epochs=10, limit_train_batches=0.1)
    trainer.fit(model=model, train_dataloaders=dataloader)



wandb.init(name="first experiment")
train()
wandb.finish()
