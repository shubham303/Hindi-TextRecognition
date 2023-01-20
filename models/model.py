import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import pytorch_lightning as pl

from converters.Attn_convertor import Attnconverter
from models.PositionalEncoding import PositionalEncoding
import torch.nn.functional as F


class STRHindi(pl.LightningModule):
    def __init__(self, image_size=224, hidden_size=384, num_of_tokens=10, num_char_per_token=7, num_characters=114,
                 dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = num_of_tokens * num_char_per_token
        self.feature_extractor_model = ViTModel.from_pretrained('facebook/dino-vits16')
        self.image_size = image_size
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 4)
        self.text_embedding = nn.Embedding(num_characters, hidden_size)
        self.linear = nn.Linear(hidden_size, num_characters)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

    def training_step(self, batch, batch_idx, tgt_mask, memory_mask=None):
        images, text = batch
        features = self.feature_extractor_model(images).last_hidden_state
        # generate text embedding
        text_embedding = self.text_embedding(text)
        # add positional embedding
        text_embedding = self.positional_encoding(text_embedding)

        out = self.decoder(text_embedding, features, tgt_mask=tgt_mask, memory_mask=memory_mask)
        out = self.linear(out)

        loss = F.cross_entropy(out.contiguous().view(-1, out.size(-1)), text.contiguous().view(-1))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.00005)
        return optimizer



if __name__ == "__main__":
    resnet = STRHindi().cuda()
    w = ["शुभम"]
    character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲ'
    max_length = 25
    c = Attnconverter(character, max_length)
    out = c.train_encode(w)
    x = torch.randn(1, 3, 224, 224)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(out[2].size(1)).cuda()  # .expand(images.size(0),-1, -1)
    print(resnet.training_step((x.cuda(), out[2].cuda()), 0, tgt_mask))
