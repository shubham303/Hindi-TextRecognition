# modify from clovaai

import logging
from random import random

import lmdb
import numpy as np
import six
import torch.utils.data
from PIL import Image
from torchvision import transforms

from dataset.base import BaseDataset

logger = logging.getLogger()


class LmdbDataset(BaseDataset):
    """ Read the data of lmdb format.
    Please refer to https://github.com/Media-Smart/vedastr/issues/27#issuecomment-691793593  # noqa 501
    if you have problems with creating lmdb format file.

    """

    def __init__(self,
                 root: str,
                 transform=None,
                 character: str = 'abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length: int = 100000,
                 max_tokens=10,
                 data_filter: bool = True, ):

        self.index_list = []
        self.is_valid_index = []
        super(LmdbDataset, self).__init__(
            root=root,
            transform=transform,
            character=character,
            batch_max_length=batch_max_length,
            data_filter=data_filter,
            max_tokens=max_tokens
        )

    def get_name_list(self):
        self.env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))
            n_samples = min(1000, n_samples)

            import tqdm
            for index in tqdm.tqdm(range(1, n_samples)):
                idx = index + 1
                label_key = 'label-%09d'.encode() % idx
                label = txn.get(label_key)

                filter, l = self.filter(label.decode('utf-8'))
                if label is None or filter:
                    continue
                else:
                    self.index_list.append(idx)

        self.samples = len(self.index_list)

    def read_data(self, index, txn):

        assert index <= len(self), 'index range error'
        index = self.index_list[index]

        # read next data item if data item is not valid.
        if self.is_valid_index[index] == 0:
            return self.read_data((index + 1) % self.samples, txn)

        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)

        if label is None or len(label) == 0:
            self.is_valid_index[index] = 0
            return self.read_data((index + 1) % self.samples, txn)

        if self.is_valid_index[index] == -1:  # if label is not verified yet
            label = label.decode('utf-8')
            filter, label = self.filter(label)

            if filter:
                self.is_valid_index[index] = 0
                return self.read_data((index + 1) % self.samples, txn)
            else:
                self.is_valid_index[index] = 1
                self.labels[index] = label

        label = self.labels[index]

        img_key = 'image-%09d'.encode() % index

        imgbuf = txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')  # for color image
        img = np.array(img)

        return img, label

    def __getitem__(self, index):

        assert index < len(self)
        index = self.index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')  # for color image
                img = np.array(img)
            except IOError:
                print(f'Corrupted image for {index}')
                img, label = self.__getitem__(random.choice(range(len(self))))
                return img, label

            img = self.transforms(img)
            return img, label


if __name__ == "__main__":
    character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲ'
    max_length = 25
    data_root = "/media/shubham/One Touch/Indic_OCR/recognition_dataset/hindi/training/MJ/MJ_train"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ]
    )

    dataset = LmdbDataset(data_root, character=character, batch_max_length=max_length, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5000)

    mean = None
    import tqdm
    for  (images, labels) in tqdm.tqdm(dataloader):
        images = images.cuda()
        if mean is None:
            mean = torch.sum(images, dim=(0,2,3))/ (224*224)
        else:
            mean  = mean+ torch.sum(images, dim=(0,2,3))/ (224*224)

    print(mean / len(dataset))
