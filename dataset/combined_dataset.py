from torch.utils.data import ConcatDataset as _ConcatDataset

from dataset.lmdb_dataset import LmdbDataset


class ConcatDatasets(_ConcatDataset):
    """ Concat different datasets.

        Args:
        datasets (list[dict]): A list of which each elements is a dataset cfg.
        batch_ratio (list[float]): Ratio of corresponding dataset will be used
                in constructing a batch. It makes effect only with balance
                sampler.
        **kwargs:

    """

    def __init__(self,
                 datasets: list,
                 batch_ratio: list = None,
                 **kwargs):
        assert isinstance(datasets, list)

        self.root = ''.join([ds.root for ds in datasets])
        data_range = [len(dataset) for dataset in datasets]
        self.data_range = [
            sum(data_range[:i]) for i in range(1,
                                               len(data_range) + 1)
        ]
        self.batch_ratio = batch_ratio
        if self.batch_ratio is not None:
            assert len(self.batch_ratio) == len(datasets), \
                'The length of batch_ratio and datasets should be equal. ' \
                f'But got {len(self.batch_ratio)} batch_ratio and ' \
                f'{len(datasets)} datasets.'
        super(ConcatDatasets, self).__init__(datasets=datasets)


if __name__ == "__main__":
    character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲ'
    max_length = 25
    data_root = "/media/shubham/One Touch/Indic_OCR/recognition_dataset/hindi/training/ST/"
    data_root2 = "/media/shubham/One Touch/Indic_OCR/recognition_dataset/hindi/training/MJ/MJ_train/"
    dataset = LmdbDataset(data_root, character=character, batch_max_length=max_length)
    dataset1 = LmdbDataset(data_root2, character=character, batch_max_length=max_length)

    dataset2 = ConcatDatasets([dataset, dataset1])

    print(len(dataset))
    print(len(dataset1))
    print(len(dataset2))
