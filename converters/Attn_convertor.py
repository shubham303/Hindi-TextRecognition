# modify from clovaai

import torch

from abfn_package import abfn
from converters.base_convert import BaseConverter


class Attnconverter(BaseConverter):
    """
    This converter is very similar to Attn_converter, Except it adds one extra special character [x] to express token
    end.
    """

    def __init__(self, character, batch_max_length):

        """
        language_list is used for multilingual model. for single language model list contains one element.
        """
        list_character = list(character)
        self.batch_max_length = batch_max_length + 1

        list_token = ['[GO]', '[s]']
        character = list_token + list_character
        super(Attnconverter, self).__init__(character=character)
        self.ignore_index = self.dict['[GO]']

    def train_encode(self, texts):

        if type(texts) is str:
            texts = [texts]
        length = []  # [len(s) + 1 for s in text]
        batch_text = torch.LongTensor(len(texts), self.batch_max_length + 1).fill_(self.ignore_index)  # noqa 501
        language_id = torch.LongTensor(len(texts))



        for idx, t in enumerate(texts):
            text = [self.dict[char] for char in t]
            text.append(self.dict['[s]'])

            try:
                batch_text[idx][1:1 + len(text)] = torch.LongTensor(text)
            except:
                print(text, t)
            length.append(len(text))

        batch_text_input = batch_text[:, :-1]
        batch_text_target = batch_text[:, 1:]

        return batch_text_input, torch.IntTensor(length), batch_text_target, language_id

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)

        return texts


if __name__ == "__main__":
    w = ["शुभम"]
    character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲ'
    max_length = 25
    c = Attnconverter(character, max_length)
    out = c.train_encode(w)
    print(out[0].shape)
    print(c.decode(out[2]))