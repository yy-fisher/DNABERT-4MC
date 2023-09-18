from torch.utils.data import DataLoader, TensorDataset, Dataset
from  Datapreprocessing_train import *
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re

tokenizer = AutoTokenizer.from_pretrained("/dnabert3/", do_lower_case=False )
class DeepLocDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, split="train", tokenizer_name='C:/Users/yy/Desktop/dnabert3/', max_length=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #         self.seq_list = df['seq_2_1'].values
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_length = max_length
        if split == "train":
            self.seqs, self.labels = self.load_dataset(train_index)
        else:
            self.seqs, self.labels = self.load_dataset(valid_index)

    def load_dataset(self, index):
        seqs = []
        for i in index:
            result = seq_dataset_bert.iloc[i]['seq']
            seqs.append(result)
        seqs_label = []
        for i in index:
            seqs_label.append(seq_dataset_bert.iloc[i]['label'])
        return seqs, seqs_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.seqs[idx]
        #         seq_feature = self.seq_list[idx]
        seq = re.sub(r"[UZOB]", "X", seq)
        encoding = self.tokenizer.encode_plus(seq, truncation=True, padding='max_length', max_length=self.max_length,
                                              pad_to_max_length=True, return_token_type_ids=True,
                                              return_attention_mask=True, return_tensors='pt')
        sample = {"texts": seq}
        sample['labels'] = torch.tensor(self.labels[idx])
        #         with torch.no_grad():
        #             embedding = model(input_ids=encoding["input_ids"],attention_mask=encoding["attention_mask"])[0]

        return seq, torch.tensor(self.labels[idx])



def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果，例如：
             src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
             target：[1, 1, 0, ...]
    """
    text, target = zip(*batch)
    text, target= list(text), list(target)
    # src是要送给bert的，所以不需要特殊处理，直接用tokenizer的结果即可
    # padding='max_length' 不够长度的进行填充
    # truncation=True 长度过长的进行裁剪
    src = tokenizer(text, padding='max_length', max_length=38, return_tensors='pt', truncation=True)

    return src, torch.LongTensor(target)

train_dataset = DeepLocDataset(split="train", tokenizer_name='dnabert3/',max_length=38)  # max_length is only capped to speed-up example.
val_dataset = DeepLocDataset(split="valid", tokenizer_name='dnabert3/', max_length=38)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=collate_fn,drop_last = True)
validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,collate_fn=collate_fn,drop_last = True)