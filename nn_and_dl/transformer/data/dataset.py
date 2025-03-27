import spacy
import os
import torch
import copy
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, pad

class TranslateDataset(Dataset):
    def __init__(self,
        en_file: str,
        zh_file: str,
    ):
        """
        初始化翻译器类
        
        Args:
            en_file (str): 英文句子的文件路径
            zh_file (str): 中文句子的文件路径     
        Raises:
            AssertionError: 如果中英文句子的数量不匹配，则抛出异常
        """
        self.zh_sentences = self._load_file(zh_file)
        self.en_sentences = self._load_file(en_file)

        assert len(self.zh_sentences) == len(self.en_sentences), "文件函数不匹配"

    def _load_file(self, file_path):
        sentences = []
        i=0
        with open(file_path, "r", encoding = 'utf-8') as f:
            for line in f:
                sentences.append(line.strip())
                i+=1
                if i>=10000:
                    break

        return sentences

    def __getitem__(self, idx: int) -> tuple[str]:
        return (self.en_sentences[idx], self.zh_sentences[idx])

    def __len__(self):
        return len(self.zh_sentences)

class TranslateDataloader(object):
    def __init__(self, 
                 dataset, # dataset 类似于multi30k的格式
                 tokenize_src, # tokenize
                 tokenize_tgt, 
                 vocab_src, 
                 vocab_tgt, 
                 split_ratio = [0.8, 0.1, 0.1], # train,valid,test
                 batch_size=16, 
                 max_padding=128, 
                 is_distributed=False,
                 device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.src_pipeline = tokenize_src
        self.tgt_pipeline = tokenize_tgt
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_padding = max_padding
        self.is_distributed = is_distributed
        self.device = device
        self.split_ratio = split_ratio
        self.src_pad_idx = vocab_src.get_stoi()["<blank>"]
        self.tgt_pad_idx = vocab_tgt.get_stoi()["<blank>"]

        self._init_loader()

    def _collate_fn(self, batch):
        bs_id = torch.tensor([0], device=self.device) # <s> token id
        eos_id = torch.tensor([1], device=self.device) # </s> token id
        src_ids, tgt_ids = [], []

        for (_src, _tgt) in batch:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_src(self.src_pipeline(_src)),
                        dtype = torch.int64,
                        device = self.device
                    ),
                    eos_id
                ],
                0
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_tgt(self.tgt_pipeline(_tgt)),
                        dtype = torch.int64,
                        device = self.device
                    ),
                    eos_id,
                ],
                0
            )
            src_ids.append(
                pad(
                    processed_src,
                    (0, self.max_padding - len(processed_src)),
                    value = self.src_pad_idx
                )
            )
            tgt_ids.append(
                pad(
                    processed_tgt,
                    (0, self.max_padding - len(processed_tgt)),
                    value = self.tgt_pad_idx
                )
            )

        return (torch.stack(src_ids), torch.stack(tgt_ids))
    
    def _init_loader(self):
        total_size = len(self.dataset)
        split_sizes = [
            int(total_size * self.split_ratio[0]), 
            int(total_size * self.split_ratio[1]), 
            int(total_size * self.split_ratio[2]) # 可能出现不能整除情况，但影响不大
        ]
        train_iter_map, valid_iter_map, test_iter_map = random_split(
            self.dataset, 
            split_sizes,
            generator=torch.Generator().manual_seed(42)  # 固定随机种子
        )

        train_sampler = (
            DistributedSampler(train_iter_map) if self.is_distributed else None
        )
        valid_sampler = (
            DistributedSampler(valid_iter_map) if self.is_distributed else None
        )
        test_sampler = (
            DistributedSampler(test_iter_map) if self.is_distributed else None
        )

        train_dataloader = DataLoader(
            train_iter_map,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=self._collate_fn,
        )
        valid_dataloader = DataLoader(
            valid_iter_map,
            batch_size=self.batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=self._collate_fn,
        )
        test_dataloader = DataLoader(
            test_iter_map,
            batch_size=self.batch_size,
            shuffle=(test_sampler is None),
            sampler=test_sampler,
            collate_fn=self._collate_fn
        )
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        #return copy.deepcopy(train_dataloader), valid_dataloader, test_dataloader
    def get_loader(self):
        #此处有bug
        return copy.deepcopy(self.train_dataloader), self.valid_dataloader, self.test_dataloader
def load_tokenizers():
    try:
        spacy_zh = spacy.load('zh_core_web_sm')
    except Exception:
        os.system('python -m spacy download zh_core_web_sm')
        spacy_zh = spacy.load('zh_core_web_sm')

    try:
        spacy_en = spacy.load('en_core_web_sm')
    except Exception:
        os.system('python -m spacy download en_core_web_sm')
        spacy_en = spacy.load('en_core_web_sm')

    return spacy_en, spacy_zh

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenize(from_to_tuple[index], tokenizer)

def build_vocabulary(spacy_en, spacy_zh):
    def tokenize_zh(text):
        return tokenize(text, spacy_zh)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building Chinese Vocabulary...")
    dataset = TranslateDataset('../data/CCAligned.en-zh_CN.en', '../data/CCAligned.en-zh_CN.zh_CN')
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(dataset, spacy_zh, index=1),
        min_freq = 2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(dataset, spacy_en, index=0),
        min_freq = 2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]
    )

    vocab_src.set_default_index(vocab_src['<unk>'])
    vocab_tgt.set_default_index(vocab_tgt['<unk>'])

    return vocab_src, vocab_tgt

def load_vocab(spacy_en, spacy_zh):
    if not os.path.exists('vocab.pt'):
        vocab_src, vocab_tgt = build_vocabulary(spacy_en, spacy_zh)
        torch.save((vocab_src, vocab_tgt), 'vocab.pt')
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")

    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipline,
    vocab_src,
    vocab_tgt,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device) # <s> token id
    eos_id = torch.tensor([1], device=device) # </s> token id
    src_ids, tgt_ids = [], []

    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocab_src(src_pipeline(_src)),
                    dtype = torch.int64,
                    device = device
                ),
                eos_id
            ],
            0
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocab_tgt(tgt_pipline(_tgt)),
                    dtype = torch.int64,
                    device = device
                ),
                eos_id,
            ],
            0
        )
        src_ids.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value = pad_id
            )
        )
        tgt_ids.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value = pad_id
            )
        )

    return (torch.stack(src_ids), torch.stack(tgt_ids))

# def main():
spacy_en, spacy_zh = load_tokenizers()

vocab_src, vocab_tgt = load_vocab(spacy_en, spacy_zh)

def tokenize_en(text):
    return tokenize(text, spacy_en)

def tokenize_zh(text):
    return tokenize(text, spacy_zh)

dataset = TranslateDataset('../data/CCAligned.en-zh_CN.en', '../data/CCAligned.en-zh_CN.zh_CN')
translate_loader = TranslateDataloader(dataset, tokenize_en, tokenize_zh, vocab_src, vocab_tgt)
t, v, t = translate_loader.get_loader()

from models.model.transformer import Transformer
transformer = Transformer(src_pad_idx = vocab_src.get_stoi()["<blank>"],
                            tgt_pad_idx = vocab_tgt.get_stoi()["<blank>"], 
                            tgt_sos_idx = 0, # output side start of sentence index
                            src_vocab_size = len(vocab_src), 
                            tgt_vocab_size = len(vocab_tgt),
                            d_model = 256, 
                            n_head = 8, 
                            max_len = 500, 
                            ffn_hidden = 512, # feed forward network hidder layer dimensions
                            n_layers = 6,   # encoder and decoder layer count
                            drop_prob = 0,
                            device = "cpu")
for i, batch in enumerate(t):
    src = batch[0]
    trg = batch[1]
    print(src.shape)

    output = transformer(src, trg[:, :-1])
