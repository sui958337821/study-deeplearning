import spacy
import os
import torch
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
        self.zh_sentences = self._load_file(zh_file)
        self.en_sentences = self._load_file(en_file)

        assert len(self.zh_sentences) == len(self.en_sentences), "文件函数不匹配"

    def _load_file(self, file_path):
        sentences = []
        with open(file_path, "r", encoding = 'utf-8') as f:
            for line in f:
                sentences.append(line.strip())

        return sentences

    def __getitem__(self, idx: int) -> tuple[str]:
        return (self.en_sentences[idx], self.zh_sentences[idx])

    def __len__(self):
        return len(self.zh_sentences)


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

# %% id="ka2Ce_WIokC_" tags=[]
def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_en,
    spacy_zh,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_zh(text):
        return tokenize(text, spacy_zh)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_en,
            tokenize_zh,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    dataset = TranslateDataset('../data/CCAligned.en-zh_CN.en', '../data/CCAligned.en-zh_CN.zh_CN')
    total_size = len(dataset)
    split_sizes = [
        int(total_size * 0.8), 
        int(total_size * 0.1), 
        total_size - int(total_size * 0.8) - int(total_size * 0.1)
    ]
    train_iter, valid_iter, test_iter = random_split(
        dataset, 
        split_sizes,
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader



def main():
    spacy_en, spacy_zh = load_tokenizers()
    # print(tokenize('this is a sentence', spacy_en))
    # print(tokenize("这是一个句子", spacy_zh))
    
    print("=====__func__:load_tokenizers======")
    vocab_src, vocab_tgt = load_vocab(spacy_en, spacy_zh)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def tokenize_zh(text):
        return tokenize(text, spacy_zh)
    # print(vocab_tgt(tokenize("this is a a", spacy_en)))
    batch = [("this is a book", "这是一本书"), ("I love you", "我爱你")]

    co_src, co_tgt = collate_batch(batch, tokenize_en, tokenize_zh, vocab_src, vocab_tgt, device="cpu")
    print(co_src)
    co_new_src, co_new_tgt = collate_batch_new(batch, tokenize_en, tokenize_zh, vocab_src, vocab_tgt, device="cpu")
    print(co_new_src)

    assert co_src == co_new_src
    return

    train_dataloader, valid_dataloader = create_dataloaders(
        "cpu",
        vocab_src,
        vocab_tgt,
        spacy_en,
        spacy_zh,
        is_distributed = False
    )
    
    single_batch = next(iter(train_dataloader))  
    print("Batch 数据结构类型:", type(single_batch))
    print("输入样例:", single_batch[0][0])  # 第一个样本的输入
    print("标签样例:", single_batch[1][0])  # 第一个样本的标签
if __name__ == "__main__":
    main()