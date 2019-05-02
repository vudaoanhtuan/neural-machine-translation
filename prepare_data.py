import torch
from torchtext.data import Field, Dataset, Iterator, BucketIterator, ReversibleField
from torchtext.datasets import TranslationDataset
from preprocess import preprocess, tokenize

def tokenize_word(text):
    text = preprocess(text)
    return tokenize(text)

SRC = Field(
    tokenize=tokenize_word,
    lower=True,
    batch_first=True
)

TRG = Field(
    tokenize=tokenize_word,
    init_token='<sos>',
    eos_token='<eos>',
    lower=True,
    batch_first=True
)

fields = [('src', SRC), ('trg', TRG)]

ds = TranslationDataset('lang.', ('en', 'de'), fields)

train_ds, test_ds = ds.split(0.9)

SRC.build_vocab(ds)
TRG.build_vocab(ds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

train_iter = BucketIterator(
    train_ds,
    batch_size=batch_size,
    device=device
)

test_iter = BucketIterator(
    test_ds,
    batch_size=batch_size,
    device=device
)


    