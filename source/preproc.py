import os
os.system('pip install transformers scikit-learn --quiet')

from tqdm import tqdm, trange, tqdm_notebook
import numpy as np
from sklearn.metrics import matthews_corrcoef
import torch
import pandas as pd
#from transformers import WarmupLinearSchedule
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
#from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
from transformers import AdamW
from transformers import (BertConfig,
                          BertForSequenceClassification,
                          BertTokenizer)
from torch.utils.data import (TensorDataset,
                              DataLoader,
                              RandomSampler,
                              SequentialSampler)


def read_data(fname):
    df = pd.read_csv(fname, delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'notes', 'sentence'])
    return df


def pre_process(df):
    sentences = df.sentence.values
    sentences = ["[CLS] " + sent + " [SEP]" for sent in sentences]
    labels = df.label.values
    return sentences, torch.tensor(labels)


def plot_token_dist(tokenized_texts):
    td = pd.DataFrame(get_len(tokenized_texts)).describe().loc['min':]
    print('Token Distribution: ', td)


def define_tokenizer(tokenizer_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
    print('Total vocab size: ', tokenizer.vocab_size)
    print('Pad token ID: ', tokenizer.pad_token_id)
    #print('Unknown token ID: ', tokenizer.unk_token_is)
    return tokenizer


def tokenize_and_ids(sentences, tokenizer):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    return tokenized_texts, input_ids


def pad_sequence_mask(sequence, max_len):
    padded_sequence = torch.zeros(len(sequence), max_len).long()
    for i, seq in enumerate(sequence):
        padded_sequence[i, :len(seq)] = torch.Tensor(seq)
    attention_masks = padded_sequence > 0
    return padded_sequence, attention_masks


def get_len(sequences):
    return [len(seq) for seq in sequences]


def get_max_len(sequences):
    return max(get_len(sequences))


def prepare_dataset(fname, tokenizer, max_len):
    df = read_data(fname)
    sentences, labels = pre_process(df)
    tokenized_texts, input_ids = tokenize_and_ids(sentences, tokenizer)
    if 'train' in fname:
        plot_token_dist(tokenized_texts)
    padded_input_ids, attention_masks = pad_sequence_mask(input_ids, max_len)
    return padded_input_ids, attention_masks, labels
