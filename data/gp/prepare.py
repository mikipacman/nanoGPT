# Based on https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

import argparse
import glob
import os

import numpy as np
import tiktoken
from datasets import Dataset  # huggingface datasets
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_text_files', type=str, help='path to the gp text files')
    parser.add_argument("--num_proc", type=int, help="number of workers in .map() call", default=8)
    args = parser.parse_args()

    # create dataset with file paths
    files = glob.glob(os.path.join(args.path_to_text_files, '*.txt'))
    dataset = Dataset.from_dict({"file": files})


    def read_file(example):
        with open(example['file'], 'r') as f:
            text = f.read()
        return {'text': text}


    # read files
    dataset = dataset.map(
        read_file,
        remove_columns=['file'],
        desc="reading the text files",
        num_proc=args.num_proc,
    )

    # split
    split_dataset = dataset.train_test_split(test_size=0.01, seed=2137, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    split_dataset.pop('train')

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 53337
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 539
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")


    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out


    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=args.num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 64

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
