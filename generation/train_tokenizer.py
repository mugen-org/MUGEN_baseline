# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

import _init_path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
import json

vocab_sizes = [1024]

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

for vocab_size in vocab_sizes:
    trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"], vocab_size=vocab_size)

    tokenizer.pre_tokenizer = Whitespace()

    manual_text_path = 'datasets/coinrun/coinrun_dataset_jsons/release/train_manual_text.txt'
    all_sentences = []
    with open(manual_text_path, 'w') as txt_file:
        for d in json.load(open('datasets/coinrun/coinrun_dataset_jsons/release/train.json'))['data']:
            all_sentences.extend([anno['text'] for anno in d['annotations'][1:2]])
        txt_file.writelines([s+'\n' for s in all_sentences])

    files = [manual_text_path]

    tokenizer.train(files, trainer)

    tokenizer.save(f"datasets/coinrun/tokenizers/tokenizer-coinrun_{vocab_size}.json")

    tokenizer = Tokenizer.from_file(f"datasets/coinrun/tokenizers/tokenizer-coinrun_{vocab_size}.json")

    output = tokenizer.encode("Mugen runs to left to right and collects a coin, yay!".lower())

    print(output.tokens)
    print(output.ids)

    print(tokenizer.encode("[PAD]").tokens)
    print(tokenizer.encode("[PAD]").ids)

    with open(files[0], "r") as f:
        data = f.readlines()

    tokens = [tokenizer.encode(text.strip().lower()).ids for text in data]
    lengths = [len(t) for t in tokens]
    temp = [t for l, t in zip(lengths, data) if l==0]
    print(len(temp))
    print(temp)
    print(f"min length: {np.min(lengths)}")
    print(f"max length: {np.max(lengths)}")
    print(f"mean length: {np.mean(lengths)}")

    counts = {}
    for token in tokens:
        for t in token:
            if t not in counts:
                counts[t] = 0
            counts[t] += 1

    print(f"length of unique tokens used: {len(counts)}")
    for i in range(6):
        temp = [v for k, v in counts.items() if v <= i]
        print(f"used fre <= {i}: {len(temp)}")
        print(f"max key: {np.max(list(counts.keys()))}")
        print(f"max value: {np.max(list(counts.values()))}")
