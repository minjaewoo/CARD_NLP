# Introduction

This repository contains codes for sentence alteration using BERT word prediction.

```
Input Sentence: I want to open a bank account tomorrow

|------------------------- Input sentence------------------------------|Predicted Word|
[MASK]  want      to      open      a       bank    account   tomorrow      'i’ 
I       [MASK]    to      open      a       bank    account   tomorrow      'have’ 
I       want      [MASK]  open      a       bank    account   tomorrow      'to’ 
I       want      to      [MASK]    a       bank    account   tomorrow      'open’
I       want      to      open    [MASK]    bank    account   tomorrow      'my’
I       want      to      open      a       [MASK]  account   tomorrow      'new’
I       want      to      open      a       bank    [MASK]    tomorrow      'account’
I       want      to      open      a       bank    account   [MASK]        'now'

New Sentence: I have to open my new account now
```

# Sample Usage
```
python3 /path/to/BERT_word_prediction.py /path/to/sample_word_prediction.txt
```
