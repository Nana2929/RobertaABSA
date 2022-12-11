"""
Author: Ching Wen Yang
Date: 2022/11/10
After running /RoBERTaABSA/ASGCN/train.py
(or alternatively /RoBERTaABSA/ASGCN/train_all_layers.sh),
collect the last 2 lines for each layer's logfile
(e.g., /RoBERTaABSA/ASGCN/training_logs/log_0) and list their max test scores
for easier comparisons
"""
import os
parentdir = '/home/P76114511/RoBERTaABSA/ASGCN'
logdir = '/home/P76114511/RoBERTaABSA/ASGCN/training_logs'
logfile = 'No-finetuned-RoBERTa-ASGCN-Laptop.log'
with open(os.path.join(parentdir, logfile), 'w') as fw:
    for filename in sorted(os.listdir(logdir), key = lambda x: int(x.split('-')[1])):
        layer = filename.split('-')[-1]
        print('- layer:', layer)
        if layer == '0':
            fw.write(f'===== {layer} (Embedding) ====')
        else:
            fw.write(f'===== {layer}====')
        fw.write('\n')
        with open(os.path.join(logdir, filename), 'r') as f:
            lines = f.readlines()
            scorelogs = lines[-2:]
            scores = ''.join(scorelogs)
            print(scores)
            fw.write(scores)

