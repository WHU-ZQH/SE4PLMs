# Self-Evolution Learning for Discriminative Language Model Pretraining

This repository contains the code for our paper accepted by [Findings of ACL2023](https://aclanthology.org/2023.findings-acl.254/).

## Requirements and Installation

- PyTorch version >= 1.10.0
- Python version >= 3.8
- For training, you'll also need an NVIDIA GPU and NCCL.
- To install **fairseq** and develop locally:

``` bash
git clone https://github.com/facebookresearch/fairseq.git
mv fairseq fairseq-setup
cd fairseq-setup
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

Note that the different version of fairseq would cause some issues, and we recommend to use this [version](https://github.com/facebookresearch/fairseq/tree/2fd9d8a972794ba919174baf0d1828a5a4c626f3) for stable training.

# Getting Started
Here, we introduce how to perform the "***self-questioning stage***" and "***self-evolution training stage***" processes, respectively.

# self-questioning stage
To perform this process, you should first prepare the training environment by the following commands:

``` 
# removing the original scripts
rm -r fairseq-setup/fairseq
rm -r fairseq-setup/fairseq_cli/train.py

# using our self-questioning scripts
cp -r fairseq-self_questioning fairseq-setup/
mv fairseq-setup/fairseq-self_questioning fairseq-setup/fairseq
cp -r fairseq_cli/train-self_questioning.py fairseq-setup/fairseq_cli/train.py
```

Then, you can obtain the hard-to-learn tokens by the following commands:

``` 
DATA_DIR=data-path
MODEL_DIR=model-path

fairseq-train $DATA_DIR \
    --train-subset train \
    --valid-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --memory-efficient-fp16 \
    --fp16-init-scale 8 \
    --arch roberta_base \
    --task masked_lm \
    --sample-break-mode "complete" \
    --batch-size 16 \
    --tokens-per-sample 512 \
    --restore-file $MODEL_DIR \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 2 \
    --no-epoch-checkpoints \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 30000 \
    --dropout 0.1 \
    --max-positions 512 \
    --attention-dropout 0.1 \
    --update-freq 6 \
    --ddp-backend=legacy_ddp \
    --total-num-update 250000 \
    --max-epoch 1 \
    --max-update 250000 \
    --no-save \
    --disable-validation \
    --log-format json --log-interval 100 2>&1 | tee $DATA_DIR/train.log
```

After obtaining the tokens, you should merge them into a single file by running the following commands with python:
```
import os
import pickle
from tqdm import tqdm

base_path="data-path/masked-tokens"
merge_path="data-path/token_mask/token_masks_merge.pkl"
merge_tokens={}

for files in tqdm(os.listdir(base_path)):
    f_read = open("{}/{}/tokens_mask-last.pkl".format(base_path, files), 'rb')
    token_mask = pickle.load(f_read)
    f_read.close()
    merge_tokens.update(token_mask)
    
f_save = open(merge_path, 'wb')
pickle.dump(merge_tokens, f_save)
f_save.close()
print("Merge finished!")
```
# self-evolutioning training stage
Sequentially, you can perform the self-evolutioning training process. Similar to the above processes, you need to prepare the training environment by:
``` 
# removing the self-questioning scripts
rm -r fairseq-setup/fairseq
rm -r fairseq-setup/fairseq_cli/train.py

# using our self-evolution scripts
cp -r fairseq-self_evolution fairseq-setup/
mv fairseq-setup/fairseq-self_evolution fairseq-setup/fairseq
cp -r fairseq_cli/train-self_evolution.py fairseq-setup/fairseq_cli/train.py
```
Lastly, you can run our self-evolution training by the following commands:

``` 
DATA_DIR=data-path
MODEL_DIR=model-path
SAVE_DIR=save-path

mkdir -p  $SAVE_DIR

fairseq-train $DATA_DIR \
    --train-subset train \
    --valid-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --memory-efficient-fp16 \
    --fp16-init-scale 8 \
    --arch roberta_base \
    --task masked_lm \
    --sample-break-mode "complete" \
    --batch-size 8 \
    --tokens-per-sample 512 \
    --restore-file $MODEL_DIR \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-interval 1 --save-interval-updates 2000 --keep-interval-updates 1 \
    --no-epoch-checkpoints \
    --optimizer adam --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-06 \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 1e-5 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 6000 \
    --dropout 0.1 \
    --max-positions 512 \
    --attention-dropout 0.1 \
    --update-freq 1 \
    --ddp-backend=legacy_ddp \
    --max-epoch 2 \
    --total-num-update 60000 \
    --max-update 60000 \
    --save-dir $SAVE_DIR \
    --log-format json --log-interval 100 2>&1 | tee $SAVE_DIR/train.log

```

# Evaluation
You can evaluate the improved models by using the original [fine-tuning scripts](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta), or any way you like.



## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@inproceedings{zhong-etal-2023-self,
    title = "Self-Evolution Learning for Discriminative Language Model Pretraining",
    author = "Zhong, Qihuang  and Ding, Liang  and Liu, Juhua  and Du, Bo  and Tao, Dacheng",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
}
```



