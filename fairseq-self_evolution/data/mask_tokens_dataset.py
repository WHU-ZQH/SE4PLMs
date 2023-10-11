# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils
import pickle

from . import BaseWrapperDataset, LRUCacheDataset
logger = logging.getLogger(__name__)

class MaskTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
        mask_multiple_length : repeat each mask index multiple times. Default
            value is 1.
        mask_stdev : standard deviation of masks distribution in case of
            multiple masking. Default value is 0.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_ref=True)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_tag=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
        mask_multiple_length: int = 1,
        mask_stdev: float = 0.0,
        token_mask=None,
        return_ref=False,
        return_tag=False,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert mask_multiple_length >= 1
        assert mask_stdev >= 0.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        self.mask_multiple_length = mask_multiple_length
        self.mask_stdev = mask_stdev
        self.return_ref=return_ref
        self.return_tag=return_tag
        self.hard_tag=2
        self.ori_tag=1

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[: self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

        if token_mask is not None:
            print("loaded token_mask, sucessfully!")
            self.token_mask=token_mask.value
            self.lens=[1,3,5,7]
            self.p = 0.2
            self.len_distrib = [self.p * (1-self.p)**(i - 1) for i in self.lens] if self.p >= 0 else None
            self.len_distrib = [x / (sum(self.len_distrib)) for x in self.len_distrib]
        else:
            self.token_mask=None

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            if self.return_ref:
                return item

            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            mask = np.full(sz, False)
            tag=np.full(sz, 0)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz / float(self.mask_multiple_length)
                + np.random.rand()
            )
            
            # decide elements to mask
            if self.token_mask is not None:
                try:
                    mask_ids=self.token_mask[index]
                    mask[mask_ids]=True

                    # for ids in mask_ids:
                    #     span_len = np.random.choice(self.lens, p=self.len_distrib)-1
                    #     if ids+int(span_len/2)+1 <sz:
                    #         mask[ids-int(span_len/2):ids+int(span_len/2)+1]=True
                    #     else:
                    #         mask[ids-int(span_len/2):sz]=True
                    
                    
                    assert (mask_ids[-1]<=item.shape[0]), "saved token_mask does not match the size in ids-{}.".format(index)

                    tag[mask_ids] = self.hard_tag
                    if self.return_tag:
                        return torch.from_numpy(tag)

                    # if mask.sum() <num_mask:
                    #     num_mask = num_mask -mask.sum()
                    # else:
                    if self.return_masked_tokens:
                        if self.mask_whole_words is not None:
                            mask = np.repeat(mask, word_lens)
                        new_item = np.full(len(mask), self.pad_idx)
                        new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                        return torch.from_numpy(new_item)
                    else:
                        new_item = np.copy(item)
                        new_item[mask] = self.mask_idx
                        return torch.from_numpy(new_item)

                except:
                    pass
            
            # multiple masking as described in the vq-wav2vec paper (https://arxiv.org/abs/1910.05453)
            # mask_idc = np.random.choice(sz, num_mask, replace=False)
            try:
                mask_idc = np.random.choice(np.arange(sz)[~mask], num_mask, replace=False)
            except:
                if self.return_masked_tokens:
                    if self.mask_whole_words is not None:
                        mask = np.repeat(mask, word_lens)
                    new_item = np.full(len(mask), self.pad_idx)
                    new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                    return torch.from_numpy(new_item)
                else:
                    new_item = np.copy(item)
                    new_item[mask] = self.mask_idx
                    return torch.from_numpy(new_item)

            if self.mask_stdev > 0.0:
                lengths = np.random.normal(
                    self.mask_multiple_length, self.mask_stdev, size=num_mask
                )
                lengths = [max(0, int(round(x))) for x in lengths]
                mask_idc = np.asarray(
                    [
                        mask_idc[j] + offset
                        for j in range(len(mask_idc))
                        for offset in range(lengths[j])
                    ],
                    dtype=np.int64,
                )
            else:
                mask_idc = np.concatenate(
                    [mask_idc + i for i in range(self.mask_multiple_length)]
                )
            mask_idc = mask_idc[mask_idc < len(mask)]
            try:
                mask[mask_idc] = True
                tag[mask_idc] = self.ori_tag
            except:  # something wrong
                print(
                    "Assigning mask indexes {} to mask {} failed!".format(
                        mask_idc, mask
                    )
                )
                raise

            if self.return_tag:
                return torch.from_numpy(tag)

            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                if self.mask_whole_words is not None:
                    mask = np.repeat(mask, word_lens)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )

            return torch.from_numpy(new_item)

