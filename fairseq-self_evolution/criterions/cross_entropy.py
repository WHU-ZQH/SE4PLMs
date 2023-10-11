# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.models.roberta import RobertaModel
import torch

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.t = 1.5
        self.tag=2
        self.eps = 0.1
        self.vanilla_label_smoothing=False

    def forward(self, model, sample, reduce=True, ref_model=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        if ref_model is not None:
            with torch.no_grad():
                model.eval()
                # net_output_ref = ref_model(**sample["net_input_ref"], return_all_hiddens=False)
                net_output_ref = model(**sample["net_input_ref"], return_all_hiddens=False)

            model.train()
            loss, loss_nll= self.compute_loss(model, net_output, sample, reduce=reduce, net_output_ref=net_output_ref)
        else:
            loss, loss_nll= self.compute_loss(model, net_output, sample, reduce=reduce, net_output_ref=None)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "loss_nll": loss_nll.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, net_output_ref=None):
        lprobs_pred= model.get_normalized_probs(net_output, log_probs=False)
        lprobs_pred = lprobs_pred.view(-1, lprobs_pred.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        mask_idx=(target-self.padding_idx).nonzero().squeeze()
        lprobs_pred=lprobs_pred.index_select(0,mask_idx)
        target_mask=target.index_select(0,mask_idx).unsqueeze(-1)
        
        if self.vanilla_label_smoothing:
            lprobs_ref = torch.full_like(lprobs_pred, 1/lprobs_pred.shape[1])

            with torch.no_grad():
                loss_ref=-torch.log(lprobs_ref).gather(dim=-1, index=target_mask)
                alpha=torch.full_like(loss_ref.squeeze(),self.eps)
                lprobs_tgt=torch.full_like(lprobs_ref, 0.0).scatter(1, target_mask, torch.full_like(target_mask, 1, dtype=torch.float32))
                lprobs_tgt=(1-alpha.unsqueeze(-1))*lprobs_tgt+alpha.unsqueeze(-1)*lprobs_ref
                loss_nll=-torch.log(lprobs_pred).gather(dim=-1, index=target_mask).squeeze()
            loss=-(lprobs_tgt*torch.log(lprobs_pred)).sum(dim=-1)

        elif net_output_ref is not None:
            lprobs_ref = model.get_normalized_probs(net_output_ref, log_probs=False)
            lprobs_ref = lprobs_ref.view(-1, lprobs_ref.size(-1))
            lprobs_ref = lprobs_ref.index_select(0,mask_idx)
            tags=sample["tag"].view(-1).index_select(0,mask_idx)

            with torch.no_grad():
                loss_ref=-torch.log(lprobs_ref).gather(dim=-1, index=target_mask)
                alpha=torch.full_like(loss_ref.squeeze(),0.1)
                lprobs_tgt=torch.full_like(lprobs_ref, 0.0).scatter(1, target_mask, torch.full_like(target_mask, 1, dtype=torch.float32))
                lprobs_tgt=(1-alpha.unsqueeze(-1))*lprobs_tgt+alpha.unsqueeze(-1)*lprobs_ref
                loss_nll=-torch.log(lprobs_pred).gather(dim=-1, index=target_mask).squeeze()
            loss=-(lprobs_tgt*torch.log(lprobs_pred)).sum(dim=-1)
        else:
            loss=-torch.log(lprobs_pred).gather(dim=-1, index=target_mask)
            loss_nll=loss

        return loss.sum(), loss_nll.sum()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_nll_sum = sum(log.get("loss_nll", 0) for log in logging_outputs)
        # loss_sl_sum = sum(log.get("loss_sl", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_nll", loss_nll_sum / sample_size / math.log(2), sample_size, round=3
        )
        # metrics.log_scalar(
        #     "loss_sl", loss_sl_sum / sample_size / math.log(2), sample_size, round=3
        # )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
