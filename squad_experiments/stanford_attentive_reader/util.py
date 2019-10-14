import torch
import torch.nn.functional as F

def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """
    Takes the softmax of `logits` over the given dimension, and sets entries
    to 0 wherever `mask` is 0

    @param logits (Tensor): input to the softmax function
    @param mask (Tensor): tensor with shape same as `logits`
    @param dim (int): The dimension to take the softmax over
    @param log_softmax (bool): a boolean flag indicating whether the
                               log-softmax function should be applied

    @returns probs (Tensor)
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_func = F.log_softmax if log_softmax else F.softmax
    probs = softmax_func(masked_logits, dim=dim)

    return probs
