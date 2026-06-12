import torch


@torch.no_grad()
def run(x):
    # Row-wise softmax over the last dimension. Computed in fp32 for numerical
    # stability, then cast back to the input dtype.
    return torch.softmax(x.to(torch.float32), dim=-1).to(x.dtype)
