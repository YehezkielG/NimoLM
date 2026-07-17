import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """
    RMSNorm with gated activation.
    Computes: output = RMSNorm(x) * activation(z)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        activation: str = "swish",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

        if activation == "swish" or activation == "silu":
            self.act_fn = F.silu
        elif activation == "sigmoid":
            self.act_fn = torch.sigmoid
        else:
            self.act_fn = F.silu

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x / rms
        x_normed = x_normed.to(input_dtype)

        return x_normed * self.weight * self.act_fn(z)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"
