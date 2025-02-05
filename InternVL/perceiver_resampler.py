import os
import torch
from torch import nn
from einops import rearrange, repeat
from torch import einsum

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads  # 512

        self.norm_media = nn.LayerNorm(dim)
        self.norm_learns = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False) # 4096×512
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False) # 4096×1024
        self.to_out = nn.Linear(inner_dim, dim, bias=False) # 512×4096

    def forward(self, x, learns): # x(b, 256, 4096), learns(b, 3, 4096)
        x = self.norm_media(x)
        learns = self.norm_learns(learns)

        b, n, h = *x.shape[:2], self.heads

        q = self.to_q(learns) # q(b, 3, 512)

        # 注意：在PerceiverResampler中，将输入和learns拼接后进行attention计算
        kv_input = torch.cat((x, learns), dim=-2) # kv_input(b, 259, 4096)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1) # (b, 259, 1024)->k, v(b, 259, 512)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v)) # q(b, 8, 3, 64)   k, v(b, 8, 259, 64)

        q = q * self.scale

        # attention计算
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1) # sim, attn(b, 8, 3, 259)

        out = einsum('b h i j, b h j d -> b h i d', attn, v) # out(b, 8, 3, 64)
        out = rearrange(out, 'b h n d -> b n (h d)') # out(b, 3, 512)
        return self.to_out(out) # return(b, 3, 4096)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 4096
        depth=2,
        dim_head=64,
        heads=8,
        num_learns=3,  # 修改为3个learned queries
        ff_mult=4,
    ):
        super().__init__()
        self.learns = nn.Parameter(torch.randn(num_learns, dim)) # 3×4096

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, 256, 4096)
        Returns:
            shape (b, 3, 4096) where 3 is self.num_learns
        """
        b, n, d = x.shape  # (b, 256, 4096)

        # 将learned queries广播到batch size
        learns = repeat(self.learns, "n d -> b n d", b=b)

        # 通过多层PerceiverAttention和FeedForward模块处理输入
        for attn, ff in self.layers:
            # learns = attn(learns, x) + learns
            learns = attn(x, learns) + learns
            learns = ff(learns) + learns

        return self.norm(learns)

# 用于前向传播的FeedForward模块
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)