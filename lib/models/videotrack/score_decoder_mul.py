"""
SPM: Score Prediction Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lib.models.mixformer_cvt.head import MLP
from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from timm.models.layers import trunc_normal_

class MulScoreDecoder(nn.Module):
    def __init__(self, num_heads=12, hidden_dim=768, nlayer_head=3, pool_size=4, num_search=10):
        super().__init__()
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.num_search = num_search
        self.time_mixlinear = nn.ModuleList(nn.Linear(num_search, num_search, bias=True) for _ in range(2))
        self.score_head = MLP(hidden_dim, hidden_dim, 1, nlayer_head)
        self.scale = hidden_dim ** -0.5
        self.search_prroipool = PrRoIPool2D(pool_size, pool_size, spatial_scale=1.0)
        self.proj_q = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_k = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_v = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.proj = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))

        self.score_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        trunc_normal_(self.score_token, std=.02)

    def forward(self, search_feats, template_feat, search_boxes):
        """
        :param search_box: with normalized coords. (x0, y0, x1, y1)
        :return:
        """
        N, b, c, h, w = search_feats.shape
        search_boxes = search_boxes.clone() * w
        # bb_pool = box_cxcywh_to_xyxy(search_box.view(-1, 4))
        bb_pools = search_boxes.reshape(-1, 4) # BN, xywh
        # Add batch_index to rois
        batch_size = bb_pools.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb_pools.device)
        target_roi = torch.cat((batch_index, bb_pools), dim=1)

        # decoder1: query for search_box feat
        # decoder2: query for template feat
        x = self.score_token.expand(self.num_search, b, -1, -1)
        x = self.norm1(x)
        search_feats = search_feats.view(-1, c, h, w) # Nb, c, h, w
        search_box_feat = rearrange(self.search_prroipool(search_feats.float(), target_roi), 'b c h w -> b (h w) c')
        search_box_feat = search_box_feat.view(N, b, -1, c)  # N, b, hw ,c
        template_feat = rearrange(template_feat, 'n b c h w -> n b (h w) c')
        kv_memory = []
        for i in range(search_box_feat.shape[0]):
            kv_memory.append([search_box_feat[i], template_feat[i]])
        for i in range(2):
            x_m = []
            for j in range(x.shape[0]):
                q = rearrange(self.proj_q[i](x[j]), 'b t (n d) -> b n t d', n=self.num_heads)
                k = rearrange(self.proj_k[i](kv_memory[j][i]), 'b t (n d) -> b n t d', n=self.num_heads)
                v = rearrange(self.proj_v[i](kv_memory[j][i]), 'b t (n d) -> b n t d', n=self.num_heads)

                attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
                attn = F.softmax(attn_score, dim=-1)
                x_ = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
                x_ = rearrange(x_, 'b h t d -> b t (h d)')   # (b, 1, c)
                x_ = self.proj[i](x_)
                x_ = self.norm2[i](x_)
                x_m.append(x_)
            x = torch.cat(x_m, dim=1) # B, N, C
            x = x.permute([0, 2, 1]) # B, C, N
            # x = x.reshape(-1, c)
            x = self.time_mixlinear[i](x)
            # x = x.view(b, c, N)
            x = x.permute([2, 0, 1]) # N, B, C
            x = x.view(N, b, -1, c)
        out_scores = self.score_head(x)  # (b, 1, 1)

        return out_scores # n, b, 1, 1
