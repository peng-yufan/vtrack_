# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : attention.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import torch.nn as nn
from einops import rearrange

from lib.utils.image import *


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, padding_mask=None, **kwargs):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C//head)

		attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, head, N, N)

		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class ClsMixAttention(nn.Module):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, t_h, t_w, s_h, s_w, online_size=1, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_cls, q_t, q_s = torch.split(q, [1, t_h * t_w * (1 + online_size), s_h * s_w], dim=2)
		k_cls, k_t, k_s = torch.split(k, [1, t_h * t_w * (1 + online_size), s_h * s_w], dim=2)
		v_cls, v_t, v_s = torch.split(v, [1, t_h * t_w * (1 + online_size), s_h * s_w], dim=2)
		# cls token attention
		attn = (q_cls @ k.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_cls = rearrange(attn @ v, 'b h t d -> b t (h d)')

		# template attention
		# qtm, qti, qtd = torch.split(q_t, [t_h * t_w, t_h * t_w, t_h * t_w], dim=2)
		# ktm, kti, ktd = torch.split(k_t, [t_h * t_w, t_h * t_w, t_h * t_w], dim=2)
		# vtm, vti, vtd = torch.split(v_t, [t_h * t_w, t_h * t_w, t_h * t_w], dim=2)
		attn = (q_t @ k_t.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
		# attn_m = (qtm @ ktm.transpose(-2, -1)) * self.scale
		# attn_i = (qti @ kti.transpose(-2, -1)) * self.scale
		# attn_d = (qtd @ ktd.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		# attn_m = attn_m.softmax(dim=-1)
		# attn_i = attn_i.softmax(dim=-1)
		# attn_d = attn_d.softmax(dim=-1)
		attn = self.attn_drop(attn)
		# attn_m = self.attn_drop(attn_m)
		# attn_i = self.attn_drop(attn_i)
		# attn_d = self.attn_drop(attn_d)
		x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')
		# x_tm = rearrange(attn_m @ vtm, 'b h t d -> b t (h d)')
		# x_ti = rearrange(attn_i @ vti, 'b h t d -> b t (h d)')
		# x_td = rearrange(attn_d @ vtd, 'b h t d -> b t (h d)')
		# x_t = torch.cat([x_tm, x_ti, x_td], dim=-2)

		# search region attention
		attn = (q_s @ k.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v, 'b h t d -> b t (h d)')
		# batch, channel = 0, 5
		# attn_m = attn_m = torch.sum(attn[batch, channel, :, 1:1 + 64].view(20, 20, -1).detach().to('cpu'), dim=-1)
		# attn_i = torch.sum(attn[batch, channel, :, 1+64:1 + 64*2].view(20, 20, -1).detach().to('cpu'), dim=-1)
		# attn_d = torch.sum(attn[batch, channel, :, 1+64*2:1 + 64*3].view(20, 20, -1).detach().to('cpu'), dim=-1)
		# plt.figure(figsize=(20, 20))
		# plt.imshow(attn_m)
		# plt.figure(figsize=(20, 20))
		# plt.imshow(attn_i)
		# plt.figure(figsize=(20, 20))
		# plt.imshow(attn_d)

		# attn_m = (torch.sum(torch.mean(attn[batch, :, :, 1:1 + 64], dim=0), dim=-1)).view(20, 20).detach().to('cpu')
		# attn_i = (torch.sum(torch.mean(attn[batch, :, :, 1+64:1 + 2*64], dim=0), dim=-1)).view(20, 20).detach().to('cpu')
		# attn_d = (torch.sum(torch.mean(attn[batch, :, :, 1+2*64:1 + 3*64], dim=0), dim=-1)).view(20, 20).detach().to('cpu')
		# plt.figure(figsize=(20, 20))
		# plt.imshow(attn_m)
		# plt.figure(figsize=(20, 20))
		# plt.imshow(attn_i)
		# plt.figure(figsize=(20, 20))
		# plt.imshow(attn_d)

		x = torch.cat([x_cls, x_t, x_s], dim=1)
		# plt.figure(figsize=(20, 20))
		# plt.imshow(torch.mean(torch.abs(x_s[0].view(20, 20, -1)), dim=-1).detach().to('cpu'))

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class MixAttention(nn.Module):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, t_h, t_w, s_h, s_w, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
		k_t, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
		v_t, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

		# template attention
		attn = (q_t @ k_t.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v, 'b h t d -> b t (h d)')

		x = torch.cat([x_t, x_s], dim=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class NottAttention(nn.Module):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, t_h, t_w, s_h, s_w, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
		k_t, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
		v_t, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

		# template attention
		attn = (q_t @ k_s.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v_s, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v, 'b h t d -> b t (h d)')

		x = torch.cat([x_t, x_s], dim=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class NossAttention(nn.Module):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, t_h, t_w, s_h, s_w, padding_mask=None):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
		k_t, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
		v_t, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

		# template attention
		attn = (q_t @ k.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k_t.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
		if padding_mask is not None:
			assert padding_mask.size()[0] == B
			assert padding_mask.size()[1] == N
			attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

		x = torch.cat([x_t, x_s], dim=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class CrossAttention(nn.Module):
	def __init__(self,
	             dim,
	             num_heads,
	             qkv_bias=False,
	             attn_drop=0.,
	             proj_drop=0.,
	             ):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, t_h, t_w, s_h, s_w):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]  # (B, head, N, C)

		q_t, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
		k_t, k_s = torch.split(k, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], dim=4)
		v_t, v_s = torch.split(v, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], dim=4)

		# template attention
		attn = (q_t @ k_s.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_t = rearrange(attn @ v_s, 'b h t d -> b t (h d)')

		# search region attention
		attn = (q_s @ k_t.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		x_s = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

		x = torch.cat([x_t, x_s], dim=1)

		x = self.proj(x)
		x = self.proj_drop(x)
		return x
