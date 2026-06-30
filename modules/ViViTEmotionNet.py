import torch
from torch import nn, einsum
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce
from modules.weight_init import constant_init_, kaiming_init_, trunc_normal_
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import math


class DropPath(nn.Module):

	def __init__(self, dropout_p=None):
		super(DropPath, self).__init__()
		self.dropout_p = dropout_p

	def forward(self, x):
		return self.drop_path(x, self.dropout_p, self.training)

	def drop_path(self, x, dropout_p=0., training=False):
		if dropout_p == 0. or not training:
			return x
		keep_prob = 1 - dropout_p
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape).type_as(x)
		random_tensor.floor_()  # binarize
		output = x.div(keep_prob) * random_tensor
		return output


class PatchEmbed(nn.Module):
	"""EEG signals to Tubelet Embedding.

    Args:
        img_size (int | list): Size of input image.
        tube_size (int | list): Size of temporal field of one 3D patch.
        in_channels (int): Channel num of input features. Defaults to 2.
        embed_dims (int): Dimensions of embedding. Defaults to 256.
        conv_type (str): Type for convolution layer. Defaults to 'Conv3d'.
    """

	def __init__(self,
				 img_size=[4, 12, 64, 64],
				 tube_size=[1, 1, 16, 16],
				 embed_dims=128,
				 conv_type='Conv3d'):
		super().__init__()
		self.img_size = img_size
		self.tube_size = tube_size
		self.tubelet_dims = self.tube_size[0] * self.tube_size[1] * self.tube_size[2] * self.tube_size[3]
		# Use conv layer to embed
		if conv_type == 'Conv3d':
			self.projection = nn.Conv3d(
				in_channels=self.tube_size[1],
				out_channels=embed_dims,
				kernel_size=(self.tube_size[0], self.tube_size[2], self.tube_size[3]),
				stride=(self.tube_size[0], self.tube_size[2], self.tube_size[3]))
		elif conv_type == 'Conv2d':
			self.projection = nn.Conv2d(
				in_channels=self.tube_size[1],
				out_channels=embed_dims,
				kernel_size=(self.tube_size[2], self.tube_size[3]),
				stride=(self.tube_size[2], self.tube_size[3]))
		elif conv_type == 'Linear':
			self.projection = nn.Linear(self.tubelet_dims, embed_dims)
		elif conv_type == 'Conv_Stem':
			self.projection = nn.Sequential(
				nn.Conv2d(in_channels=tube_size[1], out_channels=embed_dims // 8, kernel_size=(2, 2), stride=(2, 2),
						  padding=0),  # 64 -> 32
				nn.BatchNorm2d(embed_dims // 8),
				nn.ReLU(),
				nn.Conv2d(in_channels=embed_dims // 8, out_channels=embed_dims // 2, kernel_size=(2, 2), stride=(2, 2),
						  padding=0),  # 32 -> 16
				nn.BatchNorm2d(embed_dims // 2),
				nn.ReLU(),
				nn.Conv2d(in_channels=embed_dims // 2, out_channels=embed_dims * 2, kernel_size=(4, 4), stride=(4, 4),
						  padding=0),  # 16 -> 4
				nn.BatchNorm2d(embed_dims * 2),
				nn.ReLU(),
				nn.Conv2d(in_channels=embed_dims * 2, out_channels=embed_dims, kernel_size=(1, 1), stride=(1, 1),
						  padding=0),
				# nn.BatchNorm2d(embed_dims),
			)
		else:
			raise TypeError(f'Unsupported conv layer type {conv_type}')
		# self.init_weights()

	def init_weights(self):
		for module in self.projection:
			if isinstance(module, nn.Conv2d):
				kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
				constant_init_(module.bias, constant_value=0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.bias, 0)
				nn.init.constant_(module.weight, 1.0)

	def forward(self, x):
		layer_type = type(self.projection)
		if layer_type == nn.Conv3d:
			x = rearrange(x, 'b t (c tc) h w -> b (c t) tc h w', tc=self.tube_size[1])
			x = rearrange(x, 'b p c h w -> b c p h w')
			x = self.projection(x)
			x = rearrange(x, 'b c (nc nt) h w -> b nt nc (h w) c', nc=self.img_size[1] // self.tube_size[1])
		elif layer_type == nn.Conv2d:
			x = rearrange(x, 'b t (c tc) h w -> (b t c) tc h w', tc=self.tube_size[1])
			x = self.projection(x)
			x = rearrange(x, '(b t c) p h w -> b t c (h w) p', t=self.img_size[0] // self.tube_size[0],
						  c=self.img_size[1] // self.tube_size[1])
		elif layer_type == nn.Linear:
			x = rearrange(x, 'b (t tt) (c tc) (h th) (w tw) -> b t c (h w) (tt tc th tw)',
						  tt=self.tube_size[0], tc=self.tube_size[1], th=self.tube_size[2], tw=self.tube_size[3])
			x = self.projection(x)
		elif layer_type == nn.Sequential:
			x = rearrange(x, 'b t (c tc) h w -> (b t c) tc h w', tc=self.tube_size[1])
			x = self.projection(x)
			x = rearrange(x, '(b t c) p h w -> b t c (h w) p', t=self.img_size[0] // self.tube_size[0],
						  c=self.img_size[1] // self.tube_size[1])
		else:
			raise TypeError(f'Unsupported conv layer type {layer_type}')

		return x


class MultiheadAttentionWithPreNorm(nn.Module):
	"""Implements Multi-head Attention with residual connection.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`. Default: 0.0.
        norm_layer (class): Class name for normalization layer. Defaults to nn.LayerNorm.
    """

	def __init__(self,
				 embed_dims,
				 num_heads,
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 norm_layer=nn.LayerNorm,
				 layer_drop=dict(type=DropPath, dropout_p=0.),
				 **kwargs):
		super().__init__()

		self.norm_layer = norm_layer(embed_dims)

		self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=attn_dropout)

		self.proj_drop = nn.Dropout(attn_proj_dropout)

		dropout_p = layer_drop.pop('dropout_p')
		layer_drop = layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

	def forward(self, x, **kwargs):
		residual = x
		x = self.norm_layer(x)

		attn_out = self.attn(query=x, key=x, value=x)[0]

		x = residual + self.layer_drop(self.proj_drop(attn_out))
		return x


class MultiheadCrossAttentionWithPreNorm(nn.Module):
	"""Implements Multi-head Cross-Attention with residual connection.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`. Default: 0.0.
        norm_layer (class): Class name for normalization layer. Defaults to nn.LayerNorm.
    """

	def __init__(self,
				 embed_dims,
				 num_heads,
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 norm_layer=nn.LayerNorm,
				 layer_drop=dict(type=DropPath, dropout_p=0.),
				 **kwargs):
		super().__init__()

		self.norm_query = norm_layer(embed_dims)
		self.norm_key = norm_layer(embed_dims)
		self.norm_value = norm_layer(embed_dims)

		self.cross_attention = nn.MultiheadAttention(embed_dims, num_heads, dropout=attn_dropout)

		self.proj_drop = nn.Dropout(attn_proj_dropout)
		dropout_p = layer_drop.pop('dropout_p')
		layer_drop = layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

	def forward(self, query, key=None, value=None, **kwargs):
		query = self.norm_query(query)
		key = self.norm_key(key)
		value = self.norm_value(value)
		residual = query

		attn_out = self.cross_attention(query=query, key=key, value=value)[0]
		attn_out = residual + self.layer_drop(self.proj_drop(attn_out))

		return attn_out


class FFNWithPreNorm(nn.Module):
	"""Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as `MultiheadAttention`. Defaults: 256.
        hidden_channels (int): The hidden dimension of FFNs. Defaults: 1024.
        num_layers (int, optional): The number of fully-connected layers in FFNs. Default: 2.
        act_layer (dict, optional): The activation layer for FFNs. Default: nn.GELU
        norm_layer (class): Class name for normalization layer. Defaults to nn.LayerNorm.
        proj_dropout (float, optional): Probability of an element to be zeroed in FFN. Default 0.0.
    """

	def __init__(self,
				 embed_dims=256,
				 hidden_dims=1024,
				 num_layers=2,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 ffn_proj_dropout=0.,
				 layer_drop=None,
				 **kwargs):
		super().__init__()
		assert num_layers >= 2, 'num_layers should be no less ' \
								f'than 2. got {num_layers}.'

		self.norm_layer = norm_layer(embed_dims)
		self.layer_drop = nn.Dropout(ffn_proj_dropout)
		layers = []
		in_channels = embed_dims
		for _ in range(num_layers - 1):
			layers.append(nn.Linear(in_channels, hidden_dims))
			layers.append(act_layer())
			layers.append(nn.Dropout(ffn_proj_dropout))
			in_channels = hidden_dims
		layers.append(nn.Linear(hidden_dims, embed_dims))
		layers.append(nn.Dropout(ffn_proj_dropout))
		self.layers = nn.ModuleList(layers)

		if layer_drop:
			dropout_p = layer_drop.pop('dropout_p')
			layer_drop = layer_drop.pop('type')
			self.layer_drop = layer_drop(dropout_p)
		else:
			self.layer_drop = nn.Identity()

		# for layer in self.layers:
		#    self.init_weights(layer)

	def init_weights(self, module):
		if isinstance(module, nn.Linear):
			if hasattr(module, 'weight') and module.weight is not None:
				kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu', distribution='normal')
			if hasattr(module, 'bias') and module.bias is not None:
				constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		residual = x

		x = self.norm_layer(x)
		for layer in self.layers:
			x = layer(x)

		return residual + self.layer_drop(x)


class MultiscaleConv2DWithPreNorm(nn.Module):
	"Implements Multiscale CNN equation."

	def __init__(self,
				 embed_dims=128,
				 multi_conv2d_hidden_dims=256,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 multi_conv_dropout=0.0,
				 layer_drop=None,
				 filter_sizes=[(1, 1), (3, 3), (5, 5)],
				 height=4,
				 width=12,
				 ):
		super().__init__()

		self.height = height
		self.width = width
		self.num_multi_conv = int(len(filter_sizes))
		self.filter_sizes = filter_sizes
		self.norm_layer = norm_layer(embed_dims)
		self.layer_drop = nn.Dropout(multi_conv_dropout)
		multi_conv_layers = []

		for i, filter_size in enumerate(self.filter_sizes):
			multi_conv_layers.append(
				nn.Sequential(
					nn.Conv2d(embed_dims, multi_conv2d_hidden_dims, kernel_size=(filter_size[0], filter_size[1]),
							  stride=(1, 1),
							  padding=((filter_size[0] - 1) // 2, (filter_size[1] - 1) // 2)),
					nn.BatchNorm2d(multi_conv2d_hidden_dims),
					nn.Dropout(multi_conv_dropout),
					nn.ReLU(),
					nn.Conv2d(multi_conv2d_hidden_dims, embed_dims, kernel_size=(1, 1), stride=(1, 1), ),
					nn.BatchNorm2d(embed_dims),
					nn.Dropout(multi_conv_dropout),
					nn.ReLU(),
				)
			)

		self.multi_conv_layers = nn.ModuleList(multi_conv_layers)

		if layer_drop:
			dropout_p = layer_drop.pop('dropout_p')
			layer_drop = layer_drop.pop('type')
			self.layer_drop = layer_drop(dropout_p)
		else:
			self.layer_drop = nn.Identity()

	def forward(self, x):
		residual = x
		x = self.norm_layer(x)
		x = rearrange(x, 'b (h w) d -> b d h w', h=self.height, w=self.width)

		conv_layer_outs = []
		for conv_layer in self.multi_conv_layers:
			f_map = conv_layer(x)
			conv_layer_outs.append(f_map.unsqueeze(dim=1))
		x = torch.div(torch.sum(torch.cat(conv_layer_outs, dim=1), dim=1), self.num_multi_conv)

		x = rearrange(x, 'b d h w -> b (h w) d', h=self.height, w=self.width)

		return residual + self.layer_drop(x)


class MultiscaleConv1DWithPreNorm(nn.Module):
	"Implements Multiscale CNN equation."

	def __init__(self,
				 embed_dims=256,
				 hidden_dims=1024,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 multi_conv_dropout=0.0,
				 layer_drop=None,
				 filter_sizes=[1, 3, 5],
				 ):
		super().__init__()

		self.filter_sizes = filter_sizes
		self.norm_layer = norm_layer(embed_dims)
		self.layer_drop = nn.Dropout(multi_conv_dropout)
		multi_conv_layers = []

		for i, filter_size in enumerate(self.filter_sizes):
			multi_conv_layers.append(
				nn.Sequential(
					nn.Conv1d(embed_dims, embed_dims, kernel_size=filter_size, stride=(1),
							  padding=int((filter_size - 1) // 2), ),
					nn.BatchNorm1d(embed_dims),
					nn.Dropout(multi_conv_dropout),
					nn.ReLU(),
					nn.Conv1d(embed_dims, embed_dims, kernel_size=1, stride=(1), ),
					nn.BatchNorm1d(embed_dims),
					nn.Dropout(multi_conv_dropout),
					nn.ReLU(),
				)
			)

		self.multi_conv_layers = nn.ModuleList(multi_conv_layers)

		if layer_drop:
			dropout_p = layer_drop.pop('dropout_p')
			layer_drop = layer_drop.pop('type')
			self.layer_drop = layer_drop(dropout_p)
		else:
			self.layer_drop = nn.Identity()

	def forward(self, x):
		B, np, _ = x.shape[0], x.shape[1], x.shape[2]
		residual = x
		x = self.norm_layer(x)

		x = rearrange(x, 'b p d -> b d p')
		conv_layer_outs = []
		for conv_layer in self.multi_conv_layers:
			f_map = conv_layer(x)
			conv_layer_outs.append(f_map.unsqueeze(dim=1))
		x = torch.div(torch.sum(torch.cat(conv_layer_outs, dim=1), dim=1), 3)
		x = rearrange(x, 'b d p -> b p d')
		return residual + self.layer_drop(x)


class TransformerContainer(nn.Module):

	def __init__(self,
				 embed_dims,
				 num_transformer_layers,
				 num_heads,
				 hidden_dims,
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 ffn_proj_dropout=0.,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 num_layers=2,
				 drop_path_rate=0.1,
				 ):
		super().__init__()
		self.transformer_layers = nn.ModuleList([])

		for i in range(num_transformer_layers):
			self.transformer_layers.append(nn.ModuleList(
				[
					MultiheadAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						attn_dropout=attn_dropout,
						attn_proj_dropout=attn_proj_dropout,
						ffn_proj_dropout=ffn_proj_dropout,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)
					),
					FFNWithPreNorm(
						embed_dims=embed_dims,
						hidden_dims=hidden_dims,
						num_layers=num_layers,
						act_layer=act_layer,
						norm_layer=norm_layer,
						ffn_proj_dropout=ffn_proj_dropout,
						layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)
					),
				]
			))

	def forward(self, x):
		for attn, ff in self.transformer_layers:
			x = attn(x)
			x = ff(x)

		return x


class SpatialTransformerContainer(nn.Module):

	def __init__(self,
				 embed_dims,
				 num_transformer_layers,
				 num_heads,
				 multi_conv2d_hidden_dims=256,
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 multi_conv_dropout=0.,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 drop_path_rate=0.1,
				 filter_sizes=[(1, 1), (3, 3), (5, 5)],
				 height=4,
				 width=12,
				 ):
		super().__init__()

		self.transformer_layers = nn.ModuleList([])

		for i in range(num_transformer_layers):
			self.transformer_layers.append(nn.ModuleList(
				[
					MultiheadAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						attn_dropout=attn_dropout,
						attn_proj_dropout=attn_proj_dropout,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)
					),
					MultiscaleConv2DWithPreNorm(
						embed_dims=embed_dims,
						multi_conv2d_hidden_dims=multi_conv2d_hidden_dims,
						multi_conv_dropout=multi_conv_dropout,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=drop_path_rate),
						filter_sizes=filter_sizes,
						height=height,
						width=width,
					),
				]
			))

	def forward(self, x):
		for attn, multi_conv in self.transformer_layers:
			x = attn(x)
			x = multi_conv(x)

		return x


class DecoderContainer(nn.Module):

	def __init__(self,
				 embed_dims,
				 num_transformer_layers,
				 num_heads,
				 hidden_dims,
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 ffn_proj_dropout=0.,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 num_layers=2,
				 drop_path_rate=0.1,
				 ):
		super().__init__()

		self.transformer_layers = nn.ModuleList([])

		for i in range(num_transformer_layers):
			self.transformer_layers.append(nn.ModuleList(
				[
					MultiheadAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						attn_dropout=attn_dropout,
						attn_proj_dropout=attn_proj_dropout,
						ffn_proj_dropout=ffn_proj_dropout,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)
					),
					MultiheadCrossAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						attn_dropout=attn_dropout,
						attn_proj_dropout=attn_proj_dropout,
						ffn_proj_dropout=ffn_proj_dropout,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)
					),
					FFNWithPreNorm(
						embed_dims=embed_dims,
						hidden_dims=hidden_dims,
						num_layers=num_layers,
						act_layer=act_layer,
						norm_layer=norm_layer,
						ffn_proj_dropout=ffn_proj_dropout,
						layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)
					),
				]
			))

		# self.apply(self.init_weights)

	def init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.trunc_normal_(module.weight, std=.02)
			# module.weight.data.normal_(mean=0.0, std=0.02)
			# module.weight.data.normal_(mean=0.0, std=0.01)
		if isinstance(module, nn.Linear) and module.bias is not None:
			nn.init.constant_(module.bias, 0)

	def forward(self, query, key=None, value=None):

		for self_attn, cross_attn, ff in self.transformer_layers:
			query = self_attn(query)
			query = cross_attn(query=query, key=key, value=value)
			query = ff(query)

		return query


class Legoformer(nn.Module):
	"""
    Spectral Transformer Model 6:
    Parallel DE Transformer and PSD Transformer.
    DE Transformer and PSD Transformer do not share weights
        DE Transformer: Number of transformer layers is 6.
        PSD Transformer: Number of transformer layers is 6.
        Cross-Attention Transformer: Number of transformer layers is 1. DE-Q, PSD-KV.
    """

	def __init__(self,
				 embed_dims: int,
				 num_transformer_layers: int,
				 num_heads: int,
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 ffn_proj_dropout=0.,
				 drop_path_rate=0.1,
				 ):
		super().__init__()

		self.bridge_layers = nn.ModuleList([])

		for i in range(num_transformer_layers):
			self.bridge_layers.append(nn.ModuleList(
				[
					TransformerContainer(
						num_transformer_layers=1,
						embed_dims=embed_dims,
						num_heads=num_heads,
						hidden_dims=embed_dims * 4,
						attn_dropout=attn_dropout,
						attn_proj_dropout=attn_proj_dropout,
						ffn_proj_dropout=ffn_proj_dropout,
						drop_path_rate=drop_path_rate,
					),
					TransformerContainer(
						num_transformer_layers=1,
						embed_dims=embed_dims,
						num_heads=num_heads,
						hidden_dims=embed_dims * 4,
						attn_dropout=attn_dropout,
						attn_proj_dropout=attn_proj_dropout,
						ffn_proj_dropout=ffn_proj_dropout,
						drop_path_rate=drop_path_rate,
					),
					DecoderContainer(
						num_transformer_layers=1,
						embed_dims=embed_dims,
						num_heads=num_heads,
						hidden_dims=embed_dims * 4,
						attn_dropout=attn_dropout,
						attn_proj_dropout=attn_proj_dropout,
						ffn_proj_dropout=ffn_proj_dropout,
						drop_path_rate=drop_path_rate,
					),
				]
			))

	def forward(self, x):
		B, nc, _ = x.shape[0], x.shape[1], x.shape[2]
		x_de, x_psd = x[:, 0:nc // 2, :], x[:, nc // 2:, :]
		encoder_de, encoder_psd, decoder_de = x_de, x_psd, x_de

		for transformer_de, transformer_psd, cross_transformer_de in self.bridge_layers:
			encoder_de, encoder_psd = transformer_de(encoder_de), transformer_psd(encoder_psd)

			decoder_de = encoder_de + encoder_psd + decoder_de
			decoder_de = cross_transformer_de(query=decoder_de, key=encoder_de, value=encoder_de)

		return decoder_de


class ClassificationHead(nn.Module):
	"""Classification head for Video Transformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

	def __init__(self,
				 num_classes,
				 in_channels,
				 init_std=0.02,
				 eval_metrics='no_finetune',  # 'finetune'
				 **kwargs):
		super().__init__()
		self.init_std = init_std
		self.eval_metrics = eval_metrics
		self.norm_layer = nn.LayerNorm(in_channels)
		self.cls_head = nn.Linear(in_channels, num_classes)
		self.apply(self.init_weights)

	def init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=0.01)
		if isinstance(module, nn.Linear) and module.bias is not None:
			nn.init.constant_(module.bias, 0)

	def forward(self, x):
		x = self.norm_layer(x)
		cls_score = self.cls_head(x)
		return cls_score


class TemporalEncoderTransformer(nn.Module):

	def __init__(self,
				 num_frames: int,
				 embed_dims,
				 num_transformer_layers: int,
				 num_heads,
				 hidden_dims,
				 num_layers=2,
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 ffn_proj_dropout=0.,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 drop_path_rate=0.1,
				 temporal_type="mean",
				 ):
		super().__init__()

		self.num_temporal_transformer_layers = num_transformer_layers
		self.temporal_type = temporal_type

		if num_transformer_layers != 0 and temporal_type == 'Transformer':
			self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dims))

		if self.temporal_type == "MeanPooling":
			self.temporal_transformer = nn.Identity()
		elif self.temporal_type == "Transformer":
			self.temporal_transformer = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				hidden_dims=embed_dims * 4,
				attn_dropout=attn_dropout,
				attn_proj_dropout=attn_proj_dropout,
				ffn_proj_dropout=ffn_proj_dropout,
				drop_path_rate=drop_path_rate,
			)
		elif self.temporal_type == "LSTM":
			self.temporal_transformer = nn.LSTM(input_size=embed_dims, hidden_size=embed_dims,
												batch_first=True, bidirectional=False, num_layers=1)
		elif self.temporal_type == "Conv_1D":
			self.temporal_transformer = nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3,
												  stride=1, padding=1, groups=embed_dims, bias=False)
			weight = torch.zeros(embed_dims, 1, 3)
			weight[:embed_dims // 4, 0, 0] = 1.0
			weight[embed_dims // 4:embed_dims // 4 + embed_dims // 2, 0, 1] = 1.0
			weight[-embed_dims // 4:, 0, 2] = 1.0
			self.temporal_transformer.weight = nn.Parameter(weight)
		elif self.temporal_type == "CausalConv1d":
			self.temporal_transformer = nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=3,
												  stride=1, padding=2, groups=embed_dims, bias=False)

		self.init_weights()

	def init_weights(self):
		if self.num_temporal_transformer_layers != 0 and self.temporal_type == 'Transformer':
			nn.init.trunc_normal_(self.temporal_embedding, std=.02)

	def forward(self, x):
		B, nt, np = x.shape[0], x.shape[1], x.shape[2]

		if self.temporal_type == "MeanPooling":
			x = x.mean(dim=1)
		elif self.temporal_type == "Transformer":
			# x_original = x
			if self.num_temporal_transformer_layers != 0 and self.temporal_type == 'Transformer':
				x += self.temporal_embedding.cuda()
			x = self.temporal_transformer(x)
			# x = x.type(x_original.dtype) + x_original
			x = x.mean(dim=1)
		elif self.temporal_type == "LSTM":
			x_original = x
			x, _ = self.temporal_transformer(x)
			x = x.type(x_original.dtype) + x_original
			x = x.mean(dim=1)
		elif self.temporal_type == "Conv_1D":
			x_original = x
			x = rearrange(x, 'b t d -> b d t')
			x = self.temporal_transformer(x)
			x = rearrange(x, 'b d t -> b t d')
			x = x.type(x_original.dtype) + x_original
			x = x.mean(dim=1)
		elif self.temporal_type == "CausalConv1d":
			x_original = x
			x = rearrange(x, 'b t d -> b d t')
			x = self.temporal_transformer(x)
			x = rearrange(x, 'b d t -> b t d')
			x = x[:, 0:nt, :]
			x = x.type(x_original.dtype) + x_original
			x = x.mean(dim=1)
		else:
			raise ValueError('Unknown optimizer: {}'.format(self.temporal_type))

		return x


class FactorisedEncoderTransformerEncoder(nn.Module):
	""" Factorised Space Time Spectrum Transformer Encoder - Model 2
        operator_order = ['Space_Transformer', 'Spectrum_Transformer', 'Time_Transformer']
    """

	def __init__(self,
				 num_frames: int,
				 num_channels: int,
				 num_spatial: int,
				 embed_dims: int,
				 multi_conv2d_hidden_dims: int,
				 num_spatial_transformer_layers: int,
				 num_spectral_transformer_layers: int,
				 num_temporal_transformer_layers: int,
				 num_heads: int,
				 operator_order="Spatial-Spectral-Temporal",
				 spectral_type="1",
				 spatial_type="Multi_Conv2D",
				 temporal_type="mean",
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 ffn_proj_dropout=0.,
				 multi_conv_dropout=0.,
				 drop_path_rate=0.1,
				 use_spectral_pos_embedding=False,
				 multi_gpu=False,
				 use_spectral_embed=False,
				 use_temporal_embed=False,
				 device=0,
				 ):
		super().__init__()

		""" x = [B, nt, nc, embed_dims] """
		self.spectral_type = spectral_type
		self.temporal_type = temporal_type
		self.operator_order = operator_order
		self.multi_gpu = multi_gpu
		self.device = device

		self.num_spatial_transformer_layers = num_spatial_transformer_layers
		self.num_spectral_transformer_layers = num_spectral_transformer_layers
		self.num_temporal_transformer_layers = num_temporal_transformer_layers

		self.use_spectral_pos_embedding = use_spectral_pos_embedding

		if num_spatial_transformer_layers != 0:
			self.spatial_embedding = nn.Parameter(torch.zeros(1, num_spatial, embed_dims))
		if num_spectral_transformer_layers != 0 and self.use_spectral_pos_embedding:
			self.spectral_embedding = nn.Parameter(torch.zeros(1, num_channels, embed_dims))

		self.transformer_layers = nn.ModuleList([])

		if self.num_spatial_transformer_layers == 0:
			spatial_transformer = nn.Identity()
		else:
			if spatial_type == "Transformer":
				spatial_transformer = TransformerContainer(
					num_transformer_layers=num_spatial_transformer_layers,
					embed_dims=embed_dims,
					num_heads=num_heads,
					hidden_dims=embed_dims * 4,
					attn_dropout=attn_dropout,
					attn_proj_dropout=attn_proj_dropout,
					ffn_proj_dropout=ffn_proj_dropout,
					drop_path_rate=drop_path_rate,
				)
			else:
				spatial_transformer = SpatialTransformerContainer(
					num_transformer_layers=num_spatial_transformer_layers,
					embed_dims=embed_dims,
					num_heads=num_heads,
					multi_conv2d_hidden_dims=multi_conv2d_hidden_dims,
					attn_dropout=attn_dropout,
					attn_proj_dropout=attn_proj_dropout,
					multi_conv_dropout=multi_conv_dropout,
					drop_path_rate=drop_path_rate,
					filter_sizes=[(1, 1), (3, 3), (5, 5)],
					height=4,
					width=4,
				)

		self.use_spectral_embed = use_spectral_embed
		self.use_temporal_embed = use_temporal_embed

		if num_spectral_transformer_layers == 0:
			spectral_transformer = nn.Identity()
		elif self.spectral_type == "Transformer":
			spectral_transformer = TransformerContainer(
				num_transformer_layers=num_spectral_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				hidden_dims=embed_dims * 4,
				attn_dropout=attn_dropout,
				attn_proj_dropout=attn_proj_dropout,
				ffn_proj_dropout=ffn_proj_dropout,
				drop_path_rate=drop_path_rate,
			)
		elif self.spectral_type == "Legoformer":
			spectral_transformer = Legoformer(
				embed_dims=embed_dims,
				num_transformer_layers=num_spectral_transformer_layers,
				num_heads=num_heads,
				attn_dropout=attn_dropout,
				attn_proj_dropout=attn_proj_dropout,
				ffn_proj_dropout=ffn_proj_dropout,
				drop_path_rate=drop_path_rate,
			)
		else:
			spectral_transformer = nn.Identity()

		self.init_weights()

		temporal_transformer = TemporalEncoderTransformer(
			embed_dims=embed_dims,
			num_frames=num_frames,
			num_transformer_layers=num_temporal_transformer_layers,
			num_heads=num_heads,
			hidden_dims=embed_dims * 4,
			attn_dropout=attn_dropout,
			attn_proj_dropout=attn_proj_dropout,
			ffn_proj_dropout=ffn_proj_dropout,
			drop_path_rate=drop_path_rate,
			temporal_type=temporal_type,
		)

		self.transformer_layers.append(spatial_transformer)
		self.transformer_layers.append(spectral_transformer)
		self.transformer_layers.append(temporal_transformer)

	def init_weights(self):
		if self.num_spatial_transformer_layers != 0:
			nn.init.trunc_normal_(self.spatial_embedding, std=.02)
			# nn.init.normal_(self.positional_embedding, std=0.01) # ActionCLIP
			# trunc_normal_(self.pos_embed, std=.02) #BridgeTower
		if self.num_spectral_transformer_layers != 0 and self.use_spectral_pos_embedding:
			nn.init.trunc_normal_(self.spectral_embedding, std=.02)

	def forward(self, x):
		""" x = [B, nt, nc, nh * nw, embed_dims] """
		B, nt, nc, np = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		spatial_transformer, spectral_transformer, temporal_transformer = *self.transformer_layers,

		x = rearrange(x, 'b t c p d -> (b t c) p d', b=B, t=nt, c=nc)
		if self.num_spatial_transformer_layers != 0:
			x += self.spatial_embedding.cuda()

		x = spatial_transformer(x)
		x = rearrange(x, '(b t c) p d -> b t c p d', b=B, t=nt, c=nc)
		x = reduce(x, 'b t c p d -> b t c d', 'mean', b=B, t=nt, c=nc)

		x = rearrange(x, 'b t c d -> (b t) c d', b=B, t=nt)
		if self.num_spectral_transformer_layers != 0 and self.use_spectral_pos_embedding:
			x += self.spectral_embedding.cuda()

		x = spectral_transformer(x)
		x = rearrange(x, '(b t) c d -> b t c d', b=B, t=nt)

		x = reduce(x, 'b t c d -> b t d', 'mean')

		x = temporal_transformer(x)
		return x


class ViViTEmotionNet(nn.Module):
	"""ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
        <https://arxiv.org/abs/2103.15691>

    Tubelet embedding combines features from time, frequency, and spatial dimensions simultaneously
    Args:
        image_frames (int): Number of frames in the EEG signal.
        image_channels (int): Number of spectrum in the EEG signal.
        image_height (int): Height of spatial map in the EEG signal.
        image_width (int): Width of spatial map in the EEG signal.
        tubelet_frames (int): Number of frames in the tubelet embedding.
        tubelet_channels (int): Number of spectrum in the tubelet embedding.
        tubelet_height (int): Height of the tubelet embedding.
        tubelet_width (int): Width of the tubelet embedding.

        num_classes (int): Number of classes for classification. Defaults to 3.
        num_transformer_layers (int): Number of transformer layers. Defaults to 6.
        embed_dims (int): Dimensions of embedding. Defaults to 256.
        num_heads (int): Number of parallel attention heads. Defaults to 4.
        attn_dropout (float): Probability of dropout layer in Multi-Head Self-Attention on attn_output_weights. Defaults to 0..
        proj_dropout (float): Probability of dropout layer. Defaults to 0..
        embed_dropout (float): Probability of dropout layer before send into transformer layers. Defaults to 0..
        model_Type (int): Type of attentions in TransformerCoder. Defaults to 3.
        multi_gpu (bool): Boolean flag to indicate whether to use multiple GPUs
        device (int): GPU number. Defaults to 0.
    """

	def __init__(self,
				 image_frames,
				 image_channels,
				 image_height,
				 image_width,
				 tubelet_frames,
				 tubelet_channels,
				 tubelet_height,
				 tubelet_width,
				 num_classes=3,
				 num_transformer_layers=[6, 6, 6],
				 embed_dims=128,
				 num_heads=4,
				 multi_conv2d_hidden_dims=256,
				 operator_order="Spatial-Spectral-Temporal",
				 spatial_type="Multi_Conv2D",
				 spectral_type="1",
				 temporal_type="MeanPooling",
				 attn_dropout=0.,
				 attn_proj_dropout=0.,
				 ffn_proj_dropout=0.,
				 multi_conv_dropout=0.,
				 drop_path_rate=0.1,
				 dropout_after_pos_embed=0.3,
				 conv_type='Linear',
				 use_spectral_pos_embedding=False,
				 model_Type=3,
				 multi_gpu=True,
				 device=0, ):
		super().__init__()

		assert image_frames % tubelet_frames == 0 and image_channels % tubelet_channels == 0 and \
			   image_height % tubelet_height == 0 and image_width % tubelet_width == 0, 'Image dimensions must be divisible by the patch size.'

		self.image_frames = image_frames
		self.image_channels = image_channels
		self.image_height = image_height
		self.image_width = image_width

		self.tubelet_frames = tubelet_frames
		self.tublet_channels = tubelet_channels
		self.tubelet_height = tubelet_height
		self.tubelet_width = tubelet_width

		self.multi_gpu = multi_gpu
		self.device = device

		self.nHight = self.image_height // self.tubelet_height
		self.nWidth = self.image_width // self.tubelet_width

		self.num_spatial_transformer_layers = num_transformer_layers[0]
		self.num_spectral_transformer_layers = num_transformer_layers[1]
		self.num_temporal_transformer_layers = num_transformer_layers[2]

		self.embed_dims = embed_dims

		self.tubelet_embedding = PatchEmbed(
			img_size=[image_frames, image_channels, image_height, image_width],
			tube_size=[tubelet_frames, tubelet_channels, tubelet_height, tubelet_width],
			embed_dims=embed_dims,
			conv_type=conv_type)

		self.drop_after_pos = nn.Dropout(dropout_after_pos_embed)
		if model_Type == 2:  # Divided Space Time Transformer Encoder - Model 2
			""" operator_order = ['Space_Transformer', 'Spectrum_Transformer', 'Time_Transformer'] """
			if operator_order == "Spatial-Spectral-Temporal":
				self.transformer = FactorisedEncoderTransformerEncoder(
					num_frames=image_frames // tubelet_frames,
					num_channels=image_channels // tubelet_channels,
					num_spatial=self.nHight * self.nWidth,
					embed_dims=embed_dims,
					multi_conv2d_hidden_dims=multi_conv2d_hidden_dims,
					num_spatial_transformer_layers=self.num_spatial_transformer_layers,
					num_spectral_transformer_layers=self.num_spectral_transformer_layers,
					num_temporal_transformer_layers=self.num_temporal_transformer_layers,
					operator_order=operator_order,
					spatial_type=spatial_type,
					spectral_type=spectral_type,
					temporal_type=temporal_type,
					num_heads=num_heads,
					attn_dropout=attn_dropout,
					attn_proj_dropout=attn_proj_dropout,
					ffn_proj_dropout=ffn_proj_dropout,
					multi_conv_dropout=multi_conv_dropout,
					drop_path_rate=drop_path_rate,
					use_spectral_pos_embedding=use_spectral_pos_embedding,
					multi_gpu=multi_gpu,
					device=device,
				)
		scale = embed_dims ** -0.5

		self.image_proj = nn.Parameter(torch.randn(embed_dims, 512))


		self.ClassificationHead = ClassificationHead(
			num_classes=num_classes,
			in_channels=embed_dims,
			init_std=0.02,
			eval_metrics='no_finetune',  # 'finetune'
		)

		self.apply(self.init_weights)

	def init_weights(self, module):
		if isinstance(module, nn.LayerNorm):
			nn.init.constant_(module.bias, 0)
			nn.init.constant_(module.weight, 1.0)
		elif isinstance(module, nn.BatchNorm1d):
			nn.init.constant_(module.bias, 0)
			nn.init.constant_(module.weight, 1.0)
		elif isinstance(module, nn.BatchNorm2d):
			nn.init.constant_(module.bias, 0)
			nn.init.constant_(module.weight, 1.0)

		if self.image_proj is not None:
			nn.init.normal_(self.image_proj, std=self.embed_dims ** -0.5)

	def encode_image(self, image):
		image = self.tubelet_embedding(image)
		image = self.transformer(image)
		if self.image_proj is not None:
			return image @ self.image_proj
		return image

	def forward(self, x):
		x = self.tubelet_embedding(x)
		x = self.transformer(x)
		x = self.ClassificationHead(x)
		return x


def create_model(config):
	if config['network'].get('encoder_type', 'vivit') == 'stge_dual':
		from modules.STGEDualEmotionNet import create_stge_dual_model
		return create_stge_dual_model(config)

	return ViViTEmotionNet(
		image_frames=config['network']['image_frames'],
		image_channels=config['network']['image_channels'],
		image_height=config['network']['image_height'],
		image_width=config['network']['image_width'],

		tubelet_frames=config['network']['tubelet_frames'],
		tubelet_channels=config['network']['tubelet_channels'],
		tubelet_height=config['network']['tubelet_height'],
		tubelet_width=config['network']['tubelet_width'],

		num_classes=config['data']['num_classes'],

		num_transformer_layers=config['network']['num_transformer_layers'],
		embed_dims=config['network']['embed_dims'],
		num_heads=config['network']['num_heads'],

		operator_order=config['network']['operator_order'],
		spectral_type=config['network']['spectral_type'],
		spatial_type=config['network']['spatial_type'],  # "Multi_Conv2D",
		temporal_type=config['network']['temporal_type'],

		attn_dropout=config['network']['attn_dropout'],
		attn_proj_dropout=config['network']['attn_proj_dropout'],
		ffn_proj_dropout=config['network']['ffn_proj_dropout'],
		multi_conv_dropout=config['network']['multi_conv_dropout'],
		drop_path_rate=config['network']['drop_path_rate'],
		dropout_after_pos_embed=config['network']['dropout_after_pos_embed'],
		conv_type=config['network']['conv_type'],
		use_spectral_pos_embedding=config['network']['use_spectral_pos_embedding'],
		model_Type=config['network']['model_type'],
		multi_gpu=config['multi_gpu'],
		device=config['gpu_device_id'],
	)


if __name__ == '__main__':
	x = torch.rand([8, 4, 12, 64, 64]).cuda()

	initial_memory = torch.cuda.memory_allocated()

	# [BatchSize, T, C, H, W]
	vivit = ViViTEmotionNet(
		image_frames=4,
		image_channels=12,
		image_height=64,
		image_width=64,
		tubelet_frames=1,
		tubelet_channels=1,
		tubelet_height=16,
		tubelet_width=16,
		num_classes=3,
		num_transformer_layers=[0, 2, 2],  # 0.316123M
		embed_dims=128,
		num_heads=4,
		multi_conv2d_hidden_dims=128,
		operator_order="Spatial-Spectral-Temporal",  # "Spatial-Spectral-Temporal"
		spatial_type="Multi_Conv2D",  # "Multi_Conv2D"
		spectral_type="Legoformer",
		# "Trans_SingleStreamDecoder" "TwoBridgeLayer" "Transformer" "SingleStreamDecoder"
		temporal_type="Transformer",  # "MeanPooling" "Transformer" "LSTM" "CausalConv1d"
		attn_dropout=0.0,
		attn_proj_dropout=0.0,
		ffn_proj_dropout=0.1,
		multi_conv_dropout=0.1,
		drop_path_rate=0.1,
		dropout_after_pos_embed=0.3,
		conv_type='Conv_Stem',  # 'Linear' 'Conv2d' 'Conv_Stem'
		use_spectral_pos_embedding=False,
		model_Type=2,
		multi_gpu=False,
		device=0,
	).cuda()

	print(vivit)

	parameters = filter(lambda p: p.requires_grad, vivit.parameters())
	parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
	print('Trainable Parameters: %fM' % parameters)
	# for name, param in vivit.named_parameters():
	# print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
	# summary(vivit, (4, 12, 64, 64))
	out = vivit(x)
	print(out)
	memory_after_init = torch.cuda.memory_allocated() - initial_memory

	print(f"memory：{memory_after_init} bytes")
