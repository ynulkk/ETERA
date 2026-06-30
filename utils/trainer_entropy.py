import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
from utils.saving import best_saving, epoch_saving
from utils import Text_Prompt

import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR

from datasets.dataset import EEG_Dataset
from datasets import utils as dataset_utils

import clip
from utils.tools import *
from utils.KLLoss import KLLoss
import wandb
from torch.autograd import gradcheck

def _move_to_cuda_float(batch):
	if isinstance(batch, torch.Tensor):
		return batch.cuda().float()
	if isinstance(batch, (tuple, list)):
		return tuple(_move_to_cuda_float(item) for item in batch)
	return batch

def _batch_size(batch):
	if isinstance(batch, torch.Tensor):
		return batch.shape[0]
	if isinstance(batch, (tuple, list)):
		return _batch_size(batch[0])
	raise TypeError("Unsupported batch type: {}".format(type(batch)))

def _make_stge_param_groups(config, model_image):
	base_lr = config['solver']['lr']
	stge_config = config.get('network', {}).get('stge_dual', {})
	raw_lr = base_lr * float(stge_config.get('raw_lr_mult', 0.2))
	raw_prefixes = (
		'model.raw_branch.',
		'model.raw_delta.',
		'model.fusion.',
		'model.raw_gate_logit',
		'module.model.raw_branch.',
		'module.model.raw_delta.',
		'module.model.fusion.',
		'module.model.raw_gate_logit',
	)
	base_params, raw_params = [], []
	for name, param in model_image.named_parameters():
		if not param.requires_grad:
			continue
		if name.startswith(raw_prefixes):
			raw_params.append(param)
		else:
			base_params.append(param)
	param_groups = []
	if base_params:
		param_groups.append({'params': base_params, 'lr': base_lr})
	if raw_params:
		param_groups.append({'params': raw_params, 'lr': raw_lr})
	print('STGE-Dual optimizer param groups: base_lr={}, raw_lr={}, raw_lr_mult={}'.format(
		base_lr, raw_lr, stge_config.get('raw_lr_mult', 0.2)
	))
	return param_groups

class TextCLIP(nn.Module):
	def __init__(self, model):
		super(TextCLIP, self).__init__()
		self.model = model

	def forward(self, text):
		return self.model.encode_text(text)


class ImageCLIP(nn.Module):
	def __init__(self, model):
		super(ImageCLIP, self).__init__()
		self.model = model

	def forward(self, image):
		return self.model.encode_image(image)


class Trainer(object):
	def __init__(self,
				 config,
				 train_data_list: list,
				 val_data_list: list,
				 workers: int,
				 model_text: nn.Module,
				 model_image: nn.Module,
				 start_epoch: int,
				 save_result_path: str,
				 ):
		super(Trainer, self).__init__()

		self.config = config
		self.train_data_list = train_data_list
		self.val_data_list = val_data_list
		self.train_batch_size = config['solver']['train_batch_size']
		self.val_batch_size = config['solver']['val_batch_size']
		self.workers = workers
		#self.model_clip = model_clip
		self.model_image = model_image
		#self.model_text = TextCLIP(self.model_clip)
		self.model_text = model_text
		self.start_epoch = start_epoch

		self.classes_names = config['data']['classes_names']
		self.num_classes = config['data']['num_classes']
		self.logit_scale = config['solver'].get('logit_scale', 1.0)

		for param in self.model_text.parameters():
			param.requires_grad = False

		self.loss_img = KLLoss()
		self.loss_txt = KLLoss()

		self.optimizer = self._init_optimizer(self.config, self.model_image)
		self.lr_scheduler = self._init_lr_scheduler(self.config)

		self.num_epochs = config['solver']['num_epochs']

		self.multi_gpu = config['multi_gpu']

		self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
		self.save_result_path = save_result_path

	def _init_optimizer(self, config, model_image):
		if config.get('network', {}).get('encoder_type', '') == 'stge_dual':
			param_groups = _make_stge_param_groups(config, model_image)
		else:
			param_groups = [{'params': model_image.parameters(), 'lr': config['solver']['lr']}]
		if config['solver']['optim'] == 'Adam':
			optimizer = optim.Adam(param_groups,
										lr=config['solver']['lr'], betas=(0.9, 0.98), eps=1e-8,
										weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
			print('Adam')
		elif config['solver']['optim'] == 'SGD':

			optimizer = optim.SGD(param_groups,
									   config['solver']['lr'],
									   momentum=config['solver']['momentum'],
									   weight_decay=config['solver']['weight_decay'])
			print('SGD')
		elif config['solver']['optim'] == 'AdamW':
			optimizer = optim.AdamW(param_groups,
									betas=(0.9, 0.98), lr=config['solver']['lr'], eps=1e-8,
									weight_decay=config['solver']['weight_decay'])  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
			for param_group in optimizer.param_groups:
				print(param_group['lr'])
			print('AdamW')
		else:
			raise ValueError('Unknown optimizer: {}'.format(config['solver']['optim']))

		return optimizer

	def _init_lr_scheduler(self, config):
		if config['solver']['lr_type'] == 'Cosine':
			#lr_scheduler = WarmupCosineAnnealingLR(
			#	self.optimizer,
			#	config['solver']['num_epochs'],
			#	warmup_epochs=config['solver']['lr_warmup_step']
			#)
			lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
																	  T_max=config['solver']['num_epochs'],
																	  eta_min=0, last_epoch=-1)
		elif config['solver']['lr_type'] == 'multistep':
			if isinstance(config['solver']['lr_decay_step'], list):
				milestones = config['solver']['lr_decay_step']
			elif isinstance(config['solver']['lr_decay_step'], int):
				milestones = [
					config['solver']['lr_decay_step'] * (i + 1)
					for i in range(config['solver']['num_epochs'] //
								   config['solver']['lr_decay_step'])]
			else:
				raise ValueError("error learning rate decay step: {}".format(type(config['solver']['lr_decay_step'])))
			lr_scheduler = WarmupMultiStepLR(
				self.optimizer,
				milestones,
				warmup_epochs=config['solver']['lr_warmup_step']
			)
		else:
			raise ValueError('Unknown lr scheduler: {}'.format(config['solver']['lr_type']))
		return lr_scheduler

	def _calc_precision_recall_f1(self):
		precision = [0.0 for _ in range(self.num_classes)]
		recall = [0.0 for _ in range(self.num_classes)]
		f1 = [0.0 for _ in range(self.num_classes)]
		for i in range(self.num_classes):
			if 0 != np.sum(self.confusion_matrix[:, i]):
				precision[i] = self.confusion_matrix[i][i] / np.sum(self.confusion_matrix[:, i])

			if 0 != np.sum(self.confusion_matrix[i, :]):
				recall[i] = self.confusion_matrix[i][i] / np.sum(self.confusion_matrix[i, :])

		for i in range(self.num_classes):
			if 0 != (precision[i] + recall[i]):
				f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[
					i]) != 0 else 0.0
		return precision, recall, f1

	def _calc_accuracy(self):
		correct = 0
		for i in range(self.num_classes):
			correct += self.confusion_matrix[i][i]
		return correct / np.sum(self.confusion_matrix)

	def _get_data_loader(self, dataset_list, batch_size):
		dataset = EEG_Dataset(
			images_path=dataset_list,
			image_height=self.config['network']['image_height'],
			image_width=self.config['network']['image_width'],
			num_classes=self.config['data']['num_classes'],
			feature=self.config['network']['feature'],
			map_type='SST',
			return_raw=self.config['network'].get('return_raw', False),
			cfbm_smoothing=self.config['network'].get('cfbm_smoothing', {}),
		)

		return torch.utils.data.DataLoader(
			dataset=dataset,
			batch_size=batch_size,
			shuffle=True,
			pin_memory=True,
			num_workers=self.workers,
			generator=dataset_utils.make_torch_generator(self.config.get("random_seed", 42)),
			worker_init_fn=dataset_utils.make_worker_init_fn(self.config.get("random_seed", 42)),
		)

	def _encode_text_features(self, classes):
		self.model_text.eval()
		with torch.no_grad():
			text_features = self.model_text(classes.cuda())
			text_features = F.normalize(text_features, dim=-1)
		return text_features

	def _compute_class_logits(self, image_features, text_features, num_text_aug):
		image_features = F.normalize(image_features, dim=-1)
		similarity = self.logit_scale * (image_features @ text_features.T)
		similarity = similarity.view(image_features.shape[0], num_text_aug, self.num_classes)
		return similarity.mean(dim=1)

	def _validate(self, epoch, classes, val_loader, num_text_aug):
		self.model_image.eval()
		self.model_text.eval()

		num = 0
		corr_1 = 0

		with torch.no_grad():
			text_features = self._encode_text_features(classes)	# [num_text_prompt * num_classes, embed_dims]

			for iii, (images, labels) in enumerate(tqdm(val_loader)):
				batch_size = _batch_size(images)
				labels = torch.argmax(labels, dim=-1)
				labels = labels.cuda()
				images = _move_to_cuda_float(images)
				# # [1, embed_dims]
				image_features = self.model_image(images)	# [batch_size, embed_dims]

				class_logits = self._compute_class_logits(image_features, text_features, num_text_aug)
				_, preds = class_logits.topk(1, dim=-1)
				num += batch_size
				labels = labels.view(-1, 1)

				for i, (pred, label) in enumerate(zip(preds, labels)):
					self.confusion_matrix[label.item()][pred.item()] += 1
					if pred.item() == label.item():
						corr_1 += 1

		top1 = float(corr_1) / num * 100
		print('[{}/{}][Testing]: val_acc: {} '.format(epoch + 1, self.num_epochs, top1))
		#wandb.log({"val_acc": top1, "epoch": epoch})
		return top1

	def testing(self):

		val_loader = self._get_data_loader(self.val_data_list, self.config['solver']['val_batch_size'])

		start_epoch = 0
		classes, num_text_aug, text_dict = Text_Prompt.eeg_text_prompt(self.config['data']['classes_names'])

		prec1 = self._validate(start_epoch, classes, val_loader, num_text_aug)
		return prec1

	def training(self):
		"""
		os.environ["WANDB_API_KEY"] = ''
		os.environ["WANDB_MODE"] = "online"
		wandb.init(project=self.config['wandb_data']['project'], name=self.config['wandb_data']['name'],
				   config=self.config)
		wandb.config.update(self.config)
		wandb.log({"model_text": self.model_text})
		wandb.log({"model_image": self.model_image})
		wandb.watch(self.model_text)
		wandb.watch(self.model_image)
		arti_code = wandb.Artifact('ipynb', type='code')
		arti_code.add_file('./utils/trainer_entropy.py')
		arti_code.add_file('./utils/Text_Prompt.py')
		wandb.log_artifact(arti_code)
		"""
		self.loss_function = torch.nn.CrossEntropyLoss(reduction="sum")

		train_loader = self._get_data_loader(self.train_data_list, self.config['solver']['train_batch_size'])
		val_loader = self._get_data_loader(self.val_data_list, self.config['solver']['val_batch_size'])

		classes, num_text_aug, text_dict = Text_Prompt.eeg_text_prompt(self.classes_names)

		if self.save_result_path == None:
			self.save_result_path = "../train_result"

		self.save_result_path = os.path.join(self.save_result_path,
											 time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))).replace(
			'\\', '/')
		if not os.path.exists(self.save_result_path):
			os.makedirs(self.save_result_path)

		best_prec1 = 0.0

		self.model_text.eval()
		text_features = self._encode_text_features(classes)  # [num_text_prompt * num_classes, embed_dims]
		num_early_patience = 0
		for epoch in range(self.start_epoch, self.num_epochs):
			print("Epoch: {}/{}".format(epoch + 1, self.num_epochs))

			self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
			train_total_loss, corr_1, num_total = 0.0, 0, 0

			self.model_image.train()
			self.model_text.eval()
			# labels are one-hot tensor
			for i, (images, labels) in enumerate(tqdm(train_loader)):

				batch_size = _batch_size(images)
				images = _move_to_cuda_float(images)  # torch.float32
				labels = labels.cuda()
				labels_idx = torch.argmax(labels, dim=1)

				image_embedding = self.model_image(images)  # [BatciSize, 512] torch.float32

				class_logits = self._compute_class_logits(image_embedding, text_features.detach(), num_text_aug)

				loss = self.loss_function(class_logits, labels_idx)

				train_total_loss += loss.item()

				self.optimizer.zero_grad()
				loss.backward()
				#convert_models_to_fp32(self.model_image)
				self.optimizer.step()

				_, predictions = torch.max(class_logits.data, dim=1)
				corr_1 += (predictions == labels_idx).sum().item()

				num_total += batch_size

			avg_train_loss = float(train_total_loss) / num_total
			train_acc_top1 = float(corr_1) / num_total * 100
			print('[{}/{}][Training]: train_loss: {} train_acc: {}'.format(epoch + 1, self.num_epochs, avg_train_loss,
																		   train_acc_top1))
			#wandb.log({"train_loss": avg_train_loss, "epoch": epoch})
			#wandb.log({"train_acc": train_acc_top1, "epoch": epoch})

			val_total_acc_top1 = self._validate(epoch, classes, val_loader, num_text_aug)
			if val_total_acc_top1 > best_prec1:
				best_prec1 = max(val_total_acc_top1, best_prec1)

				precision, recall, f1 = self._calc_precision_recall_f1()
				accuracy = self._calc_accuracy()
				print("Confusion Matrix: ")
				print(self.confusion_matrix)
				print("Precision: " + str(precision))
				print("Recall: " + str(recall))
				print("F1: " + str(f1))
				print("mAccuracy: " + str(accuracy))
				print("mPrecision: " + str(np.mean(precision)))
				print("mRecall: " + str(np.mean(recall)))
				print("mF1: " + str(np.mean(f1)))
				num_early_patience = 0
				best_saving(self.save_result_path, epoch, self.model_image, self.optimizer)
			else :
				if (epoch + 1) % self.config['epoch_save_freq'] == 0:
					epoch_saving(self.save_result_path, epoch, self.model_image, self.optimizer)
				num_early_patience += 1
				if self.config['solver']['is_early_patience'] and num_early_patience >= self.config['solver']['early_patience']:
					print(f"Early stopping triggered! max val_acc = {best_prec1}")
					break

			#wandb.log({"max_val_acc": best_prec1, "epoch": epoch})
		print(f"Training end. max val_acc = {best_prec1}")
		#wandb.finish()
		return best_prec1
