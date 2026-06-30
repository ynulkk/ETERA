import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
from utils.saving import pretrain_best_saving, pretrain_epoch_saving
from datasets.dataset import EEG_Dataset
from datasets import utils as dataset_utils
import wandb

def _move_to_cuda_float(batch):
	if isinstance(batch, torch.Tensor):
		return batch.cuda().float()
	if isinstance(batch, (tuple, list)):
		return tuple(_move_to_cuda_float(item) for item in batch)
	return batch

def _make_stge_param_groups(config, model):
	base_lr = config['solver']['lr']
	stge_config = config.get('network', {}).get('stge_dual', {})
	raw_lr = base_lr * float(stge_config.get('raw_lr_mult', 0.2))
	raw_prefixes = (
		'raw_branch.',
		'raw_delta.',
		'fusion.',
		'raw_gate_logit',
		'module.raw_branch.',
		'module.raw_delta.',
		'module.fusion.',
		'module.raw_gate_logit',
	)
	base_params, raw_params = [], []
	for name, param in model.named_parameters():
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
	print('STGE-Dual pretrain optimizer param groups: base_lr={}, raw_lr={}, raw_lr_mult={}'.format(
		base_lr, raw_lr, stge_config.get('raw_lr_mult', 0.2)
	))
	return param_groups

class Pretrainer(object):
	def __init__(self,
				 config,
				 train_data_list: list,
				 val_data_list: list,
				 workers: int,
				 model: nn.Module,
				 loss_function,
				 save_result_path: str,
				 ):
		super(Pretrainer, self).__init__()

		self.config = config
		self.train_data_list = train_data_list
		self.val_data_list = val_data_list
		self.train_batch_size = config['solver']['train_batch_size']
		self.val_batch_size = config['solver']['val_batch_size']
		self.workers = workers

		self.num_classes = config['data']['num_classes']

		self.num_epochs = config['solver']['num_epochs']

		self.model = model
		self.loss_function = loss_function

		self.multi_gpu = config['multi_gpu']
		self.device = config['gpu_device_id']

		self.optimizer = self._init_optimizer(config)
		self.lr_scheduler = self._init_lr_scheduler(config)

		self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
		self.save_result_path = save_result_path

	def _init_optimizer(self, config):
		if config.get('network', {}).get('encoder_type', '') == 'stge_dual':
			param_groups = _make_stge_param_groups(config, self.model)
		else:
			param_groups = self.model.parameters()
		if config['solver']['optim'] == 'SGD':
			optimizer = torch.optim.SGD(params=param_groups, lr=config['solver']['lr'],
										momentum=config['solver']['momentum'], weight_decay=config['solver']['weight_decay'])
		elif config['solver']['optim'] == 'Adam':
			optimizer = torch.optim.Adam(params=param_groups, lr=config['solver']['lr'],
										 weight_decay=config['solver']['weight_decay'])
		elif config['solver']['optim'] == 'AdamW':
			optimizer = torch.optim.AdamW(params=param_groups, lr=config['solver']['lr'],
										  weight_decay=config['solver']['weight_decay'])
		else:
			raise ValueError('Unknown optimizer: {}'.format(config['solver']['optim']))
		return optimizer

	def _init_lr_scheduler(self, config):
		if config['solver']['lr_type'] == "MultiStepLR":
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['solver']['milestones'],
																gamma=config['solver']['lr_adjust_epoch'])
		elif config['solver']['lr_type'] == "Cosine":
			lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['solver']['num_epochs'],
																	  eta_min=0, last_epoch=-1)
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
				f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0.0
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

	def _train_one_epoch(self, epoch: int, data_loader, phase='training'):
		if phase == 'training':
			self.model.train()
		else:
			self.model.eval()

		running_loss, cnt_correct = 0.0, 0

		for inputs, labels in tqdm(data_loader, desc='Processing'):
			# inputs = inputs.to(self.device).float()  # dtype: torch.float64
			# labels = labels.to(self.device).float()  # dtype: torch.int64
			inputs = _move_to_cuda_float(inputs)
			labels = labels.cuda().float()  # one-hot

			if phase == 'training':
				self.optimizer.zero_grad()
				outputs = self.model(inputs)
			else:
				with torch.no_grad():
					outputs = self.model(inputs)

			# outputs = F.softmax(outputs, dim=1)
			loss = self.loss_function(outputs, labels)
			running_loss += loss.item()

			_, predictions = torch.max(outputs.data, dim=1)
			labels = torch.argmax(labels, dim=1)
			cnt_correct += (predictions == labels).sum().item()

			if phase == 'training':
				loss.backward()
				self.optimizer.step()
			else:
				with torch.no_grad():
					for pred, label in zip(predictions, labels):
						self.confusion_matrix[label][pred] += 1

		avg_epoch_loss = float(running_loss) / len(data_loader.dataset)
		avg_epoch_acc = float(cnt_correct) / float(len(data_loader.dataset))

		if phase == 'training':
			print('[%d][%s] train_loss: %.4f' % (epoch + 1, phase, avg_epoch_loss) + ' , train_acc: %.4f' % (
				avg_epoch_acc))
			wandb.log({"train_loss": avg_epoch_loss, "epoch": epoch})
			wandb.log({"train_acc": avg_epoch_acc, "epoch": epoch})
		else:
			with torch.no_grad():
				print('[%d][%s] val_loss: %.4f' % (epoch + 1, phase, avg_epoch_loss) + ' , val_acc: %.4f' % (
					avg_epoch_acc))
				wandb.log({"val_loss": avg_epoch_loss, "epoch": epoch})
				wandb.log({"val_acc": avg_epoch_acc, "epoch": epoch})

		return avg_epoch_loss, avg_epoch_acc

	def training(self):

		os.environ["WANDB_API_KEY"] = ''
		os.environ["WANDB_MODE"] = "offline"
		wandb.init(project=self.config['wandb_data']['project'], name=self.config['wandb_data']['name'], config=self.config)
		wandb.config.update(self.config)
		wandb.log({"model:": self.model})
		wandb.watch(self.model)
		arti_code = wandb.Artifact('ipynb', type='code')
		arti_code.add_file('./modules/ViViTEmotionNet.py')
		wandb.log_artifact(arti_code)

		self.train_loader = self._get_data_loader(self.train_data_list, self.config['solver']['train_batch_size'])
		self.val_loader = self._get_data_loader(self.val_data_list, self.config['solver']['val_batch_size'])

		train_losses, train_accs = [], []
		val_losses, val_accs = [], []

		if self.save_result_path == None:
			self.save_result_path = "../train_result"

		self.save_result_path = os.path.join(self.save_result_path,
											 time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))).replace(
			'\\', '/')
		if not os.path.exists(self.save_result_path):
			os.makedirs(self.save_result_path)

		train_losses_file_path = os.path.join(self.save_result_path, f'train_losses.txt').replace('\\', '/')
		train_accs_file_path = os.path.join(self.save_result_path, f'train_accs.txt').replace('\\', '/')
		val_losses_file_path = os.path.join(self.save_result_path, f'val_losses.txt').replace('\\', '/')
		val_accs_file_path = os.path.join(self.save_result_path, f'val_accs.txt').replace('\\', '/')

		for epoch in range(self.num_epochs):
			print("Epoch: {}/{}".format(epoch + 1, self.num_epochs))

			self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

			train_epoch_loss, train_epoch_acc = self._train_one_epoch(epoch, data_loader=self.train_loader,
																	  phase='training')
			val_epoch_loss, val_epoch_acc = self._train_one_epoch(epoch, data_loader=self.val_loader,
																  phase='validation')

			if epoch == 0 or val_epoch_acc > np.max(val_accs):
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

				pretrain_best_saving(self.save_result_path, epoch, self.model, self.optimizer)
				wandb.save("best_model.pt")

			if (epoch + 1) % 10 == 0:
				pretrain_epoch_saving(self.save_result_path, epoch + 1, self.model, self.optimizer)
			train_losses.append(train_epoch_loss)
			train_accs.append(train_epoch_acc)
			val_losses.append(val_epoch_loss)
			val_accs.append(val_epoch_acc)

			if (epoch + 1) % 20 == 0:
				with open(train_losses_file_path, 'w+') as f:
					f.write(str(train_losses))
				with open(train_accs_file_path, 'w+') as f:
					f.write(str(train_accs))
				with open(val_losses_file_path, 'w+') as f:
					f.write(str(val_losses))
				with open(val_accs_file_path, 'w+') as f:
					f.write(str(val_accs))
		print(f"Training end. max val_acc = {np.max(val_accs)}")
		max_val_acc = np.max(val_accs)
		wandb.log({"max_val_acc": max_val_acc})
		wandb.finish()
		return max_val_acc
