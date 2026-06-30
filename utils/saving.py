# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch

def epoch_saving(working_dir, epoch, model_image, optimizer):
    epoch_save_name = '{}/model_epoch_save.pt'.format(working_dir)
    torch.save({
                    'epoch': epoch,
                    'ViViTEmotion_state_dict': model_image.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, epoch_save_name) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model_image, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'ViViTEmotion_state_dict': model_image.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename

def pretrain_best_saving(working_dir, epoch, model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename

def pretrain_epoch_saving(working_dir, epoch, model, optimizer):
    filename = "{}/last_model_{}.pt".format(working_dir, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)  # just change to your preferred folder/filename