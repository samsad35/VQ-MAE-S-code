from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
from ...base import Train
from ...data import VoxcelebSequential
from ...model import SpeechVQVAE, SpecMAE
import matplotlib.pyplot as plt
from .follow_up_mae import Follow
import math
from einops import repeat, rearrange
from math import log2, sqrt

torch.cuda.empty_cache()
from ...tools import griffin_lim, plot_spectrogram
from scipy.io.wavfile import write
from .idr_torch import IDR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.cuda.empty_cache()


class SpecMAE_Train(Train):
    def __init__(self, mae: SpecMAE,
                 training_data: VoxcelebSequential, validation_data: VoxcelebSequential,
                 config_training: dict = None,
                 tube_bool: bool = False,
                 follow: bool = True,
                 multigpu_bool: bool = False,
                 description: dict = None):
        super().__init__()
        if multigpu_bool:
            self.idr = IDR()
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=self.idr.size,
                                    rank=self.idr.rank)
            torch.cuda.set_device(self.idr.local_rank)
        self.device = torch.device(config_training['device'])
        """ Model """
        self.model = mae
        self.model.to(self.device)

        if multigpu_bool:
            self.model = DDP(self.model, device_ids=[self.idr.local_rank], find_unused_parameters=True)

        """ Dataloader """
        if multigpu_bool:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_data,
                                                                            num_replicas=self.idr.size,
                                                                            rank=self.idr.rank,
                                                                            shuffle=True)
            self.training_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                               batch_size=config_training[
                                                                              'batch_size'] // self.idr.size,
                                                               shuffle=False,
                                                               num_workers=config_training['num_workers'],
                                                               pin_memory=True,
                                                               drop_last=True,
                                                               sampler=train_sampler)
            val_sampler = torch.utils.data.distributed.DistributedSampler(validation_data,
                                                                          num_replicas=self.idr.size,
                                                                          rank=self.idr.rank,
                                                                          shuffle=True)
            self.validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                                 batch_size=config_training[
                                                                                'batch_size'] // self.idr.size,
                                                                 shuffle=False,
                                                                 num_workers=0,
                                                                 pin_memory=True,
                                                                 sampler=val_sampler,
                                                                 drop_last=True,
                                                                 prefetch_factor=2)
        else:
            self.training_loader = DataLoader(training_data, batch_size=config_training['batch_size'], shuffle=True,
                                              num_workers=config_training['num_workers'])
            self.validation_loader = DataLoader(validation_data, batch_size=config_training['batch_size'], shuffle=True,
                                                pin_memory=True, drop_last=True)

        """ Optimizer """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config_training['lr'] * config_training['batch_size'] / 256,
                                           betas=(0.9, 0.95),
                                           weight_decay=config_training["weight_decay"])
        lr_func = lambda epoch: min((epoch + 1) / (config_training["warmup_epoch"] + 1e-8),
                                    0.5 * (math.cos(epoch / config_training["total_epoch"] * math.pi) + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func, verbose=True)

        """ Loss """
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()
        self.h5_bool = training_data.h5_path is not None
        self.tube_bool = tube_bool
        self.multigpu_bool = multigpu_bool

        """ Follow """
        if follow:
            self.follow = Follow("specmae", dir_save=r"checkpoint", multigpu_bool=multigpu_bool,
                                 description=description)

    @staticmethod
    def to_tube(input, size_patch=19, depth_t=5):
        c1 = int(input.shape[-1] / size_patch)
        t1 = input.shape[1] // depth_t
        input = rearrange(input, 'b (t1 t2) (c1 l1) -> b (t1 c1) (l1 t2)', t1=t1, t2=depth_t, c1=c1, l1=size_patch)
        return input

    @staticmethod
    def inverse_tuple(input, size_patch=19, depth_t=5):
        t1 = input.shape[1] // 19
        input = rearrange(input, 'b (t1 c1) (l1 t2) -> b (t1 t2) (c1 l1)', t1=t1, t2=depth_t, c1=19, l1=size_patch)
        return input

    def one_epoch(self):
        self.model.train()
        losses = []
        for img in tqdm(iter(self.training_loader)):
            self.optimizer.zero_grad()
            self.step_count += 1
            img = img.to(self.device)
            if self.h5_bool:
                spec = img
            if self.tube_bool:
                spec = self.to_tube(spec, size_patch=27, depth_t=10)
            predicted_spec, mask = self.model(spec)
            loss = torch.mean((predicted_spec - spec) ** 2 * mask) / self.model.mask_ratio
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    def fit(self):
        for e in range(self.load_epoch, self.config_training["total_epoch"]):
            if self.multigpu_bool:
                self.training_loader.sampler.set_epoch(e)
                self.validation_loader.sampler.set_epoch(e)
            losses = self.one_epoch()
            losses_val = self.eval()
            self.lr_scheduler.step()
            avg_loss_train = sum(losses) / len(losses)
            avg_loss_val = sum(losses_val) / len(losses_val)
            self.parameters = dict(model=self.model.state_dict(),
                                   optimizer=self.optimizer.state_dict(),
                                   scheduler=self.lr_scheduler.state_dict(),
                                   epoch=e,
                                   loss=avg_loss_train)
            print(
                f'In epoch {e}, average traning loss is {avg_loss_train}. and average validation loss is {avg_loss_val}')
            self.follow(epoch=e, loss_train=avg_loss_train, loss_validation=avg_loss_val, parameters=self.parameters)

    def plot_train(self):
        pass

    def eval(self):
        self.model.eval()
        losses = []
        for img in tqdm(iter(self.validation_loader)):
            img = img.to(self.device)
            if self.h5_bool:
                spec = img
            if self.tube_bool:
                spec = self.to_tube(spec, size_patch=27, depth_t=10)
            predicted_spec, mask = self.model(spec)
            loss = torch.mean((predicted_spec - spec) ** 2 * mask) / self.model.mask_ratio
            # loss = torch.mean((predicted_spec[mask == 1] - spec[mask == 1]) ** 2)
            losses.append(loss.item())
        predicted_spec = (predicted_spec * mask + (spec * (~mask.to(torch.bool))).type(torch.float)).type(torch.float)
        spec_mask = (spec * (~mask.to(torch.bool))).type(torch.float)
        if self.tube_bool:
            spec = self.inverse_tuple(spec, size_patch=27, depth_t=10)
            predicted_spec = self.inverse_tuple(predicted_spec, size_patch=27, depth_t=10)
            spec_mask = self.inverse_tuple(spec_mask, size_patch=27, depth_t=10)
        self.save_wav(spec, save=f"{self.follow.path_samples}/original.wav")
        self.save_wav(predicted_spec, save=f"{self.follow.path_samples}/reconstructed.wav")
        self.save_wav(spec_mask, save=f"{self.follow.path_samples}/masked.wav")
        return losses

    @staticmethod
    def save_wav(audio, save: str = None):
        audio = np.sqrt(np.abs(torch.transpose(audio[0], 0, 1).cpu().detach().numpy()))
        plot_spectrogram(audio, show=False, save=f"{save}.png")
        plot_spectrogram(audio, show=False, save=f"{save}.svg")
        # np.save(file=save, arr=np.sqrt(torch.transpose(audio.squeeze(1), 0, 1).cpu().detach().numpy()))
        signal = griffin_lim(audio)
        write(save, 16000, signal)

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        if self.multigpu_bool:
            self.model.module.load_state_dict(checkpoint['model'])  # load checkpoint for multi-GPU
        else:
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.load_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")


def decode_vqvae(
        self,
        image_embeds
):
    b, n, d = image_embeds.shape
    h = w = int(sqrt(n))
    image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
    images = self.vqvae.decoder(image_embeds.type(torch.FloatTensor).to("cuda"))
    return images
