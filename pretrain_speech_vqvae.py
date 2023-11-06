from vqmae import SpeechVQVAE, Speech_VQVAE_Train, VoxcelebSequential
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config_vqvae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    data_train = VoxcelebSequential(root=r"D:\These\data\Audio-Visual\voxceleb\test\video",
                                    h5_path=r"E:\H5\modality_spectrogram_test.hdf5",
                                    frames_per_clip=1,
                                    train=True
                                    )

    """ Model """
    vqvae = SpeechVQVAE(**cfg.model)
    # vqvae.load(path_model=r"checkpoint/VQVAE/2023-1-10/22-36/model_checkpoint")

    """ Training """
    pretrain_vqvae = Speech_VQVAE_Train(vqvae, data_train, data_train, config_training=cfg.train)
    # pretrain_vqvae.load(path=r"checkpoint/VQVAE/2022-12-28/12-7/model_checkpoint")
    pretrain_vqvae.fit()


if __name__ == '__main__':
    main()
