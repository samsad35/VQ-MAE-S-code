from vqmae import MAE, MAE_Train, SpeechVQVAE, VoxcelebSequential
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config_mae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    data_train = VoxcelebSequential(root=r"D:\These\data\Audio-Visual\voxceleb\train",
                                    h5_path=r"path-to-h5-train",
                                    frames_per_clip=200,
                                    train=True
                                    )

    data_validation = VoxcelebSequential(root=r"D:\These\data\Audio-Visual\voxceleb\test\video",
                                         h5_path=r"path-to-h5-validation",
                                         frames_per_clip=200
                                         )
    """ VQVAE """
    vqvae = SpeechVQVAE(**cfg.vqvae)
    vqvae.load(path_model=r"checkpoint/SPEECH_VQVAE/2022-12-27/21-42/model_checkpoint")

    """ MAE """
    mae = MAE(**cfg.model,
              vqvae_embedding=None,
              masking="random",
              trainable_position=True)  # ["random", "horizontal", "vertical", "mosaic"]

    """ Training """
    description = dict(encoder_depth=6, decoder_depth=4, ratio=0.50, masking="random", trainable_position=True)
    pretrain_vqvae = MAE_Train(mae,
                               vqvae,
                               data_train,
                               data_validation,
                               config_training=cfg.train,
                               tube_bool=True,
                               follow=True,
                               multigpu_bool=True,
                               description=description)
    # pretrain_vqvae.load(path="checkpoint/RSMAE/2023-2-1/11-4/model_checkpoint")
    pretrain_vqvae.fit()


if __name__ == '__main__':
    main()
