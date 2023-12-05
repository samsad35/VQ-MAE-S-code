from vqmae import MAE, SpeechVQVAE, Classifier_Train, EvaluationDatasetSpeakerDependent, h5_creation, size_model
import hydra
from omegaconf import DictConfig
import os

# ---------------------------------------------------------------------------------------------
root = r"D:\These\data\Audio\RAVDESS"
dataset_name = "ravdess"
h5_path = r"H5/ravdess.hdf5"
mae_path = r"checkpoint/RSMAE/2023-2-22/12-45"
# ---------------------------------------------------------------------------------------------


@hydra.main(config_path=f"{mae_path}/config_mae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """

    data_train = EvaluationDatasetSpeakerDependent(root=root,
                                                   ratio_train=80,
                                                   train=True,
                                                   frames_per_clip=200,
                                                   dataset=dataset_name,
                                                   h5_path=h5_path
                                                   )

    data_validation = EvaluationDatasetSpeakerDependent(root=root,
                                                        train=False,
                                                        ratio_train=80,
                                                        frames_per_clip=200,
                                                        dataset=dataset_name,
                                                        h5_path=h5_path
                                                        )

    """ VQVAE """
    vqvae = SpeechVQVAE(**cfg.vqvae)
    vqvae.load(path_model=r"checkpoint/SPEECH_VQVAE/2022-12-27/21-42/model_checkpoint")

    """ MAE """
    mae = MAE(**cfg.model, trainable_position=True)
    mae.load(path_model=f"{mae_path}//model")
    size_model(mae, "mae")
    # mae = mae.requires_grad_(False)

    """ Training """
    pretrain_classifier = Classifier_Train(mae,
                                           vqvae,
                                           data_train,
                                           data_validation,
                                           config_training=cfg.train, follow=True, query2emo=False)
    # pretrain_classifier.load(path="checkpoint/CLASSIFIER/2023-1-23/10-31/model_checkpoint")
    accuracy, f1 = pretrain_classifier.fit()

    print("-" * 50)
    print(f"Final accuracy: {accuracy}")
    print(f"Final F1: {f1}")


if __name__ == '__main__':
    main()
