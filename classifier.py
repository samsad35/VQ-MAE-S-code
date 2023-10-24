from vqmae import MAE, SpeechVQVAE, Classifier_Train, EvaluationDataset, h5_creation, size_model
import hydra
from omegaconf import DictConfig
import os
import numpy as np
from sklearn.utils import shuffle
# ---------------------------------------------------------------------------------------------
Total_folds = 5
root = r"D:\These\data\Audio\RAVDESS"
dataset_name = "ravdess"
h5_path = r"H5/ravdess.hdf5"
mae_path = r"checkpoint/RSMAE/2023-2-22/12-45"
# ---------------------------------------------------------------------------------------------


def fold_creation(list_id, num_fold, k=5):
    length = len(list_id)
    size_fold = length // k
    return list_id[size_fold * num_fold:size_fold * num_fold + size_fold]


@hydra.main(config_path=f"{mae_path}/config_mae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    dataset = EvaluationDataset(root=root,
                                speaker_retain_test=[],
                                frames_per_clip=200,
                                dataset=dataset_name,
                                )
    # h5_creation(vqvae, dataset=data_train, dir_save=r"H5/emovo.hdf5")
    all_id = shuffle(np.unique(np.array(dataset.table["id"])))
    accuracy_epoch = []
    f1_epoch = []

    for num_fold in range(Total_folds):
        print(f"Fold number: {num_fold + 1}/{Total_folds}")
        speaker_retain_test = fold_creation(list(all_id), num_fold=num_fold, k=Total_folds)
        data_train = EvaluationDataset(root=root,
                                       speaker_retain_test=speaker_retain_test,
                                       train=True,
                                       frames_per_clip=200,
                                       dataset=dataset_name,
                                       h5_path=h5_path
                                       )

        data_validation = EvaluationDataset(root=root,
                                            speaker_retain_test=speaker_retain_test,
                                            train=False,
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
        accuracy_epoch.append(accuracy)
        f1_epoch.append(f1)

        print("-" * 50)
    print(f"Accuracy final: {np.mean(accuracy_epoch)}")
    print(f"F1 final: {np.mean(f1_epoch)}")


if __name__ == '__main__':
    main()
