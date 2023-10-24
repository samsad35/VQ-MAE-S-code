
# A Vector Quantized Masked AutoEncoder for speech emotion recognition
[![Generic badge](https://img.shields.io/badge/<STATUS>-<in_progress>-<COLOR>.svg)](https://github.com/samsad35/VQ-MAE-Speech-code)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://samsad35.github.io/VQ-MAE-Speech/)
[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://test.pypi.org/project/)


![image](images/overview.svg)


## Abstract
<center>

[qualitative results](https://samsad35.github.io/VQ-MAE-Speech/#:~:text=as%20input%20representations.-,Qualitative%20Results,-Back) |  [paper](https://samsad35.github.io/VQ-MAE-Speech/)

</center>

Recent years have seen remarkable progress in speech emotion recognition (SER), thanks to advances in deep learning techniques. However, the limited availability of labeled data remains a significant challenge in the field. Self-supervised learning has recently emerged as a promising solution to address this challenge. In this paper, we propose the vector quantized masked autoencoder for speech (VQ-MAE-S), a self-supervised model that is fine-tuned to recognize emotions from speech signals. The VQ-MAE-S model is based on a masked autoencoder (MAE) that operates in the discrete latent space of a vector quantized variational autoencoder. Experimental results show that the proposed VQ-MAE-S model, pre-trained on the VoxCeleb2 dataset and fine-tuned on emotional speech data, outperforms existing MAE methods that rely on speech spectrogram representations as input.


## Setup 
- [ ] Pypi: (Soon) 

[comment]: <> (  - ``````)
- [x] Install the package locally (for use on your system):  
  - In VQ-MAE-speech directoy: ```pip install -e .```
- [x] Virtual Environment: 
  - ```conda create -n vq_mae_s python=3.8```
  - ```conda activate vq_mae_s```
  - In VQ-MAE-speech directoy: ```pip install -r requirements.txt```

## Usage
* To do:
  * [x] Pre-train Speech VQ-VAE
  * [X] Pre-train VQ-MAE-Speech
  * [X] Fine-tuning and classification
### Pre-train Speech VQ-VAE
```python
from rSMAE import SpeechVQVAE, Speech_VQVAE_Train, VoxcelebSequential
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config_vqvae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    data_train = VoxcelebSequential(root=r"D:\These\data\Audio-Visual\voxceleb\test\video",
                                    h5_path=r"E:\H5\modality_spectrogram_test.hdf5",
                                    frames_per_clip=200,
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

```
- You can download our pre-trained speech VQ-VAE [following link]().
### Pre-train VQ-MAE-Speech
```python
from rSMAE import MAE, MAE_Train, SpeechVQVAE, VoxcelebSequential
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="config_mae", config_name="config")  # You change the hyperparameter of VQ-MAE-Speech in the config-mae
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    data_train = VoxcelebSequential(root=r"path_voxceleb_train",
                                    h5_path=r"path_hdf5_train",  # for speed up training
                                    frames_per_clip=200,  #  sequence max 
                                    train=True
                                    )

    data_validation = VoxcelebSequential(root=r"path_voxceleb_validation",
                                         h5_path=r"path_hdf5_validation",  # for speed up training
                                         frames_per_clip=200
                                         )
    """ VQVAE """
    vqvae = SpeechVQVAE(**cfg.vqvae)  
    vqvae.load(path_model=r"checkpoint/SPEECH_VQVAE/2022-12-27/21-42/model_checkpoint")

    """ MAE """
    mae = MAE(**cfg.model,
              vqvae_embedding=None,
              masking="random",  # ["random", "horizontal", "vertical"]
              trainable_position=True)  

    """ Training """
    description = dict(encoder_depth=6, decoder_depth=4, ratio=0.50, masking="random", trainable_position=True)

    pretrain_vqvae = MAE_Train(mae,
                               vqvae,
                               data_train,
                               data_validation,
                               config_training=cfg.train,
                               tube_bool=True,  # if true: patch-based masking, if false: frame-based masking
                               follow=True,  # For tracking 
                               description=description)  # Add additional information about the train
    # pretrain_vqvae.load(path="path/model_checkpoint")  # If you would to continue the training
    pretrain_vqvae.fit()


if __name__ == '__main__':
    main()
```


## Pretrained models
| Model         	| Masking strategy    	| Masking ratio (%)                	|
|---------------	|---------------------	|------------------------	|
| VQ-MAE-Speech 	| Patch-based masking 	| [50]() - [60]() - [70]() - [80]() - [90]() 	|
| VQ-MAE-Speech 	| Frame-based masking 	| [50]() - [60]() - [70]() - [80]() - [90]() 	|

| Model         	| Encoder depth    	| 
|---------------	|---------------------	|
| VQ-MAE-Speech 	| [6]() - [12]() - [16]() - [20]() 	|

```

## License
GNU Affero General Public License (version 3), see LICENSE.txt.