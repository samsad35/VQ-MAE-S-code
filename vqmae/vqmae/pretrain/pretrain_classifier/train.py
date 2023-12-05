from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from ...base import Train
from ...data import EvaluationDataset
from torch.utils.data import Dataset
from ...model import MAE, Classifier, SpeechVQVAE, Query2Label
import matplotlib.pyplot as plt
from .follow_up_classifier import Follow
import math
torch.cuda.empty_cache()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from .asymmetric_loss import ASLSingleLabel
from einops import repeat, rearrange
from sklearn.metrics import f1_score


class Classifier_Train(Train):
    def __init__(self, mae: MAE,
                 vqvae: SpeechVQVAE,
                 training_data: Dataset,
                 test_data: Dataset,
                 config_training: dict = None, follow: bool = True,
                 query2emo: bool = False):
        super().__init__()
        self.device = torch.device(config_training['device'])
        """ Model """
        if query2emo:
            self.model = Query2Label(encoder=mae.encoder, num_classes=8)
        else:
            self.model = Classifier(encoder=mae.encoder, num_classes=8)
        self.model.to(self.device)
        self.vqvae = vqvae
        self.vqvae.to(self.device)

        """ Dataloader """
        batch_size = 16
        self.training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True,
                                          num_workers=0)
        self.validation_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                            pin_memory=True)

        """ Optimizer """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config_training['lr']/10.0,
                                           betas=(0.9, 0.95),
                                           weight_decay=config_training["weight_decay"])
        lr_func = lambda epoch: min((epoch + 1) / (40 + 1e-8),
                                    0.5 * (math.cos(epoch / config_training["total_epoch"] * math.pi) + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func, verbose=True)

        """ Loss """
        # weights = training_data.get_weights(num_class=8)
        # self.criterion = torch.nn.CrossEntropyLoss(reduction="mean", weight=weights.to(self.device))
        self.criterion = ASLSingleLabel(reduction="mean")
        self.acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
        self.best_acc = 0.0

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()

        """ Follow """
        if follow:
            self.follow = Follow("classifier", dir_save=r"checkpoint", variable=vars(self.model))

    @staticmethod
    def to_tube(input, size_patch=4, depth_t=5):
        c1 = int(input.shape[-1] / size_patch)
        t1 = input.shape[1] // depth_t
        input = rearrange(input, 'b (t1 t2) (c1 l1) -> b (t1 c1) (l1 t2)', t1=t1, t2=depth_t, c1=c1, l1=size_patch)
        return input

    def one_epoch(self):
        self.model.train()
        losses = []
        acces = []
        for input, label in tqdm(iter(self.training_loader)):
            self.optimizer.zero_grad()
            self.step_count += 1
            input = input.to(self.device)
            label = label.to(self.device)
            # input = self.to_tube(input, depth_t=10, size_patch=4)
            logits = self.model(input)
            loss = self.criterion(logits, label)
            acc = self.acc_fn(logits, label)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            acces.append(acc.item())
        return losses, acces

    def fit(self):
        for e in range(self.config_training["total_epoch"]):
            losses, acces = self.one_epoch()
            losses_test, acces_test, f1_test = self.eval()
            self.lr_scheduler.step()
            avg_loss_train = sum(losses) / len(losses)
            avg_train_acc = sum(acces) / len(acces)
            avg_loss_val = sum(losses_test) / len(losses_test)
            avg_test_acc = sum(acces_test) / len(acces_test)
            self.parameters = dict(model=self.model.state_dict(),
                                   optimizer=self.optimizer.state_dict(),
                                   scheduler=self.lr_scheduler.state_dict(),
                                   epoch=e,
                                   loss=avg_loss_train)
            print(
                f'In epoch {e}, average traning loss is {avg_loss_train:.3f}.'
                f' and average validation loss is {avg_loss_val:.3f}')
            print(
                f'\t - average accuracy is {avg_train_acc:.3f}.'
                f' and average validation accuracy is {avg_test_acc:.3f} and F1 score is  {f1_test:.3f}')
            self.follow(epoch=e,
                        loss_train=avg_train_acc,
                        loss_validation=avg_test_acc,
                        parameters=self.parameters,
                        f1_loss=f1_test)
        return self.follow.best_loss, self.follow.best_f1

    def eval(self):
        self.model.eval()
        losses = []
        acces = []
        y_true = torch.tensor([])
        y_pred = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for input, label in tqdm(iter(self.validation_loader)):
                y_true = torch.cat((y_true, label), dim=0)
                input = input.to(self.device)
                label = label.to(self.device)
                # input = self.to_tube(input, depth_t=10, size_patch=4)
                logits = self.model(input)
                y_pred = torch.cat((y_pred, logits.argmax(dim=-1)), dim=0)
                loss = self.criterion(logits, label)
                acc = self.acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
        # labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        f1 = f1_score(y_true.numpy(), y_pred.cpu().detach().numpy(), average="weighted")
        if sum(acces) / len(acces) > self.best_acc:
            self.best_acc = sum(acces) / len(acces)
            labels = ["W", "L", "E", "A", "F", "T", "N"]
            cm = confusion_matrix(y_true.numpy(), y_pred.cpu().detach().numpy())
            plt.figure(figsize=(15, 15))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot()
            plt.savefig(f'{self.follow.path}/matrix_confusion.png')
            plt.savefig(f'{self.follow.path}/matrix_confusion.svg')
        return losses, acces, f1

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.load_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]")

    def plot_3D(self):
        pca = TSNE(n_components=3)
        features = torch.tensor([]).to(self.device)
        labels = torch.tensor([])
        with torch.no_grad():
            for img, label in tqdm(iter(self.training_loader)):
                labels = torch.cat((labels, label), dim=0)
                img = img.to(self.device)
                # indices = self.vqvae.get_codebook_indices(img)  # .cpu().detach().numpy()
                # indices = rearrange(indices, 'b (h w) -> b h w', h=64, w=64)
                # indices = rearrange(indices, 'b (h c1) (w c2) -> b (h w) (c1 c2)', c1=4, c2=4)
                cls = self.model.get_cls(img)
                features = torch.cat((features, cls), dim=0)
        features = features.cpu().detach().numpy()
        labels = labels.numpy()
        features = pca.fit_transform(features)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # name_labels = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]  # Jaffed dataset
        name_labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        colors = ['red', 'cyan', 'violet', 'pink', 'olive', 'sienna', 'navy']
        for i in range(7):
            indx = labels == i
            data = features[indx]
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors[i], label=name_labels[i])
        plt.title('Matplot 3d scatter plot')
        plt.legend(loc=2)
        plt.show()

