from matplotlib import tri
from sklearn import metrics
import torch
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torchvision.transforms import v2 as T
from torchvision.io import decode_image

import matplotlib.pyplot as plt
from PIL import Image
import os
import warnings
import time


warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)



from utils.models import CalcMetrics, save_model, load_model



label_map = {0:"angry", 1:"disgust", 2:"fear",
            3:"happy", 4:"sad", 5:"surprise", 6: "neutral"}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def train_epoch(model, optimizer, criteria, dataloader, epoch=None):
    global device
    global metrics
    model.train()
    num = 0

    start = time.time()
    size_dataloader = len(dataloader)
    for inputs, labels  in dataloader:
        num+=1
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criteria(outputs, labels)

        loss.backward()
        optimizer.step()

        metrics.update(outputs, labels, loss=loss.item())

        if num%100 == 0:
            print(f"[{num}/{size_dataloader}] loss = {loss.item():.4f}", end = "  | ")
            metrics.compute_last(is_print=True)

            print(f"Time: {time.time() - start:.4f}s")
            start = time.time()


    print(f"Epo {epoch}:", end=' ')
    metrics.compute_last(num=150, is_test=False)
    return model


def test_epoch(model, optimizer, criteria, dataloader, epoch=None, to_save=False):
    global metrics
    global device
    model.eval()
    num = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels  in dataloader:
            num+=1
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criteria(outputs, labels)

            metrics.update(outputs=outputs, labels=labels, loss=loss.item(), is_test=True)


            preds = torch.argmax(outputs, dim=1)

            y_true.append(labels)
            y_pred.append(preds)

    if to_save:
        metrics.save_results(is_test=True, epoch=epoch)

    print("")
    print(f"Epo test  {epoch}:", end=' ')
    metrics.compute_last(num=445, is_test=True)
    print("")

    from sklearn.metrics import classification_report
    
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    target_names = [value for key, value in label_map.items()]
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        zero_division=np.nan
    ))
    print("")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    if to_save:
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(xticks_rotation='vertical')
   
    return model


def get_param_groups(
    model, 
    lr_rnn=1e-5, wd_rnn=1e-6, 
    lr_clf=1e-4, wd_clf=1e-5, 
    lr_atten=1e-5, lr_layer4=1e-6, wd_layer4=1e-7
):
    decay = []
    no_decay = []
    for name, param in model.rnn.named_parameters():
        if 'bias' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    classifier_params = list(model.clasifier.parameters())
    attention_params = list(model.attention_layer.parameters())
    layer4_params = list(model.resnet.features[7].parameters())

    return [
        {"params": decay, "lr": lr_rnn, "weight_decay": wd_rnn},
        {"params": no_decay, "lr": lr_rnn, "weight_decay": 0.0},
        {'params': attention_params, 'lr': lr_atten, 'weight_decay': 1e-7},
        {"params": classifier_params, "lr": lr_clf, "weight_decay": wd_clf},
        {"params": layer4_params, "lr": lr_layer4, "weight_decay": wd_layer4}
    ]

def adding_params_optimizaer(model):
    resnet_params = list(model.reset.parameters())

from utils.models import CustomCNN, CustomCNN_drop_025, Resnet_Custom_48, Resnet_RNN_Custom
from utils.datasets import FER2013_dataset_image, CAER_S_dataset_video
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


import time




import optuna
from torch import nn
import numpy as np

metrics = CalcMetrics(model_name="optuna_trial")
transforms = T.Compose([
    T.Grayscale(),
    T.Resize(94),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = T.Compose([
    T.Grayscale(),
    T.Resize(94),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

caer_s_dataset_train = CAER_S_dataset_video(transform=transforms)

targets = [int(label) for label in caer_s_dataset_train.labels]


class_counts = np.bincount(targets)

print(class_counts)
print(label_map)


class_weights = 1.0 / class_counts
sample_weights = class_weights[targets]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

dataloader_train = DataLoader(caer_s_dataset_train, batch_size=16, num_workers=0, sampler=sampler)

caer_s_dataset_test = CAER_S_dataset_video(transform=test_transforms, is_test=True)
dataloader_test = DataLoader(caer_s_dataset_test, batch_size=1, num_workers=0, shuffle=True)

def objective(trial):
    global device, dataloader_train, dataloader_test, label_map, metrics

    

    # --- Hyperparameters to search ---
    lr_layer4 = trial.suggest_loguniform("lr_layer4", 5e-8, 1e-4)
    wd_layer4 = trial.suggest_loguniform("wd_layer4", 1e-8, 1e-5)
    lr_rnn = trial.suggest_loguniform("lr_rnn", 1e-6, 1e-3)
    lr_clf = trial.suggest_loguniform("lr_clf", 1e-6, 1e-3)
    lr_atten = trial.suggest_loguniform("lr_clf", 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-4)
    rnn_hidden = trial.suggest_categorical("rnn_hidden", [128, 256])
    #optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    #scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "multistep", "plateau", None])
    
    # --- Model ---
    model_photo = Resnet_Custom_48(output_shape=8)
    model_photo, _, _, _, _ = load_model(model_photo, 39, 0, "Resnet14_AffectNet", to_load_optim=False)
    model = Resnet_RNN_Custom(model=model_photo, embedding_dim=2048, rnn_hidden=rnn_hidden, mode='gru')
    model.to(device)
    
    # --- Optimizer ---
    # if optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(
        get_param_groups(
            model, 
            lr_rnn=lr_rnn, wd_rnn=weight_decay, 
            lr_clf=lr_clf, wd_clf=weight_decay, 
            lr_atten=lr_atten, lr_layer4=lr_layer4, wd_layer4=wd_layer4
        )
    )
    # else:  # SGD
    #     optimizer = torch.optim.SGD(
    #         get_param_groups(
    #             model, 
    #             lr_rnn=lr, wd_rnn=weight_decay, 
    #             lr_clf=lr * 10, wd_clf=weight_decay
    #         ),
    #         momentum=0.9,
    #         nesterov=True
    #     )

    # --- Scheduler ---
    # if scheduler_name == "cosine":
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-7)
    # elif scheduler_name == "multistep":
    #     scheduler = MultiStepLR(optimizer, milestones=[2, 7], gamma=0.3)
    # elif scheduler_name == "plateau":
    #     scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    # else:
    #     scheduler = None

    # --- Loss ---
    criteria = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Metrics ---
    metrics.reset()
    
    # --- Training Loop (short, e.g. 5 epochs) ---
    for epoch in range(1, 5):  
        model = train_epoch(model, optimizer, criteria, dataloader_train, epoch)
        model = test_epoch(model, optimizer, criteria, dataloader_test, epoch)
        
        # use last computed test metrics
        results = metrics.compute_last(num=445, is_test=True, is_print=False)
        accuracy = results["accuracy"]
        
        # if epoch == 3:
            # optimizer.add_param_group()

        # step scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(accuracy)
        elif isinstance(scheduler, MultiStepLR):
            scheduler.step(epoch)
        elif scheduler is not None:
            scheduler.step()

        # report to optuna
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # maximize F1-macro
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  accuracy: {trial.value:.4f}")
    print("  Params:", trial.params)
