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
    lr_rnn=3.99e-04, wd_rnn=5.2e-06, 
    lr_clf=1.4e-04, wd_clf=5.2e-06, 
    lr_atten=1.4e-5, lr_layer4=7.6e-8, wd_layer4=1.9e-07
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
        {'params': attention_params, "lr": lr_atten, 'weight_decay': 1e-7},
        {"params": classifier_params, "lr": lr_clf, "weight_decay": wd_clf},
        {"params": layer4_params, "lr": lr_layer4, "weight_decay": wd_layer4}
    ]


from utils.models import CustomCNN, CustomCNN_drop_025, Resnet_Custom_48, Resnet_RNN_Custom
from utils.datasets import FER2013_dataset_image, CAER_S_dataset_video
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


import time



if __name__ == "__main__":
    torch.cuda.empty_cache()

    start_time = time.time()
    model_photo = Resnet_Custom_48(output_shape=8)
    model_photo_name = "Resnet14_AffectNet"

    model_photo, _, _, _, _ = load_model(model_photo, 39, 0, model_photo_name, to_load_optim=False)

    model = Resnet_RNN_Custom(model=model_photo, embedding_dim=2048, rnn_hidden=128, mode='gru')
    model_name = "VIDEO_GRU_11_affectnet_attention_trial_7"

    model.to(device)
    num_epo = 30
    start_epo = 1
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
    # for name, param in model.resnet.named_parameters():
    #     print(name)
    # exit()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    dataloader_train = DataLoader(caer_s_dataset_train, batch_size=16, num_workers=4, sampler=sampler)

    caer_s_dataset_test = CAER_S_dataset_video(transform=test_transforms, is_test=True)
    dataloader_test = DataLoader(caer_s_dataset_test, batch_size=1, num_workers=0, shuffle=True)

    metrics = CalcMetrics(model_name=model_name)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(get_param_groups(model))

    # # scheduler = MultiStepLR(optimizer, milestones=[5, 10, 20, 30, 40], gamma=0.4)  # gamma controls decay factor
    # #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True, min_lr=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-7)

    # model_name, _ = save_model(model, 0, optimizer, criteria, model_name)
    # model, optimizer, criteria, _, _ = load_model(model, start_epo-1, optimizer, model_name)

    # metrics.load_results(epoch=start_epo-1, is_test=False)
    # metrics.load_results(epoch=start_epo-1, is_test=True)
    
    # model = test_epoch(model, optimizer, criteria, dataloader_test, 0)

    print(f'time:{time.time()-start_time:.4f}')

    for epoch in range(start_epo, num_epo+1):
        try:
            model = train_epoch(model, optimizer, criteria, dataloader_train, epoch)
            print("")

            print(f'time:{time.time()-start_time:.4f}')

            if (epoch % 5 == 0 and epoch < 15) or epoch > 15:
                model = test_epoch(model, optimizer, criteria, dataloader_test, epoch, to_save=True)
            else:
                model = test_epoch(model, optimizer, criteria, dataloader_test, epoch)


            #val_loss = np.mean(metrics.loss_test_all)
            #scheduler.step(val_loss)
            scheduler.step(epoch)

            if (epoch % 2 == 0 and epoch < 4) or epoch > 0:
                model_name, _ = save_model(model, epoch, optimizer, criteria, model_name)
                metrics.save_results(epoch=epoch, is_test=False)


        except KeyboardInterrupt:
            print('Stopping by manual command')
            model = test_epoch(model, optimizer, criteria, dataloader_test, epoch)
            print("Drawing")
            metrics.draw_results()
            print("Results has been drawed")
            exit()
            

    print("Drawing")
    metrics.draw_results()
    print("Results has been drawed")