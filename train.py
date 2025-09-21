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



from utils.datasets import FER2013_dataset_tensor
from utils.models import CalcMetrics, ResNet50_module, save_model, load_model



label_map = {0:"angry", 1:"disgust", 2:"fear",
            3:"happy", 4:"sad", 5:"surprise", 6: "neutral"}

affectnet_labels_names =   [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
  ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def train_epoch(model, optimizer, criteria, dataloader, epoch=None):
    global device
    model.train()
    num = 0

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

    #target_names = [value for key, value in label_map.items()]
    target_names = affectnet_labels_names

    print(classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        zero_division=np.nan
    ))
    print("")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # try:
    #     plt.close()
    # except:
    #     pass

    # cm = confusion_matrix(y_true, y_pred)
    # ConfusionMatrixDisplay(cm, display_labels=target_names).plot(xticks_rotation='vertical')
    # plt.show()
   
    return model





def add_param_group(optimizer, model, layer_name, lr=1e-4, weight_decay=1e-5):
    new_params = [
        param for name, param in model.named_parameters()
        if (layer_name in name and param.requires_grad and not any(param is pg_param for pg in optimizer.param_groups for pg_param in pg['params']))
    ]
    if new_params:
        optimizer.add_param_group({
            'params': new_params,
            'lr': lr,
            'weight_decay': weight_decay
        })




from utils.models import CustomCNN, CustomCNN_drop_025, Resnet_Custom_48
from utils.datasets import FER2013_dataset_image, AffectNet_dataset
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.utils.class_weight import compute_class_weight


if __name__ == "__main__":
    torch.cuda.empty_cache()


    model = Resnet_Custom_48(output_shape=8)
    model_name = "Resnet14_AffectNet"


    model.to(device)
    num_epo = 60
    start_epo = 1
    transforms = T.Compose([
        T.Resize(94),
        T.RandomHorizontalFlip(),
        T.RandomRotation(7),
        T.RandomResizedCrop(94, scale=(0.8, 1.0)),
        T.Grayscale(1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transforms = T.Compose([
        T.Resize((94, 94)),
        T.Grayscale(1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    fer_dataset_train = AffectNet_dataset(transform=transforms, cache_to_ram=True)

    targets = fer_dataset_train.labels

    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    dataloader_train = DataLoader(fer_dataset_train, batch_size=16, sampler=sampler)

    fer_dataset_test = AffectNet_dataset(transform=test_transforms, is_test=True)
    dataloader_test = DataLoader(fer_dataset_test, batch_size=16, shuffle=True)

    metrics = CalcMetrics(model_name=model_name)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW([
    {'params': model.conv1.parameters(), 'lr': 5e-5},
    {'params': model.fc.parameters(),    'lr': 1e-3}
    ], weight_decay=1e-4)

    # scheduler = MultiStepLR(optimizer, milestones=[5, 10, 20, 30, 40], gamma=0.4)  # gamma controls decay factor
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True, min_lr=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    model_name, _ = save_model(model, 0, optimizer, criteria, model_name)
    # model, optimizer, criteria, _, _ = load_model(model, start_epo-1, optimizer, model_name)

    # metrics.load_results(epoch=start_epo-1, is_test=False)
    # metrics.load_results(epoch=start_epo-1, is_test=True)
    
    model = test_epoch(model, optimizer, criteria, dataloader_test, 0)
    
    for epoch in range(start_epo, num_epo+1):
        try:
            if epoch == 4:
                for name, param in model.named_parameters():
                    if 'layer1' in name:
                        param.requires_grad = True
                add_param_group(optimizer, model, 'layer1', lr=1e-4)

            if epoch == 8:
                for name, param in model.named_parameters():
                    if 'layer2' in name:
                        param.requires_grad = True
                add_param_group(optimizer, model, 'layer2', lr=1e-4)

            if epoch == 15:
                for name, param in model.named_parameters():
                    if 'layer3' in name:
                        param.requires_grad = True
                add_param_group(optimizer, model, 'layer3', lr=1e-4)

            model = train_epoch(model, optimizer, criteria, dataloader_train, epoch)
            print("")

            if (epoch % 5 == 0 and epoch < 15) or epoch > 15:
                model = test_epoch(model, optimizer, criteria, dataloader_test, epoch, to_save=True)
            else:
                model = test_epoch(model, optimizer, criteria, dataloader_test, epoch)


            #val_loss = np.mean(metrics.loss_test_all)
            #scheduler.step(val_loss)
            scheduler.step(epoch)

            if (epoch % 5 == 0 and epoch < 15) or epoch > 15:
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