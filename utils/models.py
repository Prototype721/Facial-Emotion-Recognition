import torch
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torchvision.transforms import v2 as T
from torchvision.io import decode_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


from PIL import Image
import os
import numpy as np

import time

label_map = {0:"angry", 1:"disgust", 2:"fear",
            3:"happy", 4:"sad", 5:"surprise", 6: "neutral"}



"""
--------------------------------------------CalcMetrics----------------------------
"""



class CalcMetrics:
    def __init__(self, model_name=None):
        self.outputs_train_all = []
        self.labels_train_all = []
        self.loss_train_all = []
        self.outputs_test_all = []
        self.labels_test_all = []
        self.loss_test_all = []
        self.model_name = model_name


    def reset(self):
        self.outputs_train_all = []
        self.labels_train_all = []
        self.loss_train_all = []
        self.outputs_test_all = []
        self.labels_test_all = []
        self.loss_test_all = []

    def reset_test(self):
        self.outputs_test_all = []
        self.labels_test_all = []
        self.loss_test_all = []

    def compute(self, outputs, labels):
        outputs = torch.cat([x.unsqueeze(0) if x.dim() == 1 else x for x in outputs], dim=0)
        labels = torch.cat([x.unsqueeze(0) if x.dim() == 0 else x for x in labels], dim=0)

        preds = torch.argmax(outputs, dim=1)

        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            #"prec_micro": precision_score(labels, preds, average='micro', zero_division=0),
            #"recall_micro": recall_score(labels, preds, average='micro', zero_division=0),
            "f1_micro": f1_score(labels, preds, average='micro', zero_division=0),
            #"prec_macro": precision_score(labels, preds, average='macro', zero_division=0),
            #"recall_macro": recall_score(labels, preds, average='macro', zero_division=0),
            "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
        }

        return metrics


    def get_req_data(self, is_test=False):
        """
        returns outputs_all, labels_all, loss_all
        """
        if is_test:
            outputs_all = self.outputs_test_all
            labels_all = self.labels_test_all
            loss_all = self.loss_test_all
        else:
            outputs_all = self.outputs_train_all
            labels_all = self.labels_train_all  
            loss_all = self.loss_train_all

        return outputs_all, labels_all, loss_all


    def update(self, outputs, labels, loss, is_test=False):
        if is_test:
            self.outputs_test_all.extend(outputs.detach())
            self.labels_test_all.extend(labels.detach())
            self.loss_test_all.append(loss)
        else:
            self.outputs_train_all.extend(outputs.detach())
            self.labels_train_all.extend(labels.detach())
            self.loss_train_all.append(loss)
    

    def compute_last(self, num=16, is_test=False, is_print=True):
        outputs_all, labels_all, _ = self.get_req_data(is_test)
        outputs = outputs_all[-num:]
        labels = labels_all[(-1)*num:]
        metrics = self.compute(outputs, labels)

        if is_print:
            for k, v in metrics.items():
                print(f"{k} = {v:.4f}", end=' | ')

        return metrics


    def compute_all(self, is_test=False, is_print=True):
        if is_test:
            labels = torch.cat([x.unsqueeze(0) if x.dim() == 0 else x for x in self.labels_test_all], dim=0)
        else:
            labels = torch.cat([x.unsqueeze(0) if x.dim() == 0 else x for x in self.labels_train_all], dim=0)
        length = len(labels)

        return self.compute_last(num=length, is_test=is_test, is_print=is_print)


    def save_results(self, is_test=False, epoch=None):

        os.makedirs(f"metrics/{self.model_name}", exist_ok=True)

        outputs_all, labels_all, loss_all = self.get_req_data(is_test)

        outputs_all = [x.cpu() for x in outputs_all]
        labels_all = [x.cpu() for x in labels_all]

        package = {
            "is_test":is_test,
            "epoch":epoch,
            "outputs_all":outputs_all,
            "labels_all":labels_all,
            "loss_all":loss_all
        }
        if is_test:
            test_name = "test"
        else:
            test_name = "train"
        if epoch is not None:
            file_name = f"metrics/{self.model_name}/{self.model_name}_{test_name}_{epoch}.pt"
        else:
            file_name = f"metrics/{self.model_name}/{self.model_name}_{test_name}.pt"
        
        torch.save(package, file_name)

    def load_results(self, is_test=False, epoch=None):

        if is_test:
            test_name = "test"
        else:
            test_name = "train"

        if epoch is not None:
            file_name = f"metrics/{self.model_name}/{self.model_name}_{test_name}_{epoch}.pt"
        else:
            file_name = f"metrics/{self.model_name}/{self.model_name}_{test_name}.pt"

        package = torch.load(file_name, weights_only=False)

        outputs = package["outputs_all"]
        labels = package["labels_all"]
        loss = package["loss_all"]

        if is_test:
            self.outputs_test_all.extend(outputs)
            self.labels_test_all.extend(labels)
            self.loss_test_all.extend(loss)
        else:
            self.outputs_train_all.extend(outputs)
            self.labels_train_all.extend(labels)
            self.loss_train_all.extend(loss)


    def draw_results(self, len_of_window=150):
        """
        A function that goes through every metrics, calculated by sliding window 
        (just given outputs and labels to compute() function) and draws every metrics and loss on pyplot
        
        loss with blue to train and orange to test data
        others metrics every separately (because update works not simmetrically)

        save photos with file_name = f"metrics/{self.model_name}/{self.model_name}_{~metric_name~}_{is_test}.png"
        (for loss just f"metrics/{self.model_name}/{self.model_name}_loss.png") 
        """

        os.makedirs(f"metrics/{self.model_name}", exist_ok=True)

        def sliding_metrics(outputs_all, labels_all, loss_all, is_test):

            metrics_list = []
            for i in range(0, len(outputs_all) - len_of_window - 3, len_of_window//2): # NGL мне влом считать, сломаться не должно :)
                window_outputs = torch.stack([t.detach().cpu() for t in outputs_all[i:i + len_of_window]])
                window_labels = torch.stack([t.detach().cpu() for t in labels_all[i:i + len_of_window]])
                metrics = self.compute(window_outputs, window_labels)
                metrics_list.append(metrics)
            return metrics_list

        # Convert loss lists to numpy arrays for plotting
        loss_train = np.array(self.loss_train_all)
        loss_test = np.array(self.loss_test_all)

        # Smooth losses using sliding window mean
        def smooth_loss(loss):
            return np.convolve(loss, np.ones(len_of_window) / len_of_window, mode='valid')

        # Plot and save loss
        plt.figure()
        if len(loss_test) >= len_of_window and len(loss_train) >= len_of_window:
            smoothed_test = smooth_loss(loss_test)
            smoothed_train = smooth_loss(loss_train)

            # Stretch test loss to match train loss length
            stretched_test = np.interp(
                np.linspace(0, len(smoothed_test) - 1, num=len(smoothed_train)),
                np.arange(len(smoothed_test)),
                smoothed_test
            )

            plt.plot(smoothed_train, label="Train Loss", color='blue')
            plt.plot(stretched_test, label="Test Loss (stretched)", color='red')

        plt.title("Loss over time")
        plt.xlabel("Batch (sliding window)")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"metrics/{self.model_name}/{self.model_name}_loss.png")
        plt.close()

        # Calculate metrics over sliding windows
        metrics_train = sliding_metrics(self.outputs_train_all, self.labels_train_all, self.loss_train_all, is_test=False)
        metrics_test = sliding_metrics(self.outputs_test_all, self.labels_test_all, self.loss_test_all, is_test=True)

        # List of metric names to plot
        metric_names = list(metrics_train[0].keys()) if metrics_train else []

        # Plot each metric separately
        for metric_name in metric_names:
            plt.figure()

            if metrics_train:
                train_values = [m[metric_name] for m in metrics_train]
                plt.plot(train_values, label='Train', color='blue')

            if metrics_test and metrics_train:
                test_values = [m[metric_name] for m in metrics_test]
                # Stretch test metrics to match train metrics length
                stretched_test_values = np.interp(
                    np.linspace(0, len(test_values) - 1, num=len(train_values)),
                    np.arange(len(test_values)),
                    test_values
                )
                plt.plot(stretched_test_values, label='Test (stretched)', color='orange')

            plt.title(f"{metric_name} over time")
            plt.xlabel("Window")
            plt.ylabel(metric_name)
            plt.legend()
            plt.savefig(f"metrics/{self.model_name}/{self.model_name}_{metric_name}.png")
            plt.close()




"""
--------------------------------------------Save and Load----------------------------
"""


def save_model(model, epoch, optimizer, criteria, model_name):
    pack = {
        "model_name": model_name,
        "epoch": epoch,
        "model_weights": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "criteria": criteria
    }
    file_name = f"models/{model_name}_{epoch}.pt"
    torch.save(pack, file_name)

    print(f"Saved {file_name}")
    return model_name, file_name

def load_model(model, epoch, optimizer, model_name, to_load_optim=True):
    """
    returns model, optimizer, pack["criteria"], pack["model_name"], pack["epoch"]
    """
    file_name = f"models/{model_name}_{epoch}.pt"
    pack = torch.load(file_name, weights_only=False)

    model.load_state_dict(pack["model_weights"])
    if to_load_optim:
        optimizer.load_state_dict(pack["optimizer"])

    print(f"Loaded {file_name}")

    return model, optimizer, pack["criteria"], pack["model_name"], pack["epoch"]




















"""
--------------------------------------------Models----------------------------
"""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

def ResNet50_module(is_load=False):
    global device
    
    if is_load:
        model = torch.load("models/resnet50_model.pth", weights_only=False)
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        input_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=input_features, out_features=len(label_map)) # TODO add softmax maybe dropout

    
    return model



def Resnet_Custom_48(output_shape=7, load_path=None, to_freeze=True): # or 96 pixels
    if load_path is None:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # changing input shape fror gray scale
        pretrained_conv1 = model.conv1.weight
        new_conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data = pretrained_conv1.mean(dim=1, keepdim=True)
        model.conv1 = new_conv1

        # change output shape for classes
        model.fc = nn.Linear(model.fc.in_features, output_shape)

        # freeze layers for trans-learn
        if to_freeze:
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.fc.parameters():
                param.requires_grad = True

            for name, param in model.named_parameters():
                if 'conv1' in name:
                    param.requires_grad = True

    else:
        model = torch.load(load_path, weights_only=False)

    return model



class Resnet_Feature_Extractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])  # delete classifier
        self.flatten = nn.Flatten()

    def forward(self, x):  # x: (B, C, H, W)
        x = self.features(x)
        x = self.flatten(x)  # (B, feature_dim)
        return x


class Resnet_RNN_Custom(nn.Module):
    
    def __init__(self, model, embedding_dim=512, rnn_hidden=256, num_classes=7, num_layers=1, mode='lstm'):
        super().__init__()

        self.resnet = Resnet_Feature_Extractor(model)
        if mode == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden, dropout=0.2,
                                num_layers=num_layers, batch_first=True, bidirectional=False)
        elif mode == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=rnn_hidden, dropout=0.2,
                                num_layers=num_layers, batch_first=True, bidirectional=False)
        else:
            raise ValueError(f"Unexpected argument mode - {mode}")

        self.attention_layer = nn.Linear(in_features=rnn_hidden, out_features=1)
        
        self.clasifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden, num_classes)
        )

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.size()
        x_seq = x_seq.view(B * T, C, H, W)
        feats = self.resnet(x_seq)
        feats = feats.view(B, T, -1)  # (B, T, C, H, W)

        rnn_out, _ = self.rnn(feats)
        attention_weights = torch.softmax(self.attention_layer(rnn_out), dim=1)
        rnn_out = torch.sum(rnn_out*attention_weights, dim=1)
        out = self.clasifier(rnn_out)
        # out = self.clasifier(rnn_out[:, -1, :])

        return out







from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomCNN, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.35)


        self.conv_layer3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(128, 256, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=0.35)

        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 48, 48)  # (batch_size, channels, H, W)
            x = self._forward_features(dummy_input)
            flatten_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_size, 1024)
        self.relu1 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)


    def _forward_features(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = self.drop1(x)

        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.batch_norm2(x)
        x = self.max_pool2(x)
        x = self.drop2(x)
        return x
    

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.batch_norm1(out)
        out = self.max_pool1(out)
        out = self.drop1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.batch_norm2(out)
        out = self.max_pool2(out)
        out = self.drop2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.drop3(out)
        out = self.fc2(out)

        return out
    




class CustomCNN_drop_025(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomCNN_drop_025, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.25)


        self.conv_layer3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(128, 256, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=0.25)

        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 48, 48)  # (batch_size, channels, H, W)
            x = self._forward_features(dummy_input)
            flatten_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_size, 1024)
        self.relu1 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)


    def _forward_features(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = self.drop1(x)

        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.batch_norm2(x)
        x = self.max_pool2(x)
        x = self.drop2(x)
        return x
    

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.batch_norm1(out)
        out = self.max_pool1(out)
        out = self.drop1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.batch_norm2(out)
        out = self.max_pool2(out)
        out = self.drop2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.drop3(out)
        out = self.fc2(out)

        return out