from cv2 import transform
import torch
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torchvision.transforms import v2 as T
from torchvision.io import decode_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


from PIL import Image
import os

label_map = {0:"angry", 1:"disgust", 2:"fear",
            3:"happy", 4:"sad", 5:"surprise", 6: "neutral"}



class FER2013_dataset_tensor(Dataset):
    def __init__(self, path_to_tensors="datasets/fer2013_tensors", is_test=False):
        self.tensor_paths = []
        self.labels = []

        if is_test:
            path_to_tensors = os.path.join(path_to_tensors, 'test')
        else:
            path_to_tensors = os.path.join(path_to_tensors, 'train')

        for emotion in os.listdir(path_to_tensors):
            label = [key for key, value in label_map.items() if value == emotion][0]
            emotion_folder = os.path.join(path_to_tensors, emotion)
            for tensor_file in os.listdir(emotion_folder):
                if tensor_file.endswith('.pt'):
                    full_path = os.path.join(emotion_folder, tensor_file)
                    self.tensor_paths.append(full_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor = torch.load(self.tensor_paths[idx], map_location='cpu')
        label = self.labels[idx]
        return tensor, label
    



import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class FER2013_dataset_image(Dataset):
    def __init__(self, path_to_img="datasets\\fer2013", is_test=False, transform=None, cache_to_ram=True):
        self.images = []
        self.labels = []
        self.transform = transform
        self.cache_to_ram = cache_to_ram

        if is_test:
            path_to_img = os.path.join(path_to_img, 'test')
        else:
            path_to_img = os.path.join(path_to_img, 'train')

        for emotion in os.listdir(path_to_img):
            emotion_label = [int(key) for key, value in label_map.items() if value == emotion][0]
            path_for_emotion = os.path.join(path_to_img, emotion)
            for image_name in os.listdir(path_for_emotion):
                image_path = os.path.join(path_for_emotion, image_name)
                
                if cache_to_ram:
                    img = Image.open(image_path).convert("L")
                    img_tensor = torch.from_numpy(np.array(img))  # shape (H, W), dtype uint8
                    self.images.append(img_tensor)
                else:
                    self.images.append(image_path)  # just save path

                self.labels.append(emotion_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # If we cached tensors, convert to PIL on the fly
        if self.cache_to_ram:
            img = Image.fromarray(img.numpy())  # Back to PIL

        else:
            img = Image.open(img).convert("L")  # read from disk

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label





class CAER_S_dataset_video(Dataset):

    def __init__(self, path_to_dir='datasets\\CAER_S_splited', is_test=False, transform=None):
        self.videos = []
        self.labels = []
        self.transform = transform

        if is_test:
            categories = ['validation']
        else:
            categories = ['test', 'train']
        
        # self.broken = []
        # with open('video_broken.txt', 'r') as file:
            # for n, line in enumerate(file, 1):
            #     line = line.rstrip('\n')
            #     self.broken.append(line)

        for category in categories:
            path_to_category = os.path.join(path_to_dir, category)

            emotions = os.listdir(path_to_category)

            for emotion in emotions:
                emotion_label = [key for key, value in label_map.items() if value == emotion.lower()][0]
                path_to_emotion = os.path.join(path_to_category, emotion)

                for cur_video in os.listdir(path_to_emotion):
                    # if cur_video in self.broken:
                    #     continue

                    self.videos.append(os.path.join(path_to_emotion, cur_video))
                    self.labels.append(emotion_label)
    

    def __len__(self):
        return len(self.labels)
    

    def sample_frames(self, frames, num_frames=24):
        total = len(frames)
        idxs = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = [frames[i] for i in idxs]
        return frames

    def sample_frames_2(self, frames, num_frames=24):
        total = len(frames)
        if total >= num_frames:
            idxs = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = [frames[i] for i in idxs]
        else:
            # Repeat last frame to pad
            pad = [frames[-1]] * (num_frames - total)
            frames = frames + pad
        return frames

    

    def __getitem__(self, idx):
        import cv2

        video = self.videos[idx]
        label = self.labels[idx]
        video_frames = []
        cap = cv2.VideoCapture(video)
        
        if not cap.isOpened(): # if can't read file
            with open('video_broken.txt', 'a') as file:
                file.write(f'{video}\n')
            raise ValueError(f'Cant open video {video}')

        while True:
            ret, frame = cap.read()

            if not ret:     # end of video
                break

            if self.transform:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
            else:
                raise ValueError("cant transform CAER-S dataset")

            video_frames.append(frame)
        
        cap.release()

        if len(video_frames) == 0:
            with open('video_broken.txt', 'a') as file:
                file.write(f'{video}\n')
            raise ValueError(f"No frames read from {video}")

        video_frames = self.sample_frames(video_frames)

        video_tensor = torch.stack(video_frames)

        return video_tensor, label
    





labels_names =   [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
  ]

class AffectNet_dataset(Dataset):
    def __init__(self, path_to_dict="datasets\\affectnet-yolo-format", is_test=False, transform=None, cache_to_ram=True):
        self.images = []
        self.labels = []
        self.transform = transform
        self.cache_to_ram = cache_to_ram

        if is_test:
            path_to_dict = os.path.join(path_to_dict, 'test')
        else:
            path_to_dict = os.path.join(path_to_dict, 'train')

        path_to_imgs = os.path.join(path_to_dict, 'images')
        path_to_lbls = os.path.join(path_to_dict, 'labels')
        for image_name in os.listdir(path_to_imgs):
            try:
                image_path = os.path.join(path_to_imgs, image_name)

                if cache_to_ram:
                    img = Image.open(image_path).convert("L")
                    img_tensor = torch.from_numpy(np.array(img))
                    self.images.append(img_tensor)
                else:
                    self.images.append(image_path)

                label_path = os.path.join(path_to_lbls, image_name[:-3])
                with open(label_path+'txt', 'r') as file:
                    line = file.readline()
                    digit = int(line[0])
                self.labels.append(digit)

            except Exception as e:
                print(f"Error in {self.__class__.__name__} - {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # If we cached tensors, convert to PIL on the fly
        if self.cache_to_ram:
            img = Image.fromarray(img.numpy())  # Back to PIL

        else:
            img = Image.open(img).convert("L")  # read from disk

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

