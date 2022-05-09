import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import torchvision.models as models

from pathlib import Path
from PIL import Image


def glob_file(file_holder, file_type):
    '''
    collect all files with file_type extension in file_holder 
    '''
    all_file = []
    for root, dirs, files in os.walk(file_holder):
        for file in files:
            if os.path.splitext(file)[1] == file_type:
                all_file.append(os.path.join(root, file))
    all_file.sort()
    return all_file


def data_loader(path, pic_size=224):
    input_image = Image.open(path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(pic_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def get_data(path_list, pic_size=224):
    data_list = []
    for path in path_list:
        data = data_loader(path, pic_size)
        data_list.append(data)
    return data_list


def get_outputs(data_list, model, type='raw'):
    model.eval()
    results = []
    for data in data_list:
        if torch.cuda.is_available():
            data = data.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(data)

        if type == 'raw':
            results.append(output[0])
        elif type == 'prob':
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            results.append(probabilities)
    return results


def get_class_score(outputs, class_idx):
    scores = []
    for output in outputs:
        scores.append(output[class_idx])
    return scores

def get_topk_class(prob, k=5):
    with open("data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    results = []
    topk_prob, topk_catid = torch.topk(prob, k)
    for i in range(k):
        results.append((categories[topk_catid[i]], topk_prob[i].item()))
    return results