import torch
from PIL import Image
from clip import CLIP
from utils.dataloader import ClipDataset2, dataset_collate2
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import dataloader,my_dataset
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import font_manager
from utils.dataloader import ClipDataset2
import os
def get_code( data_loader, length: int):
    encoder_time = 0
    valP1 = []
    valP2 = []
    
    for i, (index, image, text, label) in enumerate(tqdm(data_loader)):
        start_encoder_time = time.time()

        image = image.to(0, non_blocking=True)
        text = text.to(0, non_blocking=True)

        image_hash = torch.sign(image)
        text_hash = torch.sign(text)  

        encoder_time = time.time() - start_encoder_time
        
        valP1.append(image_hash.cpu()) 
        valP2.append(text_hash.cpu())   

    img_buffer = torch.cat(valP1, dim=0)  
    text_buffer = torch.cat(valP2, dim=0)  
    
    return img_buffer, text_buffer, encoder_time


datasets_path               = "/data/datasets/zhuhaoran/"
datasets_train_json_path      = "datasets/train.json"
datasets_data_json_path      = "datasets/dataset.json"
datasets_val_json_path      = "datasets/quiry.json"
batch_size                  = 128
num_workers                 = 4


modelclip = CLIP()
val_lines = json.load(open(datasets_data_json_path, mode='r', encoding='utf-8'))
num_val = len(val_lines)

val_dataset = ClipDataset2([modelclip.config['input_resolution'], modelclip.config['input_resolution']], val_lines, datasets_path, random=False)
gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                        drop_last=False, collate_fn=dataset_collate2, sampler=None)

lda_features = []
ida_features = []
tda_features = []
for iteration, batch in tqdm(enumerate(gen_val)):
    label, images, texts = batch
    lda_features.append(label)

    with torch.no_grad():
        if modelclip.cuda:
            images = images.cuda()

        images_feature,_ = modelclip.detect_image_for_eval(images, texts=None)

        ida_features.append(images_feature)

texts = gen_val.dataset.text
num_text = len(texts)
for i in tqdm(range(0, num_text, batch_size)):
    text = texts[i: min(num_text, i + batch_size)]
    with torch.no_grad():
        _, texts_feature = modelclip.detect_image_for_eval(images=None, texts=text)
        tda_features.append(texts_feature)

lda_features = sum(lda_features, [])
ida_features = torch.cat(ida_features, 0)
tda_features = torch.cat(tda_features, 0)

lda_features_np = lda_features  
ida_features_np = ida_features.cpu().numpy().astype('float64') 
tda_features_np = tda_features.cpu().numpy().astype('float64')
import numpy as np

features_dict = {
    'ida_features': ida_features_np,
    'tda_features': tda_features_np,
    'lda_features': lda_features_np
}

np.save('features_datasets.npy', features_dict)
