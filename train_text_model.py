import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,\
    accuracy_score, f1_score, precision_score, recall_score, classification_report
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from tqdm import tqdm

from datasets import ImageFolderWithSubfolders


def eval_model_2clasees(model, valid_dataloader, positive_classes, negative_classes, plot=False):
    with torch.no_grad():
        model.eval()
        result = 0
        n = 0
        val_preds = np.zeros(len(valid_dataloader.dataset))
        val_labels = np.zeros(len(valid_dataloader.dataset))
        start = 0
        class_to_idxs_dict = {}
        for negative_class in negative_classes:
            class_to_idxs_dict[valid_dataloader.dataset.class_to_idx[negative_class]]=0
        for positive_class in positive_classes:
            class_to_idxs_dict[valid_dataloader.dataset.class_to_idx[positive_class]]=1
        for images, labels in valid_dataloader:
            batch_size = images.size(0)
            n += batch_size
            #images = images.cuda()
            pred = F.softmax(model(images))
            prediction = torch.argmax(pred, 1)

            val_preds[start:start + batch_size] = prediction.cpu().numpy()
            val_labels[start:start + batch_size] = labels.numpy()
            start += batch_size

    val_labels = np.vectorize(class_to_idxs_dict.get)(val_labels)
    val_preds = np.vectorize(class_to_idxs_dict.get)(val_preds)
    print('Precision: ', precision_score(val_labels, val_preds, average='weighted'))
    print('Recall: ', recall_score(val_labels, val_preds, average='weighted'))
    print('F1 score: ', f1_score(val_labels, val_preds, average='weighted'))

    if plot:
        cm = confusion_matrix(val_labels, val_preds,
                              labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['nswf','neutral'])
        disp.plot()
        plt.show()

def eval_model(model, valid_dataloader, plot=False):
    with torch.no_grad():
        model.eval()
        result = 0
        n = 0
        val_preds = np.zeros(len(valid_dataloader.dataset))
        val_labels = np.zeros(len(valid_dataloader.dataset))
        start = 0
        for images, labels in valid_dataloader:
            batch_size = images.size(0)
            n += batch_size
            #images = images.cuda()
            pred = F.softmax(model(images))
            prediction = torch.argmax(pred, 1)

            val_preds[start:start + batch_size] = prediction.cpu().numpy()
            val_labels[start:start + batch_size] = labels.numpy()
            start += batch_size

    print('Precision: ', precision_score(val_labels, val_preds, average='weighted'))
    print('Recall: ', recall_score(val_labels, val_preds, average='weighted'))
    print('F1 score: ', f1_score(val_labels, val_preds, average='weighted'))

    if plot:
        cm = confusion_matrix(val_labels, val_preds,
                              labels=list(valid_dataloader.dataset.class_to_idx.values()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=list(valid_dataloader.dataset.class_to_idx.keys()))
        disp.plot(cmap = plt.cm.Blues)
        plt.show()

def get_dataloaders(dataset_dirs):

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = ImageFolderWithSubfolders(root='./data/input/train',
                                              transform=train_transform,
                                              subfolders=dataset_dirs)
    valid_dataset = ImageFolderWithSubfolders(root='./data/input/valid',
                                              transform=valid_transform,
                                              subfolders=dataset_dirs)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

    return train_dataloader, valid_dataloader



def train_model(model, train_dataloader, valid_dataloader, start_epoch=0, num_epochs=5, plot=False):
    #model.cuda()
    optimizer = optim.Adam(params=model.fc.parameters(), lr=0.0001, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(start_epoch,num_epochs):
        print(epoch)
        model.train()
        for images, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            #images = images.cuda()
            #labels = labels.cuda()

            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        eval_model(model, valid_dataloader)
        #torch.save(model, '../checkpoints/text_classifier_{}.pt'.format(epoch+1))
        torch.save(model, '../checkpoints/text_classifier_2cats_{}.pt'.format(epoch+1))
        if plot:
            plt.plot(losses)

    return model

def create_model(num_classes=4):

    model = resnet18(pretrained=True)
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, num_classes)
    return model

if __name__=='__main__':
    dataset_dirs = ['packs', 'graffiti']
    train_dataloader, valid_dataloader = get_dataloaders(dataset_dirs)
    #model = create_model(2)
    model = torch.load(r'../checkpoints/text_classifier_2cats_10.pt')
    #model = train_model(model, train_dataloader, valid_dataloader, start_epoch=0, num_epochs=10)
    model.eval()
    eval_model(model, valid_dataloader, plot=True)
    #eval_model_2clasees(model, valid_dataloader, negative_classes=['graffiti'],
    #                    positive_classes=['packs'], plot=True)