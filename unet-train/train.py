import torch
from configparser import Interpolation
import torchvision.transforms
from preprocess import SegDataset
import torch.utils.data
from model import UNet
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import glob

#Provide each class a weight (Abandon, there is hardly improvment.)
def class_weight(label_path, n_classes) -> torch.tensor:
    label_paths = glob.glob(os.path.join(label_path, '*.png'))

    samples = []
    samples_0 = []
    samples_1 = []
    for label_path in label_paths:
        label = torchvision.io.read_image(path=label_path, mode=torchvision.io.ImageReadMode.GRAY)
        samples.append(label.shape[1] * label.shape[2])
        samples_0.append((label == 0).sum().item())
        samples_1.append((label == 255).sum().item())

    n_samples = float(sum(samples)) #total number of all samples
    n_samples_0 = float(sum(samples_0)) #total number of category 0 of samples
    n_samples_1 = float(sum(samples_1)) #total number of category 1 of samples

    weight_0 = n_samples / (n_classes * n_samples_0)
    weight_1 = n_samples / (n_classes * n_samples_1)

    return torch.tensor([weight_0, weight_1])

#Calculate the metrixs, including precision, recall and f1_score, which evaluate the model's performance
def evaluate(pred_tensor: torch.tensor, target_tensor: torch.tensor, epsilon: float = 1e-6):
    """"
    Calculate metrixs, including precision, recall and f1_score(dice coefficient).
    Args:
        pred_tensor, target_tensor: N * H * W
        epsilon: avoid diving by zero
    Returns:
        dict: {'precision': xxx, 'recall': xxx, 'f1_score': xxx}
    """
    assert pred_tensor.size() == target_tensor.size()
    assert torch.all(torch.logical_or(pred_tensor == 0, pred_tensor == 1))
    assert torch.all(torch.logical_or(target_tensor == 0, target_tensor == 1))

    #计算混淆矩阵
    true_positives = torch.sum((pred_tensor == 1) & (target_tensor == 1)).float().item()
    false_positive = torch.sum((pred_tensor == 1) & (target_tensor == 0)).float().item()
    false_negatives = torch.sum((pred_tensor == 0) & (target_tensor == 1)).float().item()

    #计算Precision、Recall、F1_Score
    precision = (true_positives / max(true_positives + false_positive, epsilon))
    recall = (true_positives / max(true_positives + false_negatives, epsilon))
    f1_score = 2 * precision * recall / max(precision + recall, epsilon)

    return precision, recall, f1_score

def unet_train(
        net,
        device,
        data_path,
        epochs: int,
        batch_size: int,
        lr: float,
        val_percent: float
):

    #1、Divide the dataset into train dataset and validate dataset
    seg_dataset = SegDataset(data_path)
    n_val = int(len(seg_dataset) * val_percent)
    n_train = len(seg_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(seg_dataset, [n_train, n_val],
                                                  generator = torch.Generator().manual_seed(0))

    #2、Create daatset loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    #2、Calculate each class's weights and create loss function
    #class_weights = class_weight(label_path=data_path.replace('image', 'label'), n_classes=2)
    #class_weights = torch.tensor([1.0, 10.0])
    #class_weights=class_weights.to(device=device)
    criterion = nn.CrossEntropyLoss()

    #3、Create optimizer
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)

    #4、Load model to device and begin training
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.to(device=device)
    epoch_plt = []
    loss_plt = []
    precision_plt = []
    recall_plt = []
    f1_score_plt = []
    f1_score_last = 0
    logging.info(f"Starting training...")
    try:
        for epoch in range(epochs):
            #model's mode
            net.train()
            loss_epoch = []
            #train batch one by one
            for image, label in train_loader:
                #load train data to device
                image = image.to(device=device, dtype=torch.float32)
                label = label.squeeze(dim=1)
                label = label.to(device=device, dtype=torch.int64)
                #forward
                pred = net(image)
                #calculate loss
                loss = criterion(pred, label)
                loss_epoch.append(loss.item())
                #backward
                loss.backward()
                #update arguments
                optimizer.step()
                optimizer.zero_grad()
            #Reserve the mean loss for this round
            epoch_plt.append(epoch)
            loss_plt.append(np.mean(loss_epoch))

            #5、Valuate the model's performance
            precision_epoch = []
            recall_epoch = []
            f1_score_epoch = []
            net.eval()
            for image, label in val_loader:
                #load the val data
                image = image.to(device=device, dtype=torch.float32)
                label = label.squeeze(dim=1)
                label = label.to(device=device, dtype=torch.int64)
                #forward
                pred = net(image)
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = pred.argmax(dim=1)
                #evaluate
                precision, recall, f1_score = evaluate(pred, label)
                precision_epoch.append(precision)
                recall_epoch.append(recall)
                f1_score_epoch.append(f1_score)
            #Reserve the metrixs for this round
            precision_plt.append(np.mean(precision_epoch))
            recall_plt.append(np.mean(recall_epoch))
            f1_score_plt.append(np.mean(f1_score_epoch))
            #Log metrixs in time
            logging.info(f"epoch:{epoch}, loss/train:{np.mean(loss_epoch)}, "
                         f"precision:{np.mean(precision_epoch)}, recall:{np.mean(recall_epoch)}, "
                         f"f1_score:{np.mean(f1_score_epoch)}")
            #save model、loss、precision、recall and f1_score based the max f1_score
            if np.mean(f1_score_epoch) > f1_score_last:
                f1_score_last = np.mean(f1_score_epoch)
                torch.save(net.state_dict(), 'best_model.pth')
                epoch_new = epoch
                loss_new = np.mean(loss_epoch)
                precision_new = np.mean(precision_epoch)
                recall_new = np.mean(recall_epoch)
                f1_score_new = f1_score_last
        
        #6、training overed, then return the list of loss and score
        logging.info(f"Training overed!")
        logging.info(f"Ultimate epoch:{epoch_new}, "
                    f"loss/train:{loss_new}, "
                    f"precision:{precision_new}, "
                    f"recall:{recall_new}, "
                    f"f1_score:{f1_score_new}")
        return {'epoch_list': epoch_plt,
            'loss_list': loss_plt,
            'precision_list': precision_plt,
            'recall_list': recall_plt,
            'f1_score_list': f1_score_plt}
    except KeyboardInterrupt:
        logging.info(f"Training overed!")
        logging.info(f"Ultimate epoch:{epoch_new}, "
                     f"loss/train:{loss_new}, "
                     f"precision:{precision_new}, "
                     f"recall:{recall_new}, "
                     f"f1_score:{f1_score_new}")
        return {'epoch_list': epoch_plt,
                'loss_list': loss_plt,
                'precision_list': precision_plt,
                'recall_list': recall_plt,
                'f1_score_list': f1_score_plt}

def visualization(epoch_list, loss_list, precision_list, recall_list, f1_score_list):
    """Visualization the loss, precision, recall, f1_score of each epoch"""
        #loss
    plt.figure()
    plt.plot(epoch_list, loss_list, label='train_loss', marker='.')
    plt.legend()
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('train_loss_plot.png')
        #precision
    plt.figure()
    plt.plot(epoch_list, precision_list, label='precision', marker='.')
    plt.legend()
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.savefig('precision_plot.png')
        #recall
    plt.figure()
    plt.plot(epoch_list, recall_list, label='recall', marker='.')
    plt.legend()
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.savefig('recall_plot.png')
        #f1_score
    plt.figure()
    plt.plot(epoch_list, f1_score_list, label='f1_score', marker='.')
    plt.legend()
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.savefig('f1_score_plot.png')

if __name__ == "__main__":
    #Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #1、Initial model and arguments
    u_net = UNet(in_channels=3, out_classes=2)
    logging.info(f'UNet: In_Channel is {u_net.in_channels}, Out_Classes is {u_net.out_classes}')

    #2、Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device using {device}')

    #3、Initial dataset path
    data_path = '/root/autodl-tmp/u-net-pytroch-v2_5/dataset/train/agu_data/image/'
    logging.info(f'data_path is {data_path}')

    #4、Initial super arguments
    epochs = 200
    batch_size = 8
    lr= 0.00001
    val_percent = 0.10
    logging.info(f'epochs is {epochs}, batch_size is {batch_size}, learn_rate is {lr}, val_percent is {val_percent}')

    #5、Begin training
    train_dict = unet_train(
        net=u_net,
        device=device,
        data_path=data_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_percent=val_percent
    )

    #6、Visualization
    visualization(
        epoch_list=train_dict['epoch_list'],
        loss_list=train_dict['loss_list'],
        precision_list=train_dict['precision_list'],
        recall_list=train_dict['recall_list'],
        f1_score_list=train_dict['f1_score_list']
    )