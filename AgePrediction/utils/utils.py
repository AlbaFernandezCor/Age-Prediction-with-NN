import os
import torch
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatasetAge(Dataset):
    """Custom Dataset for loading face images"""

    def __init__(self, csv_path, img_dir, loss, num_classes, dataset, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.dataset = dataset
        self.csv_path = csv_path
        if self.dataset == 'AFAD':
            self.img_paths = df['path']
        elif self.dataset == 'CACD':
            self.img_paths = df['file'].values
        else:
            raise ValueError("Incorrect model name introduced.")
        self.y = df['age'].values
        self.transform = transform
        self.loss = loss
        self.NUM_CLASSES = num_classes

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_paths[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        if self.loss != 'ce':
            levels = [1] * label + [0] * (self.NUM_CLASSES - 1 - label)
            levels = torch.tensor(levels, dtype=torch.float32)
            return img, label, levels

        return img, label

    def __len__(self):
        return self.y.shape[0]


def return_paths(df):
    ll_df = []
    if df == 'CACD':
        ll_df.append('./coral-cnn-master/datasets/cacd_train.csv')  # path train
        ll_df.append('./coral-cnn-master/datasets/cacd_valid.csv')  # validation train
        ll_df.append('./coral-cnn-master/datasets/cacd_test.csv')  # test train
        ll_df.append('./coral-cnn-master/datasets/CACD2000-centered')  # path train
        return ll_df
    elif df == 'AFAD':
        ll_df.append('./coral-cnn-master/datasets/afad_train.csv')  # path train
        ll_df.append('./coral-cnn-master/datasets/afad_valid.csv')  # validation train
        ll_df.append('./coral-cnn-master/datasets/afad_test.csv')  # test train
        ll_df.append('./coral-cnn-master/datasets/tarball-master/AFAD-Full')  # path train
        return ll_df
    else:
        raise ValueError("Incorrect dataset introduced.")


def task_importance_weights(label_array, imp_weight, num_classes):
    if not imp_weight:
        imp = torch.ones(num_classes - 1, dtype=torch.float)
        return imp
    elif imp_weight == 1:
        uniq = torch.unique(label_array)
        num_examples = label_array.size(0)
        m = torch.zeros(uniq.shape[0])
        for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
            m_k = torch.max(torch.tensor([label_array[label_array > t].size(0),
                                          num_examples - label_array[label_array > t].size(0)]))
            m[i] = torch.sqrt(m_k.float())
        imp = m / torch.max(m)
        imp = imp[0:num_classes - 1]
        return imp
    else:
        raise ValueError('Incorrect importance weight parameter.')


def df_loader(train_p, valid_p, test_p, image_p, batch_size, n_workers, loss_dataset, num_classes, dataset):
    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.RandomCrop((120, 120)),
                                           transforms.ToTensor()])

    train_dataset = DatasetAge(csv_path=train_p,
                               img_dir=image_p,
                               loss=loss_dataset, num_classes=num_classes, dataset=dataset,
                               transform=custom_transform)

    custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                            transforms.CenterCrop((120, 120)),
                                            transforms.ToTensor()])

    test_dataset = DatasetAge(csv_path=test_p,
                              img_dir=image_p,
                              loss=loss_dataset, num_classes=num_classes, dataset=dataset,
                              transform=custom_transform2)

    valid_dataset = DatasetAge(csv_path=valid_p,
                               img_dir=image_p,
                               loss=loss_dataset, num_classes=num_classes, dataset=dataset,
                               transform=custom_transform2)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=n_workers)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=n_workers)
    return train_loader, valid_loader, test_loader, len(train_dataset)


def cost_fn(nom_model, logits=None, levels=None, imp=None, targets=None):
    if nom_model == 'ce':
        return F.cross_entropy(logits, targets)
    if nom_model == 'coral':
        val = (-torch.sum((F.logsigmoid(logits) * levels
                           + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                          dim=1))
        return torch.mean(val)
    if nom_model == 'ordinal':
        val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1] * levels
                           + F.log_softmax(logits, dim=2)[:, :, 0] * (1 - levels)) * imp, dim=1))
        return torch.mean(val)
    else:
        raise ValueError('ERROR EN LA TRIA DE MODEL (cost_fn)')


def compute_mae_and_mse(model, data_loader, device, nom_model):
    mae, mse, num_examples = 0, 0, 0
    for i, tupla in enumerate(data_loader):
        if nom_model == 'ce':
            features = tupla[0]
            targets = tupla[1]
            levels = None
        else:
            features = tupla[0]
            targets = tupla[1]
            levels = tupla[2]
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        if nom_model != 'ce':
            predict_levels = probas > 0.5
            predicted_labels = torch.sum(predict_levels, dim=1)
        else:
            _, predicted_labels = torch.max(probas, 1)

        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets) ** 2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse
