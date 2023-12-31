import os
import sys
import time
import argparse

import wandb

from utils.utils import *
from models.models import *

########## ARGUMENTS #############
parser = argparse.ArgumentParser()
parser.add_argument('--cuda',
                    type=int,
                    default=-1)

parser.add_argument('--numworkers',
                    type=int,
                    default=6)

parser.add_argument('--seed',
                    type=int,
                    default=-1)

parser.add_argument('--outpath',
                    type=str,
                    required=True)

parser.add_argument('--imp_weight',
                    type=int,
                    default=0)

parser.add_argument('--dataset',
                    type=str,
                    default='CACD')

parser.add_argument('--loss',
                    type=str,
                    default='coral')

parser.add_argument('--starting_params',
                    type=int,
                    default=-1)

parser.add_argument('--state_dict_path',
                    type=str)

args = parser.parse_args()

NUM_WORKERS = args.numworkers
DATASET = args.dataset
LOSS = args.loss
STARTING_PARAMS = args.starting_params

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

IMP_WEIGHT = args.imp_weight

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')

path_list = return_paths(DATASET)
TRAIN_CSV_PATH = path_list[0]
VALID_CSV_PATH = path_list[1]
TEST_CSV_PATH = path_list[2]
IMAGE_PATH = path_list[3]

if STARTING_PARAMS >= 0:
    STATE_DICT_PATH = args.state_dict_path

header = []
header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Output Path: %s' % PATH)
header.append('Script: %s' % sys.argv[0])

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()

########## SETTINGS #############

# Hyperparameters
learning_rate = 0.0005
num_epochs = 5

# Architecture
if DATASET == 'CACD':
    NUM_CLASSES = 49
elif DATASET == 'AFAD':
    NUM_CLASSES = 26
else:
    raise ValueError("Incorrect dataset introduced.")

BATCH_SIZE = 256
GRAYSCALE = False

if LOSS != 'ce':
    df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
    ages = df['age'].values
    del df
    ages = torch.tensor(ages, dtype=torch.float)
    if not IMP_WEIGHT:
        imp = torch.ones(NUM_CLASSES - 1, dtype=torch.float)
    elif IMP_WEIGHT == 1:
        imp = task_importance_weights(ages)
        imp = imp[0:NUM_CLASSES - 1]
    else:
        raise ValueError('Incorrect importance weight parameter.')
    imp = imp.to(DEVICE)
else:
    imp = None

# Transformations
train_loader, valid_loader, test_loader, len_train_dataset = df_loader(TRAIN_CSV_PATH, VALID_CSV_PATH,
                                                                       TEST_CSV_PATH, IMAGE_PATH, BATCH_SIZE,
                                                                       NUM_WORKERS, LOSS,
                                                                       NUM_CLASSES, DATASET)

# Create the model and optimizer
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = resnet34(NUM_CLASSES, GRAYSCALE, LOSS)

if args.starting_params >= 0:
    model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=DEVICE))

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

########## TRAIN LOOP #############
best_mae, best_rmse, best_epoch = 999, 999, -1
for epoch in range(num_epochs):
    model.train()

    for batch_idx, tupla in enumerate(train_loader):
        if (LOSS == 'ce'):
            features = tupla[0]
            targets = tupla[1]
            levels = None
        else:
            features = tupla[0]
            targets = tupla[1]
            levels = tupla[2]
            levels = levels.to(DEVICE)

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = cost_fn(nom_model=LOSS, logits=logits, levels=levels, imp=imp, targets=targets)
        optimizer.zero_grad()
        cost.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        if not batch_idx % 50:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                 % (epoch + 1, num_epochs, batch_idx,
                    len_train_dataset // BATCH_SIZE, cost))
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):
        valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                                   device=DEVICE, nom_model=LOSS)

    if valid_mae < best_mae:
        best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
        ########## SAVE MODEL #############
        torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))

    s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
        valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time) / 60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

model.eval()
with torch.set_grad_enabled(False):  # save memory during inference

    train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE, nom_model=LOSS)
    valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                               device=DEVICE, nom_model=LOSS)
    test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE, nom_model=LOSS)

    s = 'MAE/RMSE: | Train: %.2f/%.2f | Valid: %.2f/%.2f | Test: %.2f/%.2f' % (
        train_mae, torch.sqrt(train_mse),
        valid_mae, torch.sqrt(valid_mse),
        test_mae, torch.sqrt(test_mse))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

s = 'Total Training Time: %.2f min' % ((time.time() - start_time) / 60)
print(s)
with open(LOGFILE, 'a') as f:
    f.write('%s\n' % s)

########## EVALUATE BEST MODEL ######
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
model.eval()

with torch.set_grad_enabled(False):
    train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE, nom_model=LOSS)
    valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                               device=DEVICE, nom_model=LOSS)
    test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE, nom_model=LOSS)

    s = 'MAE/RMSE: | Best Train: %.2f/%.2f | Best Valid: %.2f/%.2f | Best Test: %.2f/%.2f' % (
        train_mae, torch.sqrt(train_mse),
        valid_mae, torch.sqrt(valid_mse),
        test_mae, torch.sqrt(test_mse))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

########## SAVE PREDICTIONS ######
all_pred_str = []
all_pred_int = []
all_probas = []
with torch.set_grad_enabled(False):
    for batch_idx, tupla in enumerate(test_loader):
        lst_str = []
        lst_int = []
        if (LOSS == 'ce'):
            features = tupla[0]
            targets = tupla[1]
            levels = None
        else:
            features = tupla[0]
            targets = tupla[1]
            levels = tupla[2]

        features = features.to(DEVICE)
        logits, probas = model(features)
        all_probas.append(probas)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        for i in predicted_labels:
            lst_str.append(str(int(i)))
            lst_int.append(int(i))
        all_pred_str.extend(lst_str)
        all_pred_int.extend(lst_int)

all_pred_int = torch.tensor(all_pred_int, dtype=torch.int)
df = pd.read_csv(TEST_CSV_PATH, index_col=0)
ages = df['age'].values  # Labels
del df
ages = torch.tensor(ages, dtype=torch.float)
dif = ages - all_pred_int

# for i in range(len(dif)):
#     print("Pred:", int(all_pred_int[i]), "Age:", int(ages[i]), "Dif:", int(dif[i]))

print("\nmean dif:")
print(torch.mean(dif.float()))
print("\nmean abs(dif):")
print(torch.mean(torch.abs(dif.float())))
print("\nstd:")
print(torch.std(dif.float()))
print("\nmin:")
print(torch.min(dif.float()))
print("\nmax:")
print(torch.max(dif.float()))

if (LOSS != 'ce'):
    torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)

with open(TEST_PREDICTIONS, 'w') as f:
    all_pred_str = ','.join(all_pred_str)
    f.write(all_pred_str)
