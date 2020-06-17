import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

import os
import glob
import argparse
import statistics
import random
from tqdm import tqdm

from utils import load_dataset
from model import Unet

DATA_DIR = '../train'
CASE_LIST_PATH = DATA_DIR + '/case_list.txt'


class nFoldCrossVaridation:
    def __init__(self, n_fold, seed):
        random.seed(seed)
        with open(CASE_LIST_PATH, 'r') as f:
            self.case_list = [row.strip() for row in f]
             
        self.num_data  = len(self.case_list)
        self.n_fold = n_fold
        assert self.num_data % self.n_fold == 0
        self.step = self.num_data // self.n_fold

        random.shuffle(self.case_list)

        self.d_lis = []
        self.l_lis = []
        for case in self.case_list:
            self.d_lis.append(os.path.join(DATA_DIR, 'Image', case + '.mhd'))
            self.l_lis.append(os.path.join(DATA_DIR, 'Label', case + '.mhd'))

        self.logger()


    def __iter__(self):
        stIdx = 0
        edIdx = self.step

        for _ in range(self.n_fold):
            x_train = self.d_lis[:stIdx] + self.d_lis[edIdx:]
            y_train = self.l_lis[:stIdx] + self.l_lis[edIdx:]
            x_valid = self.d_lis[stIdx:edIdx]
            y_valid = self.l_lis[stIdx:edIdx]

            stIdx += self.step
            edIdx += self.step

            yield x_train, y_train, x_valid, y_valid
    
    def logger(self):
        
        stIdx = 0
        edIdx = self.step

        log = []
        for fold in range(self.n_fold):
            train_case = self.case_list[:stIdx] + self.case_list[edIdx:]
            valid_case = self.case_list[stIdx:edIdx]
            log.append('Fold {}, TRAIN: {}\n'.format(fold+1, train_case))
            log.append('Fold {}, VALID: {}\n'.format(fold+1, valid_case))

            stIdx += self.step
            edIdx += self.step

        with open('./fold_log.txt', mode='w', encoding='UTF-8') as f:
            f.writelines(log)   


def calc_dice(pred, label):
    pred = pred.detach().cpu().numpy() > 0.5
    label = label.detach().cpu().numpy().astype(bool)

    dice_score = []
    for i in range(pred.shape[0]):
        dice = 2. * (pred[i] & label[i]).sum() / (pred[i].sum() + label[i].sum())
        dice_score.append(dice)

    return statistics.mean(dice_score) 

def train(args, x_train, y_train, x_valid, y_valid):

    writer = SummaryWriter()

    best_dice = 0 

    model = Unet().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    bce_loss = torch.nn.BCELoss()

    train_dataloader = load_dataset(x_train, y_train, args.batch_size, True)
    valid_dataloader = load_dataset(x_valid, y_valid, args.batch_size, False)

    result = {}
    result['train/BCE'] = []
    result['train/Dice'] = []
    result['valid/BCE'] = []
    result['valid/Dice'] = []

    for epoch in range(args.epochs):
        print('train step: epoch {}'.format(str(epoch+1).zfill(4)))

        train_bce = []
        train_dice = []

        for inp_im, lab_im in tqdm(train_dataloader):
            inp_im = inp_im.to(args.device)
            lab_im = lab_im.to(args.device)

            pred = model(inp_im)

            bce = bce_loss(pred, lab_im)
            dice = calc_dice(pred, lab_im)

            train_bce.append(bce.item())
            train_dice.append(dice)

            model.zero_grad()
            bce.backward()
            optimizer.step()
        
        result['train/BCE'].append(statistics.mean(train_bce))
        result['train/Dice'].append(statistics.mean(train_dice))

        writer.add_scalar('train/BinaryCrossEntropy', result['train/BCE'][-1], epoch+1)
        writer.add_scalar('train/DiceScore', result['train/Dice'][-1], epoch+1)

        print('BCE: {}, Dice: {}'.format(result['train/BCE'][-1], result['train/Dice'][-1]))

        if (epoch+1) % 10 == 0 or (epoch+1) == 1:

            with torch.no_grad():
                print('valid step: epoch {}'.format(str(epoch+1).zfill(4)))
                model.eval()

                valid_bce = []
                valid_dice = []
                for inp_im, lab_im in tqdm(valid_dataloader):
                    inp_im = inp_im.to(args.device)
                    lab_im = lab_im.to(args.device)

                    pred = model(inp_im)

                    bce = bce_loss(pred, lab_im)
                    dice = calc_dice(pred, lab_im)

                    valid_bce.append(bce.item())
                    valid_dice.append(dice)
                
                result['valid/BCE'].append(statistics.mean(valid_bce))
                result['valid/Dice'].append(statistics.mean(valid_dice))

                writer.add_scalar('valid/BinaryCrossEntropy', result['valid/BCE'][-1], epoch+1)
                writer.add_scalar('valid/DiceScore', result['valid/Dice'][-1], epoch+1)

                print('BCE: {}, Dice: {}'.format(result['valid/BCE'][-1], result['valid/Dice'][-1]))


                if best_dice < result['valid/Dice'][-1]:
                    best_dice = result['valid/Dice'][-1]

                    best_model_name = os.path.join(args.save_model_path, f'best_model_{epoch + 1:04}.pth')
                    print('save model ==>> {}'.format(best_model_name))
                    torch.save(model.state_dict(), best_model_name)
        

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_model_dir', type=str, default='./model/')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_fold', type=int, default=6)

    args = parser.parse_args()
    return args

def main(args):

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cross_valid = nFoldCrossVaridation(args.n_fold, args.seed)

    fold = 1
    for x_train, y_train, x_valid, y_valid in cross_valid:
        print('start {} fold'.format(fold))

        args.save_model_path = os.path.join(args.save_model_dir, 'Fold-{}'.format(str(fold)))
        check_dir(args.save_model_path)

        train(args, x_train, y_train, x_valid, y_valid)

        fold += 1


if __name__ == '__main__':
    args = arg_parser()
    main(args)
