#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import json
import os

import os

# local_rank = int(os.environ["LOCAL_RANK"])
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.nn.parallel import DistributedDataParallel
import argparse

import sys
import pdb
import shutil
import time
import copy
import pickle
import torch
from tqdm import tqdm
from augmenter import NormalAugmenter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from option import parser, verify_input_args
import models.search.darts.utils as utils
import data
import jsonlines
import torch.backends.cudnn as cudnn
from vocab import Vocabulary  # necessary import
from artemis_model import ARTEMIS, FoundARTEMIS
from architect import Architect
from loss import LossModule,TripletLoss
from evaluate import validate, update_arch_parameters
from logger import AverageMeter
from perturb import Random_alpha, Linf_PGD_alpha
import logging
import torch.optim as op
from utils import count_parameters, save_pickle, save

################################################################################
# *** UTILS
################################################################################

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def resume_from_ckpt_saved_states(args, model, optimizer):
    """
    Load model, optimizer, and previous best score.
    """

    # load checkpoint
    assert os.path.isfile(args.ckpt), f"(ckpt) File not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt)
    print(f"Loading file {args.ckpt}.")

    # load model
    if torch.cuda.is_available():
        # load parameters pretrained from last 8 epoch to this new model
        model.load_state_dict(ckpt['model'])
    else:
        state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['model']
        model.load_state_dict(state_dict)
    print("Model: resume from provided state.")

    # load the optimizer state
    optimizer.load_state_dict(ckpt['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v
                if torch.cuda.is_available():
                    state[k] = state[k].cuda()
    print("Optimizer: resume from provided state.")

    # load the previous best score
    best_score = ckpt['best_score']
    print("Best score: obtained from provided state.")

    return model, optimizer, best_score


################################################################################
# *** TRAINING FOR ONE EPOCH
################################################################################

def train_model(epoch, data_loader, model, criterion, optimizer, perturb_alpha, epsilon_alpha, arch_optimizer, args):
    # Switch to train mode
    model.train()

    # Average meter to record the training statistics
    loss_info = AverageMeter(precision=8)  # precision: number of digits after the comma
    ce = F.cross_entropy
    max_itr = len(data_loader)
    for itr, data in tqdm(enumerate(data_loader)):

        # Get data
        img_src, txt, txt_len, img_trg, real_text, _ = data  # img_src torch.Size([32, 3, 224, 224]) ,txt torch.Size([32, 21]), img_trg  torch.Size([32, 3, 224, 224])

        if torch.cuda.is_available():
            img_src, img_trg, txt, txt_len = img_src.cuda(), img_trg.cuda(), txt.cuda(), txt_len.cuda()

        r = model.get_image_embedding(img_src,epoch)
        m = model.get_txt_embedding(txt, real_text, txt_len)
        t = model.get_image_embedding(img_trg,epoch)


        if perturb_alpha:
            perturb_alpha(model, epsilon_alpha)
            optimizer.zero_grad()
            arch_optimizer.zero_grad()

        # Forward pass
        scores = model(r, m, t)  # 32 * 32

        scores = scores.cuda()
        if args.learn_temperature:
            scores *= model.temperature.exp()
        loss = criterion(scores)

        total_loss = loss

        loss_info.update(total_loss.item())

        optimizer.zero_grad()

        total_loss.backward()

        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        optimizer.step()

        # Print log info
        if itr > 0 and (itr % args.log_step == 0 or itr + 1 == max_itr):
            log_msg = 'loss: %s' % str(loss_info)
            logging.info('[%d][%d/%d] %s' % (epoch, itr, max_itr, log_msg))

    return loss_info.avg


def unpdate_arch_params(model: object, args: object, vocab: object, phase, epoch: object = -1,
                        best_score: object = None, split: object = 'val', architect: object = None) -> object:
    model.train()

    start = time.time()
    update_arch_parameters(model, args, vocab, phase, epoch, split=split, architect=architect)
    end = time.time()

    log_msg = "\nProcessing time : %f" % (end - start)
    logging.info(log_msg)


def validate_model_found(model: object, args: object, vocab: object, phase, epoch: object = -1,
                         best_score: object = None, split: object = 'val', architect: object = None) -> object:
    model.eval()

    with torch.no_grad():
        start = time.time()
        message, val_mes = validate(model, args, vocab, phase, epoch, split=split, architect=architect)
        end = time.time()

    log_msg = "[%s][%d] >> EVALUATION <<" % (args.exp_name, epoch)
    log_msg += "\nProcessing time : %f" % (end - start)
    log_msg += message

    if best_score:
        log_msg += '\nCurrent best score: %.2f' % (best_score)

    logging.info(log_msg)

    return val_mes


def update_best_score(new_score, old_score, is_higher_better=True):
    if not old_score:
        score, updated = new_score, True
    else:
        if is_higher_better:
            score = max(new_score, old_score)
            updated = new_score > old_score
        else:
            score = min(new_score, old_score)
            updated = new_score < old_score
    return score, updated


def save_ckpt(state, is_best, args, filename='ckpt.pth', split='val'):
    ckpt_path = os.path.join(args.ckpt_dir, args.exp_name, filename)
    torch.save(state, ckpt_path)
    if is_best:
        model_best_path = os.path.join(args.ckpt_dir, args.exp_name, split, 'model_best.pth')
        shutil.copyfile(ckpt_path, model_best_path)
        logging.info('Updating the best model checkpoint: {}'.format(model_best_path))


def main():
    # Parse & correct arguments
    args = verify_input_args(parser.parse_args())
    print(args)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif args.perturb_alpha == 'random':
        perturb_alpha = Random_alpha

        # Load vocabulary
    vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')
    assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
    vocab = pickle.load(open(vocab_path, 'rb'))

    model = ARTEMIS(vocab.word2idx, args)
    model.cuda()
    torch.backends.cudnn.benchmark = True
    print("Model version:", args.model_version)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.num_epochs))

    criterion = LossModule(args)  # CE loss

    trn_loader = data.get_train_loader(args, vocab)

    arch_optimizer = op.Adam(model.arch_parameters(),
                             lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    architect = Architect(model, args, criterion, arch_optimizer)

    best_score = 0

    for epoch in range(args.num_epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # update arch params
        for split in args.validate:
            print("**********************starting update arch parameters************************")

            unpdate_arch_params(model, args, vocab, "search", epoch, split=split, architect=architect)
            optimizer.zero_grad()
            arch_optimizer.zero_grad()
            genotype = model.genotype()
            logger.info("after update genotype " + str(genotype))

            print("**********************finish update arch parameters************************")
        print(" ")

        if args.perturb_alpha:
            epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.num_epochs
            logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

        # train for one epoch and update model params
        print("**********************update model parameters************************")

        model.fusion_net.softmax_arch_parameters()

        train_model(epoch, trn_loader, model, criterion, optimizer, perturb_alpha, epsilon_alpha, arch_optimizer, args)

        model.fusion_net.restore_arch_parameters()

        num_params = 0
        num_params += count_parameters(model)  # 此时模型参数的数量
        logger.info("Fusion Model Params: {}".format(num_params))
        genotype = model.genotype()
        logger.info("training stage: " + str(genotype))

        print("**********************finish update model parameters************************")
        print(" ")
        # evaluate
        val_score = validate_model_found(model, args, vocab, "test", epoch, best_score,
                                         split='val', architect=architect)
        # remember best validation score
        update_best_score(val_score, best_score)
        if val_score > best_score:

            best_score = val_score
            genotype = model.genotype()
            best_genotype = copy.deepcopy(genotype)
            best_epoch = epoch
            # best_model_sd = copy.deepcopy(model.state_dict())

            if args.parallel:
                torch.save(model.module,
                           os.path.join(args.ckpt_dir, args.exp_name, "best_model.pt"))
            else:
                torch.save(model.state_dict(),
                           os.path.join(args.ckpt_dir, args.exp_name, "best_model.pt"))

            save_pickle(best_genotype,
                        os.path.join(args.ckpt_dir, args.exp_name, "best_val_score_genotype.pkl"))

            print("")
            num_params = 0
            num_params += count_parameters(model)  # 此时模型参数的数量
            logger.info("*******************update current best genotype*********************")
            logger.info("Fusion Model Params: {}".format(num_params))

            genotype = model.genotype()
            logger.info("best epoch:" + str(best_epoch))
            logger.info("val stage: " + str(genotype))
        save_pickle(best_genotype, os.path.join(args.ckpt_dir, args.exp_name, "cur_epoch_genotype.pkl"))


def found_and_test():
    args = verify_input_args(parser.parse_args())

    # Load vocabulary
    vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')
    assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
    vocab = pickle.load(open(vocab_path, 'rb'))

    best_genotype_path = os.path.join(args.ckpt_dir, "artemis_experiments/fashionIQ-ARTEMIS", "best_genotype.pkl")
    args.save = '{}-{}'.format(args.ckpt_dir, "search", time.strftime("%Y%m%d-%H%M%S"))

    # utils.create_exp_dir(args.save, scripts_to_save=None)

    # log
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logging.info("args = %s", args)

    genotype = utils.load_pickle(best_genotype_path)
    print("best_genotyepe:", str(genotype))

    criterion = LossModule(args)

    model = FoundARTEMIS(vocab.word2idx, args, genotype, criterion).float()

    # model = ARTEMIS(vocab.word2idx, args)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    if args.use_clip:
        param_dict = model.clip_model.get_params()
        param_dict.append({'params': model.Transform_m.parameters(), 'lr': args.lr})
        param_dict.append({'params': model.fusion_net.parameters(), 'lr': args.lr})
        optimizer = torch.optim.AdamW(param_dict)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    best_score = {split: None for split in args.validate}
    if args.ckpt != 'None':
        print(args.ckpt)

        model, optimizer, best_score = resume_from_ckpt_saved_states(args, model, optimizer)

    train_loader = data.get_train_loader(args, vocab)

    start_time = time.time()


    for epoch in range(args.num_epochs):
        # step_lr
        if epoch != 0 and epoch % args.step_lr == 0:
            for g in optimizer.param_groups:
                print("Learning rate: {} --> {}\n".format(g['lr'], g['lr'] * args.gamma_lr))
                g['lr'] *= args.gamma_lr

        # train for one epoch
        train_model(epoch, train_loader, model, criterion, optimizer, None, None, None, args)
        num_params = 0
        num_params += count_parameters(model)  # 此时模型参数的数量
        logger.info("Fusion Model Params: {}".format(num_params))
        genotype = model.genotype()
        logger.info(str(genotype))

        for split in args.validate:
            print("Validating on the {} split.".format(split))

            # evaluate the current split
            with torch.no_grad():
                val_score = validate_model_found(model, args, vocab, "found", epoch, best_score[split], split=split)

            # remember best validation score
            best_score[split], updated = update_best_score(val_score, best_score[split])

            # save ckpt
            save_ckpt({
                'args': args,
                'epoch': epoch,
                'best_score': best_score,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, updated, args, split=split)

        print("")

    time_elapsed = time.time() - start_time

    logger.info("*" * 50)
    logger.info('Total duration {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # logger.info('Final model {} F1: {}'.format(args.f1_type, model_f1))
    return


if __name__ == '__main__':
    main()
    # found_and_test()
