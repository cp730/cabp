import pdb

import torch
import models.auxiliary.scheduler as sc
import copy
import time

from jsonlines import jsonlines
# from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from models.search.darts.utils import count_parameters, save, save_pickle
from loss import LossModule
from logger import AverageMeter
import logging
from train import validate_model,save_ckpt,dev_phase_model,update_best_score
import data
from data import get_train_loader,get_eval_loaders
from evaluate import validate
def train_fashionIQ_track_f1(model, architect,
                             criterion, optimizer, scheduler, dataloaders,
                             device, num_epochs,
                             parallel, logger, plotter, args,
                             vocab,
                             status='search'):
    criterion = LossModule(args)
    loss_info = AverageMeter(precision=8)  # precision: number of digits after the comma

    best_genotype = None
    best_epoch = 0
    best_score = 0

    # best_test_genotype = None
    # best_test_score = init_f1
    # best_test_epoch = 0
    failsafe = True
    cont_overloop = 0
    while failsafe:
        for epoch in range(num_epochs):

            logger.info('Epoch: {}'.format(epoch))
            logger.info("EXP: {}".format(args.save))

            if status == 'search':
                phases = ['train', 'dev']
            else:
                # while evaluating, add dev set to train also
                phases = ['train', 'dev', 'test']

            # Each epoch has a training and validation phase
            for phase in phases:  # train 、 dev
                if phase == 'train':
                    if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                        scheduler.step()
                    if architect is not None:
                        architect.log_learning_rate(logger)
                    model.train()  # Set model to training mode
                    trn_loader = get_train_loader(args, vocab)
                    # model.genotype()
                    # queries_loader, targets_loader = data.get_eval_loaders(args, vocab, 'val')
                    max_itr = len(trn_loader)

                    for itr, data in tqdm(enumerate(trn_loader)):
                            # Iterate over data.
                            # get the inputs
                            img_src, txt, txt_len, img_trg, _, _ = data  # batchseize = 32
                            # pdb.set_trace()
                            '''
                                    img_src:torch.Size([32, 3, 224, 224])
                                    txt:torch.Size([32, 19])
                                    txt_len:len = 32 ,eg:tensor([19, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 11, 11, 11, 
                                                                    11, 11, 11, 10, 10,  8,  6])
                                    img_trg：torch.Size([32, 3, 224, 224])
                            '''
                            img_src = img_src.to(device)
                            txt = txt.to(device)
                            img_trg = img_trg.to(device)
                            txt_len = txt_len.to(device)

                            # 如果在dev或test阶段，即数据集使用测试集或验证集
                            if status == 'search' and (phase == 'dev' or phase == 'test'):
                                # 更新架构参数，此时更新的架构参数是在
                                architect.step((txt, img_src), img_trg, logger)
                            # zero the parameter gradients，模型参数梯度归零
                            optimizer.zero_grad()

                            # pdb.set_trace()
                            with torch.set_grad_enabled(phase == 'train' or (phase == 'dev' and status == 'eval')):
                                output = model((txt, img_src), txt_len) # 32*32

                                loss = criterion(output)
                                # pdb.set_trace()
                                loss_info.update(loss.item())
                                # backward + optimize only if in training phase
                                # 训练集上训练，更新模型参数

                                if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                    scheduler.step()
                                    scheduler.update_optimizer(optimizer)
                                loss.backward()
                                optimizer.step()  # train阶段，更新模型参数
                                # Print log info
                                if itr > 0 and (itr % args.log_step == 0 or itr + 1 == max_itr):
                                    log_msg = 'loss: %s' % str(loss_info)
                                    logging.info('[%d][%d/%d] %s' % (epoch, itr, max_itr, log_msg))
                                # pdb.set_trace()
                    if parallel:
                        num_params = 0
                        for reshape_layer in model.module.reshape_layers:
                            num_params += count_parameters(reshape_layer)

                        num_params += count_parameters(model.module.fusion_net)
                        logger.info("Fusion Model Params: {}".format(num_params))

                        genotype = model.module.genotype()
                    else:
                        num_params = 0
                        # for reshape_layer in model.reshape_layers:
                        #     num_params += count_parameters(reshape_layer)
                        num_params += count_parameters(model.imagenet)
                        num_params += count_parameters(model.textnet)
                        num_params += count_parameters(model.fusion_net)
                        logger.info("Fusion Model Params: {}".format(num_params))

                        genotype = model.genotype()

                        logger.info(str(genotype))

                        pdb.set_trace()
                elif phase == 'dev':
                    if status == 'eval':
                        if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.step()
                    model.train()
                    # 每一个epoch结束后，如在测试/验证集上时，测试此时架构模型的精度
                    if status == 'search' and (phase == 'dev' or phase == 'test'):
                        if args.data_name == 'amazon':
                            for split in args.validate:
                                print("Validating on the {} split.".format(split))  # split = dev

                                # evaluate the current split
                                with torch.no_grad():
                                    # 根据模型的结果在jsonl文件中添加内容。
                                    txt_embs = model.get_txt_embedding(txt, txt_len)

                                    epoch_score, modify_annotations = dev_phase_model(model, args, vocab, epoch,
                                                                                      best_score[split],
                                                                                      split=split)

                                # remember best validation score
                                best_score[split], updated = update_best_score(epoch_score, best_score[split])
                                if updated:
                                    print("-----------modify_amazon_dev_file-------------")
                                    with jsonlines.open('./data/amazon/dev_file_res50_bs32_80.jsonl', 'w') as writer:
                                        writer.write_all(modify_annotations)
                                # save ckpt
                                save_ckpt({
                                    'args': args,
                                    'epoch': epoch,
                                    'best_score': best_score,
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                }, updated, args, split=split)
                            print("")
                        elif args.data_name == 'fashionIQ':
                            # evaluate the model & save state if best
                            for split in args.validate:
                                print("Validating on the {} split.".format(split))  # split

                                # evaluate the current split
                                with torch.no_grad():
                                    start = time.time()
                                    message, val_mes, _ = validate(model, args, vocab, split=split)
                                    end = time.time()

                                log_msg = "[%s][%d] >> EVALUATION <<" % (args.exp_name, epoch)
                                log_msg += "\nProcessing time : %f" % (end - start)
                                log_msg += message

                                if best_score:
                                    log_msg += '\nCurrent best score: %.2f' % (best_score)

                                logging.info(log_msg)

                                pdb.set_trace()
                                best_score[split], updated = update_best_score(epoch_score, best_score[split])

                                save_ckpt({
                                    'args': args,
                                    'epoch': epoch,
                                    'best_score': best_score,
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                }, updated, args, split=split)
                        if epoch_score > best_score:
                            best_score = epoch_score
                            best_genotype = copy.deepcopy(genotype)
                            best_epoch = epoch
                            best_model_sd = copy.deepcopy(model.state_dict())

                            if parallel:
                                save(model.module, os.path.join(args.save, 'best', 'GLG2ADD.pt'))
                            else:
                                save(model, os.path.join(args.save, 'best', 'GLG2ADD.pt'))

                            best_genotype_path = os.path.join(args.save, 'best', 'best_genotype1.pkl')
                            save_pickle(best_genotype, best_genotype_path)

                else: # phase = test
                    model.eval()  # Set model to evaluate mode
                    if epoch_score > best_test_score:
                        best_test_score = epoch_score
                        best_test_genotype = copy.deepcopy(genotype)
                        best_test_epoch = epoch

                        if parallel:
                            save(model.module, os.path.join(args.save, 'best', 'best_test_model.pt'))
                        else:
                            save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))

                        best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
                        save_pickle(best_test_genotype, best_test_genotype_path)
                #       手动更新进度条信息
                #         postfix_str = 'batch_loss: {:.03f}'.format(loss.item())
                #         t.set_postfix_str(postfix_str)
                #         t.update()
                # epoch_loss = running_loss / dataset_sizes[phase]

                # y_pred = torch.cat(list_preds, dim=0).numpy()
                # y_true = torch.cat(list_label, dim=0).numpy()

                # epoch_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
                # epoch_f1 = f1_score(y_true, y_pred, average=f1_type, zero_division=1)

                # logger.info('{} Loss: {:.4f}'.format(
                #     phase))


                # if phase == 'train' and epoch_loss != epoch_loss:
                #     logger.info("Nan loss during training, escaping")
                #     model.eval()
                #     return best_score

                # if phase == 'dev' and status == 'search':


                # if phase == 'test':


            file_name = "epoch_{}".format(epoch)
            file_name = os.path.join(args.save, "architectures", file_name)
            plotter.plot(genotype, file_name, task='fashionIQ')

            logger.info("Current best dev {} F1: {}, at training epoch: {}".format(f1_type, best_f1, best_epoch))
            logger.info(
                "Current best test {} F1: {}, at training epoch: {}".format(f1_type, best_test_f1, best_test_epoch))

        if best_score != best_score and num_epochs == 1 and cont_overloop < 1:
            failsafe = True
            logger.info('Recording a NaN F1, training for one more epoch.')
        else:
            failsafe = False

        cont_overloop += 1

    if best_score != best_score:
        best_score = 0.0

    if status == 'search':
        pdb.set_trace()
        return best_score, best_genotype
    else:
        return best_test_score, best_test_genotype
#
# def train_fashionIQ_track_f1(model, architect,
#                           criterion, optimizer, scheduler, dataloaders,
#                           dataset_sizes, device, num_epochs,
#                           parallel, logger, plotter, args,
#                           f1_type='weighted', init_f1=0.0, th_fscore=0.3,
#                           status='search'):
#
#     criterion = LossModule(args)
#     loss_info = AverageMeter(precision=8)  # precision: number of digits after the comma
#
#
#     best_genotype = None
#     best_f1 = init_f1 #
#     best_epoch = 0
#
#     best_test_genotype = None
#     best_test_f1 = init_f1
#     best_test_epoch = 0
#
#     failsafe = True
#     cont_overloop = 0
#     while failsafe:
#         for epoch in range(num_epochs):
#
#             logger.info('Epoch: {}'.format(epoch))
#             logger.info("EXP: {}".format(args.save))
#
#             phases = []
#             if status == 'search':
#                 phases = ['train', 'dev']
#             else:
#                 # while evaluating, add dev set to train also
#                 phases = ['train', 'dev', 'test']
#
#             # Each epoch has a training and validation phase
#             for phase in phases:  # train 、 dev
#                 if phase == 'train':
#                     if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
#                         scheduler.step()
#                     if architect is not None:
#                         architect.log_learning_rate(logger)
#                     model.train()  # Set model to training mode
#                     list_preds = []
#                     list_label = []
#                 elif phase == 'dev':
#                     if status == 'eval':
#                         if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
#                             scheduler.step()
#                     model.train()
#                     list_preds = []
#                     list_label = []
#                 else:
#                     model.eval()  # Set model to evaluate mode
#                     list_preds = []
#                     list_label = []
#
#                 running_loss = 0.0
#                 running_f1 = init_f1
#
#                 max_itr = len(dataloaders[phase])
#                 with tqdm(dataloaders[phase]) as t:
#                     # Iterate over data.
#                     for itr, data in dataloaders[phase]:
#                         # get the inputs
#                         img_src, txt, txt_len, img_trg, _, _ = data # batchseize = 32
#                         '''
#                             img_src:torch.Size([32, 3, 224, 224])
#                             txt:torch.Size([32, 19])
#                             txt_len:len = 32 ,eg:tensor([19, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 11, 11, 11,
#                                                             11, 11, 11, 10, 10,  8,  6])
#                             img_trg：torch.Size([32, 3, 224, 224])
#                         '''
#                         # if torch.cuda.is_available():
#                         #     img_src, img_trg, txt, txt_len = img_src.cuda(), img_trg.cuda(), txt.cuda(), txt_len.cuda()
#                         # device
#                         img_src = img_src.to(device)
#                         txt = txt.to(device)
#                         img_trg = img_trg.to(device)
#                         txt_len = txt_len.to(device)
#
#                         # ????????
#                         if status == 'search' and (phase == 'dev' or phase == 'test'):
#                             architect.step((txt, img_src), img_trg, logger)
#
#                         # zero the parameter gradients
#                         optimizer.zero_grad()
#
#                         # forward
#                         # track history if only in train
#                         with torch.set_grad_enabled(phase == 'train' or (phase == 'dev' and status == 'eval')):
#                             output = model((txt, img_src),txt_len)
#
#                             if isinstance(output, tuple):
#                                 output = output[-1]
#
#                             pdb.set_trace()
#
#                             loss = criterion(output)
#                             loss_info.update(loss.item())
#
#                             preds_th = torch.sigmoid(output) > th_fscore
#
#                             # backward + optimize only if in training phase
#                             if phase == 'train' or (phase == 'dev' and status == 'eval'):
#                                 if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
#                                     scheduler.step()
#                                     scheduler.update_optimizer(optimizer)
#                                 loss.backward()
#                                 optimizer.step()
#                             # Print log info
#                             if itr > 0 and (itr % args.log_step == 0 or itr + 1 == max_itr):
#                                 log_msg = 'loss: %s' % str(loss_info)
#                                 logging.info('[%d][%d/%d] %s' % (epoch, itr, max_itr, log_msg))
#
#
#                 #       手动更新进度条信息
#                 #         postfix_str = 'batch_loss: {:.03f}'.format(loss.item())
#                 #         t.set_postfix_str(postfix_str)
#                 #         t.update()
#                 # epoch_loss = running_loss / dataset_sizes[phase]
#
#                 # y_pred = torch.cat(list_preds, dim=0).numpy()
#                 # y_true = torch.cat(list_label, dim=0).numpy()
#
#                 # epoch_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
#                 # epoch_f1 = f1_score(y_true, y_pred, average=f1_type, zero_division=1)
#
#                 logger.info('{} Loss: {:.4f}'.format(
#                     phase, ))
#                 if parallel:
#                     num_params = 0
#                     for reshape_layer in model.module.reshape_layers:
#                         num_params += count_parameters(reshape_layer)
#
#                     num_params += count_parameters(model.module.fusion_net)
#                     logger.info("Fusion Model Params: {}".format(num_params))
#
#                     genotype = model.module.genotype()
#                 else:
#                     num_params = 0
#                     # for reshape_layer in model.reshape_layers:
#                     #     num_params += count_parameters(reshape_layer)
#                     num_params += count_parameters(model.)
#                     num_params += count_parameters(model.fusion_net)
#                     logger.info("Fusion Model Params: {}".format(num_params))
#
#                     genotype = model.genotype()
#                 logger.info(str(genotype))
#
#                 if phase == 'train' and epoch_loss != epoch_loss:
#                     logger.info("Nan loss during training, escaping")
#                     model.eval()
#                     return best_f1
#
#                 if phase == 'dev' and status == 'search':
#                     if epoch_f1 > best_f1:
#                         best_f1 = epoch_f1
#                         best_genotype = copy.deepcopy(genotype)
#                         best_epoch = epoch
#                         # best_model_sd = copy.deepcopy(model.state_dict())
#
#                         if parallel:
#                             save(model.module, os.path.join(args.save, 'best', 'GLG2ADD.pt'))
#                         else:
#                             save(model, os.path.join(args.save, 'best', 'GLG2ADD.pt'))
#
#                         best_genotype_path = os.path.join(args.save, 'best', 'best_genotype1.pkl')
#                         save_pickle(best_genotype, best_genotype_path)
#
#                 if phase == 'test':
#                     if epoch_f1 > best_test_f1:
#                         best_test_f1 = epoch_f1
#                         best_test_genotype = copy.deepcopy(genotype)
#                         best_test_epoch = epoch
#
#                         if parallel:
#                             save(model.module, os.path.join(args.save, 'best', 'best_test_model.pt'))
#                         else:
#                             save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))
#
#                         best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
#                         save_pickle(best_test_genotype, best_test_genotype_path)
#
#             file_name = "epoch_{}".format(epoch)
#             file_name = os.path.join(args.save, "architectures", file_name)
#             plotter.plot(genotype, file_name, task='fashionIQ')
#
#             logger.info("Current best dev {} F1: {}, at training epoch: {}".format(f1_type, best_f1, best_epoch))
#             logger.info(
#                 "Current best test {} F1: {}, at training epoch: {}".format(f1_type, best_test_f1, best_test_epoch))
#
#         if best_f1 != best_f1 and num_epochs == 1 and cont_overloop < 1:
#             failsafe = True
#             logger.info('Recording a NaN F1, training for one more epoch.')
#         else:
#             failsafe = False
#
#         cont_overloop += 1
#
#     if best_f1 != best_f1:
#         best_f1 = 0.0
#
#     if status == 'search':
#         return best_f1, best_genotype
#     else:
#         return best_test_f1, best_test_genotype


def test_mmimdb_track_f1(model, criterion, dataloaders,
                         dataset_sizes, device,
                         parallel, logger, args,
                         f1_type='weighted', init_f1=0.0, th_fscore=0.3):
    best_test_genotype = None
    best_test_f1 = init_f1
    best_test_epoch = 0

    model.eval()  # Set model to evaluate mode
    list_preds = []
    list_label = []

    running_loss = 0.0
    running_f1 = init_f1
    phase = 'test'

    with tqdm(dataloaders[phase]) as t:
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            image, text, label = data['image'], data['text'], data['label']
            # device
            image = image.to(device)
            text = text.to(device)
            label = label.to(device)

            output = model((text, image))
            if isinstance(output, tuple):
                output = output[-1]

            _, preds = torch.max(output, 1)
            loss = criterion(output, label)
            preds_th = torch.sigmoid(output) > th_fscore
            # if phase == 'dev':
            list_preds.append(preds_th.cpu())
            list_label.append(label.cpu())

            # statistics
            running_loss += loss.item() * image.size(0)

            batch_pred_th = preds_th.data.cpu().numpy()
            batch_true = label.data.cpu().numpy()
            batch_f1 = f1_score(batch_pred_th, batch_true, average='samples')

            postfix_str = 'batch_loss: {:.03f}, batch_f1: {:.03f}'.format(loss.item(), batch_f1)
            t.set_postfix_str(postfix_str)
            t.update()

    epoch_loss = running_loss / dataset_sizes[phase]

    # if phase == 'dev':
    y_pred = torch.cat(list_preds, dim=0).numpy()
    y_true = torch.cat(list_label, dim=0).numpy()

    epoch_f1 = f1_score(y_true, y_pred, average=f1_type)

    logger.info('{} Loss: {:.4f}, {} F1: {:.4f}'.format(
        phase, epoch_loss, f1_type, epoch_f1))

    if parallel:
        num_params = 0
        for reshape_layer in model.module.reshape_layers:
            num_params += count_parameters(reshape_layer)

        num_params += count_parameters(model.module.fusion_net)
        logger.info("Fusion Model Params: {}".format(num_params))
        genotype = model.module.genotype()
    else:
        num_params = 0
        for reshape_layer in model.reshape_layers:
            num_params += count_parameters(reshape_layer)

        num_params += count_parameters(model.fusion_net)
        logger.info("Fusion Model Params: {}".format(num_params))
        genotype = model.genotype()
    logger.info(str(genotype))
    best_test_f1 = epoch_f1
    return best_test_f1