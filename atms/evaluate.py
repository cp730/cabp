import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pdb
import time
from tqdm import tqdm
import pickle
import copy
import torch

from option import parser, verify_input_args
import data
from vocab import Vocabulary
from artemis_model import ARTEMIS
from tirg_model import TIRG


def validate(model: object, args: object, vocab: object, phase: object, epoch: object, output_type: object = "metrics",
             max_retrieve: object = 50,
             split: object = 'val', architect: object = None, ) -> object:
    # Special case for CIRR: metrics are computed at the end, based on the rankings
    output_type_inpractice = "rankings" if args.data_name == "cirr" else output_type

    # Initializations
    results = []
    categories = args.name_categories if ("all" in args.categories) else args.categories.split(
        ' ')

    for category in categories:

        opt = copy.deepcopy(args)
        if args.study_per_category and (args.number_categories > 1):
            opt.categories = category  # 如果数据集有多个类别，data_loader的类别

        queries_loader, targets_loader = data.get_eval_loaders(opt, vocab, split)
        start = time.time()
        res = compute_and_process_compatibility_scores(queries_loader, targets_loader,
                                                       model,epoch, args, output_type_inpractice,
                                                       max_retrieve)
        end = time.time()
        print("\nProcessing time : ", end - start)

        results.append(res)

    if output_type == "metrics":
        message, val_mes = results_func(results, args)
        return message, val_mes
    return results


def update_arch_parameters(model: object, args: object, vocab: object, phase: object, epoch: object,
                           output_type: object = "metrics",
                           max_retrieve: object = 50,
                           split: object = 'val', architect: object = None, ) -> object:

    # Initializations
    results = []
    categories = args.name_categories if ("all" in args.categories) else args.categories.split(' ')
    for category in categories:

        # specify the category to be studied, if applicable
        opt = copy.deepcopy(args)
        if args.study_per_category and (args.number_categories > 1):
            opt.categories = category  # 如果数据集有多个类别，data_loader的类别

        queries_loader, targets_loader = data.get_eval_loaders(opt, vocab, split)
        # compute & process compatibility scores
        # no need to retain the computational graph and gradients
        start = time.time()
        update_architect(queries_loader, targets_loader, model, args, phase, epoch, architect)
        end = time.time()
        print("\nProcessing time : ", end - start)

    return results


def update_architect(data_loader_query, data_loader_target,
                     model, args, phase, epoch,  architect=None):
    with torch.no_grad():
        all_img_embs = compute_necessary_embeddings_img(data_loader_target, model, epoch,args)

    for data in tqdm(data_loader_query):
        img_src, txt, txt_len, img_src_ids, img_trg_ids, real_text, indices, img_trg = data

        if torch.cuda.is_available():
            txt, txt_len = txt.cuda(), txt_len.cuda()
            img_trg = img_trg.cuda()

        with torch.no_grad():
            txt_embs = model.get_txt_embedding(txt, real_text, txt_len)
            src_img_embs = all_img_embs[img_src_ids]

        if phase == "search":
            architect.step(src_img_embs, txt_embs, img_trg, epoch)


def compute_and_process_compatibility_scores(data_loader_query, data_loader_target,
                                             model,epoch, args, output_type="metrics",
                                             max_retrieve=50):
    nb_queries = len(data_loader_query.dataset)  # 所有的验证集数据数量 4034
    if output_type == "metrics":
        # return the rank of the best ranked correct target
        ret = torch.zeros(nb_queries, requires_grad=False)
    else:
        # return the top propositions for each query
        ret = torch.zeros(nb_queries, max_retrieve, requires_grad=False).int()
    with torch.no_grad():
        all_img_embs = compute_necessary_embeddings_img(data_loader_target, model,epoch, args)

    for data in tqdm(data_loader_query):

        img_src, txt, txt_len, img_src_ids, img_trg_ids, real_text, indices, img_trg = data  # img_trg_ids：batch中每一个query对应的target images的列表，有32个列表组成
        # pdb.set_trace()
        if torch.cuda.is_available():
            txt, txt_len = txt.cuda(), txt_len.cuda()
        with torch.no_grad():
            txt_embs = model.get_txt_embedding(txt, real_text, txt_len)
            src_img_embs = all_img_embs[img_src_ids]

        with torch.no_grad():
            batch_size = args.batch_size
            dataset_length = data_loader_target.dataset.__len__()
            iter_num = int(dataset_length / batch_size)  # 119
            for i in range(iter_num + 1):
                if i == iter_num:
                    k = dataset_length - batch_size * i
                    num = args.batch_size - k + 1
                    batch_img_trg_embs = all_img_embs[i * batch_size:-1]
                    _, d = batch_img_trg_embs.shape
                    extra = torch.zeros(num, d)
                    extra = extra.cuda()
                    batch_img_trg_embs = torch.cat([batch_img_trg_embs, extra], 0)
                else:
                    batch_img_trg_embs = all_img_embs[i * batch_size:i * batch_size + batch_size]

                batch_img_trg_embs = batch_img_trg_embs.cuda()
                batch_scores = model.compute_score(src_img_embs, txt_embs, batch_img_trg_embs)

                if i == 0:
                    all_scores = batch_scores
                else:
                    if i == iter_num:
                        temp = torch.zeros([batch_scores.shape[0], k])
                        for j in range(batch_scores.shape[0]):
                            temp[j] = batch_scores[j][:k]
                        temp = temp.cuda()
                        all_scores = torch.cat([all_scores, temp], 1)
                    else:
                        all_scores = torch.cat([all_scores, batch_scores], 1)

            for i, index in enumerate(indices):  # for each ref-img in this batch

                img_src_id = img_src_ids[i]
                GT_indices = img_trg_ids[i]

                cs = all_scores[i]
                cs = cs.squeeze(0)
                cs[img_src_id] = float('-inf')

                cs_sorted_ind = cs.sort(descending=True)[1]  # 根据score从高到低的排名，得到对应的image的id

                if output_type == "metrics":
                    ret[index] = get_rank_of_GT(cs_sorted_ind, GT_indices)[0]  # 返回所有target image中，rank最高的那张的的排名
                    # ret[index] ：id为index的这张query，他对应的所有candidates中，score最高的那张candidate再所有图片中的score中的排名。
                else:
                    ret[index, :max_retrieve] = cs_sorted_ind[:max_retrieve].cpu().int()

    return ret


def compute_necessary_embeddings_img(data_loader_target: object, model: object,epoch:int, args: object) -> object:
    img_trg_embs = None
    img_trg_embs7x7 = None

    for data in tqdm(data_loader_target):

        # Get target data
        img_trg, _, indices = data
        indices = torch.tensor(indices)
        if torch.cuda.is_available():
            img_trg = img_trg.cuda()
        # Compute embedding
        img_trg_emb = model.get_image_embedding(img_trg,epoch)  # 32*2048*7*7  and 32 *512

        # Initialize the output embeddings if not done already
        if img_trg_embs is None:
            emb_sz = [len(data_loader_target.dataset), args.embed_dim]
            img_trg_embs = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False)
            if torch.cuda.is_available():
                img_trg_embs = img_trg_embs.cuda()

        # Preserve the embeddings by copying them
        if torch.cuda.is_available():
            img_trg_embs[indices] = img_trg_emb



        else:
            img_trg_embs[indices] = img_trg_emb.cpu()

    return img_trg_embs


# 只用于amazon数据集的验证集
def compute_and_process_compatibility_scores_dev(data_loader_query, data_loader_target,
                                                 model, args, output_type="metrics",
                                                 max_retrieve=50):
    nb_queries = len(data_loader_query.dataset)  # 所有的验证集数据数量 4034

    # Initialize output
    if output_type == "metrics":
        # return the rank of the best ranked correct target
        ret = torch.zeros(nb_queries, requires_grad=False)
    else:
        # return the top propositions for each query
        ret = torch.zeros(nb_queries, max_retrieve, requires_grad=False).int()

    # Pre-compute image embeddings (includes all target & reference images) --- target image embedding
    all_img_embs = compute_necessary_embeddings_img(data_loader_target, model, args)  # 按self.image_id2name中的顺序

    # Compute and process compatibility scores (process by batch)
    for data in tqdm(data_loader_query):

        # Get query data
        _, txt, txt_len, img_src_ids, img_trg_ids, _, indices, img_src_pids, img_trg_pids = data  # img_trg_ids：batch中每一个query对应的target images的列表，有32个列表组成
        if torch.cuda.is_available():
            txt, txt_len = txt.cuda(), txt_len.cuda()

        txt_embs = model.get_txt_embedding(txt, txt_len)  # 32 * 512

        # Process each query of the batch one by one
        for i, index in enumerate(indices):  # indices:这个batch的data数据的索引,size=32。 index：此时这张query图片的idx

            # Select data related to the current query
            txt_emb = txt_embs[i]
            img_src_id = img_src_ids[i]
            GT_indices = img_trg_ids[i]  # 列表，size=32,一个元素为该batch中的每一个query对应的candidate的index组成的列表
            img_src_emb = all_img_embs[img_src_id]

            # Compute compatibility scores between the query and each candidate target
            cs = model.get_compatibility_from_embeddings_one_query_multiple_targets(
                img_src_emb, txt_emb, all_img_embs)

            # pdb.set_trace()
            # Remove the source image from the ranking
            cs[img_src_id] = float('-inf')

            # Rank targets
            cs_sorted_ind = cs.sort(descending=True)[1]  # 根据score从高到低的排名，得到对应的image的id
            '''
                img_src_id:query image pid
                1.找到对应的 source_pid那一行
                2.根据cs_sorted_scores和GT_indices修改
                3.写入一个新的json文件
                找到每一个candidate的分数
            '''
            modify_annotations = data_loader_query.dataset.annotations
            assert modify_annotations[index]["source_pid"] == img_src_pids[
                i], "query image does not match image in the json file"
            candidates_per_query_image = modify_annotations[index]["candidates"]  # [{},{},{},{}...]

            for j in range(len(candidates_per_query_image)):  # for each candidate related to this query
                id = img_trg_pids[i].index(candidates_per_query_image[j]["candidate_pid"])
                candidates_per_query_image[j]['score'] = float(cs[GT_indices[id]])

            modify_annotations[index]["candidates"] = candidates_per_query_image

            # Store results
            if output_type == "metrics":
                ret[index] = get_rank_of_GT(cs_sorted_ind, GT_indices)[0]  # 返回所有target image中，rank最高的那张的的排名
            # ret[index] ：id为index的这张query，他对应的所有candidates中，score最高的那张candidate再所有图片中的score中的排名。
            else:
                ret[index, :max_retrieve] = cs_sorted_ind[:max_retrieve].cpu().int()

    return ret, modify_annotations


def get_rank_of_GT(sorted_ind, GT_indices):
    """
    Get the rank of the best ranked correct target provided the target ranking
    (targets are identified by indices). Given two acceptable correct targets of
    respective indices x and y, if the target of index x has a better rank than
    the target of index y, then the returned value for `rank_of_GT ` is the rank
    of the target of index x, and the value of `best_GT` is x.

    Input:
        sorted_ind: tensor of size (number of candidate targets), containing the
            candidate target indices sorted in decreasing order of relevance with
            regard to a given query.
        GT_indices: list of correct target indices for a given query.

    Output:
        rank_of_GT: rank of the best ranked correct target, if it is found
            (+inf is returned otherwise)
        best_GT: index of the best ranked correct target

    """
    rank_of_GT = float('+inf')
    best_GT = None
    for GT_index in GT_indices:
        tmp = torch.nonzero(sorted_ind == GT_index)
        if tmp.size(0) > 0:  # the GT_index was found in the ranking
            tmp = tmp.item()
            if tmp < rank_of_GT:
                rank_of_GT = tmp
                best_GT = GT_index
    return rank_of_GT, best_GT


def get_recall(rank_of_GT, K):
    return 100 * (rank_of_GT < K).float().mean()


def results_func(results, args):
    """
    Compute metrics over the dataset and present them properly.
    The result presentation and the computation of the metric might depend
    on particular options/arguments (use the `args`).

    Input:
        results: list containing one tensor per data category (or just one
            tensor if the dataset has no particular categories). The tensor is
            of size (number of queries) and contains the rank of the best ranked
            correct target.
        args: argument parser from option.py

    Ouput:
        message: string message to print or to log
        val_mes: measure to monitor validation (early stopping...)
    """

    nb_categories = len(results)

    # --- Initialize a dictionary to hold the results to present
    H = {"r%d" % k: [] for k in args.recall_k_values}
    H.update({"medr": [], "meanr": [], "nb_queries": []})

    # --- Iterate over categories
    for i in range(nb_categories):
        # get measures about the rank of the best ranked correct target
        # for category i
        for k in args.recall_k_values:
            H["r%d" % k].append(get_recall(results[i], k))
        H["medr"].append(torch.floor(torch.median(results[i])) + 1)
        H["meanr"].append(results[i].mean() + 1)
        H["nb_queries"].append(len(results[i]))

    # --- Rearrange results (aggregate category-specific results)
    H["avg_per_cat"] = [sum([H["r%d" % k][i] for k in args.recall_k_values]) / len(args.recall_k_values) for i in
                        range(nb_categories)]
    val_mes = sum(H["avg_per_cat"]) / nb_categories
    H["nb_total_queries"] = sum(H["nb_queries"])
    for k in args.recall_k_values:
        H["R%d" % k] = sum([H["r%d" % k][i] * H["nb_queries"][i] for i in range(nb_categories)]) / H["nb_total_queries"]
    H["rsum"] = sum([H["R%d" % k] for k in args.recall_k_values])
    H["med_rsum"] = sum(H["medr"])
    H["mean_rsum"] = sum(H["meanr"])

    # --- Present the results of H in a single string message
    message = ""

    # multiple-category case: print category-specific results
    if nb_categories > 1:
        categories = args.name_categories if ("all" in args.categories) else args.categories
        cat_detail = ", ".join(["%.2f ({})".format(cat) for cat in categories])

        message += ("\nMedian rank: " + cat_detail) % tuple(H["medr"])
        message += ("\nMean rank: " + cat_detail) % tuple(H["meanr"])
        for k in args.recall_k_values:
            message += ("\nMetric R@%d: " + cat_detail) \
                       % tuple([k] + H["r%d" % k])

        # for each category, average recall metrics over the different k values
        message += ("\nRecall average: " + cat_detail) % tuple(H["avg_per_cat"])

        # for each k value, average recall metrics over categories
        # (remove the normalization per the number of queries)
        message += "\nGlobal recall metrics: {}".format( \
            ", ".join(["%.2f (R@%d)" % (H["R%d" % k], k) \
                       for k in args.recall_k_values]))

    # single category case
    else:
        message += "\nMedian rank: %.2f" % (H["medr"][0])
        message += "\nMean rank: %.2f" % (H["meanr"][0])
        for k in args.recall_k_values:
            message += "\nMetric R@%d: %.2f" % (k, H["r%d" % k][0])

    message += "\nValidation measure: %.2f\n" % (val_mes)
    '''
    e.g.
     message:'\nMedian rank: 15.00\nMean rank: 71.36\nMetric R@1: 6.58\nMetric R@10: 41.74\nMetric R@50: 76.12\nValidation measure: 41.48\n'
     val_mes:tensor(41.4807)
    '''
    # pdb.set_trace()
    return message, val_mes


def load_model(args):
    # Load vocabulary
    vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')
    assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
    vocab = pickle.load(open(vocab_path, 'rb'))

    # Setup model
    if args.model_version == "TIRG":
        model = TIRG(vocab.word2idx, args)
    else:
        # model version is ARTEMIS or one of its ablatives
        model = ARTEMIS(vocab.word2idx, args)
    print("Model version:", args.model_version)

    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    # Load model weights
    if args.ckpt:

        # load checkpoint
        assert os.path.isfile(args.ckpt), f"(ckpt) File not found: {args.ckpt}"
        print(f"Loading file {args.ckpt}.")

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args.ckpt)['model'])
        else:
            state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['model']
            model.load_state_dict(state_dict)
        print("Model: resume from provided state.")

    return args, model, vocab


if __name__ == '__main__':

    args = verify_input_args(parser.parse_args())

    # Load model & vocab
    args, model, vocab = load_model(args)

    start = time.time()
    with torch.no_grad():
        message, _ = validate(model, args, vocab, split=args.studied_split)
    print(message)

    # save printed message on .txt file
    basename = ""
    if os.path.basename(args.ckpt) != "model_best.pth":
        basename = "_%s" % os.path.basename(os.path.basename(args.ckpt))
    save_txt = os.path.abspath(os.path.join(args.ckpt, os.path.pardir, os.path.pardir, 'eval_message%s.txt' % basename))
    with open(save_txt, 'a') as f:
        f.write(args.data_name + ' ' + args.studied_split + ' ' + args.exp_name + '\n######')
        f.write(message + '\n######\n')

    end = time.time()
    print("\nProcessing time : ", end - start)
