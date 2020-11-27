"""
This script provides an exmaple to wrap UER-py for classification (cross validation).
"""
import torch
import random
import argparse
import collections
import torch.nn as nn
import numpy as np
import gc
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.layers.embeddings import *
from uer.encoders.bert_encoder import *
from uer.encoders.rnn_encoder import *
from uer.encoders.birnn_encoder import *
from uer.encoders.cnn_encoder import *
from uer.encoders.attn_encoder import *
from uer.encoders.gpt_encoder import *
from uer.encoders.mixed_encoder import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from run_classifier import *
from data_loader import multi_cls_loader
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def evaluate_sk(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    y = []
    y_ = []
    args.model.eval()

    for i, (src_batch, tgt_batch,  seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            loss, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for p, g in zip(pred, gold):
            y.append(g.cpu().numpy())
            y_.append(p.cpu().numpy())
    
    f1 = f1_score(y, y_, average='macro')
    precision = precision_score(y, y_, average='macro')
    recall = recall_score(y, y_, average='macro')
    accuracy = accuracy_score(y, y_)

    return f1, precision, recall, accuracy

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--train_features_path", type=str, required=True,
                        help="Path of the train features for stacking.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", "synt", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                                              default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3" ], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Cross validation options.
    parser.add_argument("--folds_num", type=int, default=5,
                        help="The number of folds for cross validation.")
    
    parser.add_argument('--test_run', type=bool, default=False)
    
    args = parser.parse_args()

    print('Starting')
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)
    # change for short test run
    if args.test_run:
        args.epoch = 1
    # Build tokenizer.
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Training phase. Count the number of labels. 
    dataset, labels, groups, args.labels_num = multi_cls_loader(args, args.train_path)
    print('Total sample number: {}'.format(len(dataset)))
    instances_num = len(dataset)
    batch_size = args.batch_size


    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    train_features = []
    train_features_idxs = []

    total_loss, result = 0., 0.
    acc, marco_f1 = 0., 0.

    print('StratifiedKFold for CV.')
    kf = StratifiedKFold(n_splits=args.folds_num, shuffle=True, random_state=args.seed)
    dataset = np.array(dataset, dtype=object)
    labels = np.array(labels)
    groups = np.array(groups)
    fold_id = 0
    for train_index, test_index in kf.split(dataset, labels, groups=groups):
        X_train, X_val = dataset[train_index], dataset[test_index]
        y_train, y_val = labels[train_index], labels[test_index]
        g_train, g_val = groups[train_index], groups[test_index]
        # Build classification model.
        model = Classifier(args)

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(args.device)
        load_or_initialize_parameters(args, model)
        optimizer, scheduler = build_optimizer(args, model)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)
            args.amp = amp
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        args.model = model

        trainset = X_train
        
        train_src = torch.LongTensor([example[0] for example in trainset])
        train_tgt = torch.LongTensor([example[1] for example in trainset])
        train_seg = torch.LongTensor([example[2] for example in trainset])

        if args.soft_targets:
            train_soft_tgt = torch.FloatTensor([example[3] for example in trainset])
        else:
            train_soft_tgt = None

        devset = X_val

        dev_src = torch.LongTensor([example[0] for example in devset])
        dev_tgt = torch.LongTensor([example[1] for example in devset])
        dev_seg = torch.LongTensor([example[2] for example in devset])
        dev_soft_tgt = None

        for epoch in range(1, args.epochs_num+1):
            model.train()
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, train_src, train_tgt, train_seg, train_soft_tgt)):    
                loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    print("Fold id: {}, Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(fold_id, epoch, i+1, total_loss / args.report_steps))
                    total_loss = 0.

        model.eval()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, dev_src, dev_tgt, dev_seg, dev_soft_tgt)): 
             src_batch = src_batch.to(args.device)
             seg_batch = seg_batch.to(args.device)
             with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)
             prob = nn.Softmax(dim=1)(logits)
             prob = prob.cpu().numpy().tolist()        
             train_features.extend(prob)
        for i in g_val:
            train_features_idxs.append(i)

        output_model_name = ".".join(args.output_model_path.split(".")[:-1])
        output_model_suffix = args.output_model_path.split(".")[-1]
        save_model(model, output_model_name+"-fold_"+str(fold_id)+"."+output_model_suffix)
        f1, precision, recall, accuracy = evaluate_sk(args, devset)
        print(
                u'val_f1: %.5f, 'u'val_precision: %.5f, 'u'val_recall: %.5f, 'u'val_accuracy: %.5f\n' %
                (f1, precision, recall, accuracy,)
            )
        args.model = None
        gc.collect()
        del model
        torch.cuda.empty_cache()
        fold_id += 1
        
    train_features = np.array(train_features)
    train_features_idxs = np.array(train_features_idxs)
    np.save(args.train_features_path, train_features)
    np.save(args.train_features_path+'idx_order', train_features_idxs)



if __name__ == "__main__":
    main()
