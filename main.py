
import argparse
import os
import torch
import torch.optim as optim
import thirdparty.data_reader_tf as data_reader_tf

from nlm.utils import disp_hparam
from nlm.model import CNLM
from nlm.train import train_eval

#FILE_NAME_LIST = ['_train.txt.prepro', '_valid.txt.prepro']
FILE_NAME_LIST = ['train.txt', 'valid.txt', 'test.txt']

def arg_parser():
    tips  = "######################################################################\n"
    tips += "# Function            : the baseline CNLM system for paper           #\n"
    tips += "# Author              : AmbyerHan                                    #\n"
    tips += "# Email               : amberhan0301@163.com                         #\n"
    tips += "# Date                : 12/22/2017                                   #\n"
    tips += "# Last Modified in    : Neu NLP lab., Shenyang                       #\n"
    tips += "######################################################################\n"

    parser = argparse.ArgumentParser(description=tips, formatter_class=argparse.RawDescriptionHelpFormatter)

    # data params
    parser.add_argument('--data_dir', '-data', metavar = 'PATH', dest = 'ddir', default = './data_dir/',
                        help = 'todo')
    parser.add_argument('--train_dir', '-train', metavar = 'PATH', dest = 'tdir', default = './train_dir/',
                        help = 'todo')
    parser.add_argument('--model_dir', '-model', metavar = 'PATH', dest = 'mdir', default = './model_dir/',
                        help = 'todo')

    # model params
    parser.add_argument('--rnn_hidden', '-hdim', type = int, dest = 'hdim', default = 650,
                        help = 'todo')
    parser.add_argument('--char_embed_size', '-cedim', type = int, dest = 'cedim', default = 15,
                        help = 'todo')
    parser.add_argument('--word_embed_size', '-wedim', type = int, dest = 'wedim', default = 128,
                        help = 'todo')
    parser.add_argument('--high_layers', '-hlayers', type = int, dest = 'hlayers', default = 2,
                        help = 'todo')
    parser.add_argument('--rnn_layers', '-rlayers', type = int, dest = 'rlayers', default = 2,
                        help = 'todo')
    parser.add_argument('--dropout', '-drop', type = float, dest = 'drop', default = 0.5,
                        help = 'todo')
    parser.add_argument('--kernels_width', '-ker_wid', dest = 'ker_wid', default = '[1, 2, 3, 4, 5, 6, 7]',
                        help = 'todo')
    parser.add_argument('--kernels_nums', '-ker_num', dest = 'ker_num', default = '[50, 100, 150, 200, 200, 200, 200]',
                        help = 'todo')
    parser.add_argument('--param_init', '-pinit', type = float, dest = 'param_init', default = 0.05,
                        help = 'todo')

    # optimize params
    parser.add_argument('--learning_rate_decay', '-lr_decay', type = float, dest = 'lr_decay', default = 0.5,
                        help = 'todo')
    parser.add_argument('--decay_when', '-decay_when', type = float, dest = 'decay_when', default = 1.0,
                        help = 'todo')
    parser.add_argument('--learning_rate', '-lr', type = float, dest = 'lr', default = 1.0,
                        help = 'todo')
    parser.add_argument('--batch_size', '-batch', type = int, dest = 'batch', default = 20,
                        help = 'todo')
    parser.add_argument('--max_epoch', '-mepoch', type = int, dest = 'mepoch', default = 25,
                        help = 'todo')
    parser.add_argument('--max_steps', '-msteps', type = int, dest = 'msteps', default = 10000,
                        help = 'todo')
    parser.add_argument('--max_sent_len', '-maxlen', type = int, dest = 'maxlen', default = 35,
                        help = 'todo')
    parser.add_argument('--max_word_len', '-maxwdlen', type = int, dest = 'maxwdlen', default = 30,
                        help = 'todo')
    parser.add_argument('--clip', '-clip', type = float, dest = 'clip', default = 5.0,
                        help = 'todo')

    # book keeping
    parser.add_argument('--seed', '-seed', type = int, dest = 'seed', default = 3435,
                        help = 'todo')
    parser.add_argument('--display_frequency', '-dfreq', type = int, dest = 'dfreq', default = 100,
                        help = 'todo')
    parser.add_argument('--eval_frequency', '-efreq', type = int, dest = 'efreq', default = 1000,
                        help = 'todo')
    parser.add_argument('--use_cuda', '-cuda', dest = 'cuda', action = 'store_true',
                        help = 'todo')
    parser.add_argument('--save_model', '-save', dest = 'save', action = 'store_true',
                        help = 'todo')

    disp_hparam(vars(parser.parse_args())) # todo
    return (parser.parse_args())

def str_to_list(str):
    """ the str look like '[1, 2, 3, ..., n]' and return the correspponding list """
    tmp = str[1: -1]
    tmp = tmp.split(",")

    return [int(t.strip()) for t in tmp]

def i2s_my(reader, dict, file):

    fout = open("./debug/%s.txt" % file, "w")
    cnt = 0
    for x, y in reader.iter():
        if cnt > 200:
            break
        cnt += 1
        # x <batch, maxlen>
        for s, t in zip(x, y):
            # per batch
            sln, tln = [], []
            for si, ti in zip(s, t):
                sln.append(dict.token(si))
                tln.append(dict.token(ti))
            fout.write("%40s\t-\t%40s\n" % (" ".join(sln), " ".join(tln)))
        fout.write("\t\t------batch--------\n")

    fout.close()

def main(args):
    if not os.path.exists(args.tdir):
        os.mkdir(args.tdir)
        print('[Create] %s' % args.tdir)

    if args.cuda:
        assert torch.cuda.is_available()

    word_vocab, char_vocab, \
    word_tensor, char_tensor, \
    actual_maxwdlen = data_reader_tf.load_data(args.ddir, 65, FILE_NAME_LIST)

    args.wvsize = word_vocab.size

    train_reader = data_reader_tf.DataReader(word_tensor[FILE_NAME_LIST[0]], char_tensor[FILE_NAME_LIST[0]],
                                            args.batch, args.maxlen)
    valid_reader = data_reader_tf.DataReader(word_tensor[FILE_NAME_LIST[1]], char_tensor[FILE_NAME_LIST[1]],
                                              args.batch, args.maxlen)
    test_reader  = data_reader_tf.DataReader(word_tensor[FILE_NAME_LIST[2]], char_tensor[FILE_NAME_LIST[2]],
                                              args.batch, args.maxlen)

    args.cvsize = char_vocab.size
    args.wvsize = word_vocab.size
    #i2s_my(train_reader, word_vocab, 'my_rnn')

    if args.seed:
        torch.manual_seed(args.seed)

    ker_sizes = str_to_list(args.ker_wid)
    ker_feats = str_to_list(args.ker_num)
    c2w = sum(ker_feats)
    nlm = CNLM(cvsize = args.cvsize,
               cedim = args.cedim,
               wvsize = args.wvsize,
               wedim = args.wedim,
               cnn_size = c2w,
               hdim = args.hdim,
               kernel_sizes = ker_sizes,
               kernel_features = ker_feats,
               nhlayers = args.hlayers,
               nrlayers = args.rlayers,
               tie_weight = False,
               droprate = args.drop)

    print(nlm)

    if args.cuda:
        nlm.cuda()

    opt = optim.SGD(nlm.parameters(), lr = args.lr)

    train_eval(nlm, opt, train_reader, valid_reader, test_reader, args)


if __name__ == "__main__":
    args = arg_parser()

    main(args)
    print('done')
