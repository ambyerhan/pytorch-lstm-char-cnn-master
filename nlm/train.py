
import torch
import time
import math

from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.nn import functional as F


def repackage_hidden(h): # todo
    """ wraps hidden states in new Variable, to detach them from their history """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model, opt, lr, reader, e, gstep, args):
    ave_loss = 0.
    ave_train_loss = 0.
    reachMaxStep = False
    b_t_b = time.time()
    hidden_state = model.init_hidden(args.batch)
    for x, y in reader.iter():
        model.train()
        x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y.T)) # <batch, maxlen, wdlen>
        x = torch.transpose(x, 0, 1)
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        hidden_state = repackage_hidden(hidden_state)
        model.zero_grad() # <maxlen, batch, wdlen>
        logits, hidden_state = model(x, hidden_state)  # step1 todo model_output <batch, maxlen, vsize>, y <batch, maxlen>
        output = logits.contiguous().view(-1, args.wvsize)
        labels = y.contiguous().view(-1)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm = args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        ave_loss += loss.data[0]
        ave_train_loss += 0.05 * (loss.data[0] - ave_train_loss)
        gstep += 1

        # do some verbose
        if gstep % args.dfreq == 0:
            ave_loss /= args.dfreq
            cost_time = (time.time() - b_t_b)
            print("[LOSS] -%02d- global_step = %5d, loss/ppl = %.3f/%.3f, speed = %.3f (step/sec), time = %.3f (secs)" % (e, gstep, ave_loss, math.exp(ave_loss), (args.dfreq / cost_time), cost_time))
            b_t_b = time.time()
            ave_loss = 0.

        if gstep > args.msteps:
            print("[INFO] Reached maximum global steps!")
            reachMaxStep = True
            break

    return ave_train_loss, gstep, reachMaxStep


def eval(model, reader, e, gstep, args):
    print("[VALID] Evaluating the model on the validation set")
    ave_loss = 0.0
    hidden_state = model.init_hidden(args.batch)
    cnt = 0
    for x, y in reader.iter():
        model.eval()
        cnt += 1
        x, y = Variable(torch.from_numpy(x), volatile = True), Variable(torch.from_numpy(y.T), volatile = True)
        x = torch.transpose(x, 0, 1)
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        hidden_state = repackage_hidden(hidden_state)
        logits, hidden_state = model(x, hidden_state)
        output = logits.contiguous().view(-1, args.wvsize)
        labels = y.contiguous().view(-1)
        loss = F.cross_entropy(output, labels)

        ave_loss += loss.data[0]

    ave_loss /= cnt

    return ave_loss, hidden_state


def train_eval(nlm, opt, train_reader, valid_reader, test_reader, args):
    gstep = 0
    total_time = 0.
    best_valid_loss = None
    lr = args.lr
    min = [10000, 10000, 10000, 10000] # t_loss, t_ppl, v_loss, v_ppl
    for e in range(1, args.mepoch + 1):
        print("\n[EPOCH] Start of epoch <%d>" % e)
        e_t_b = time.time()

        tloss, gstep, reachMaxStep = train(nlm, opt, lr, train_reader, e, gstep, args)
        vloss, hidden_state = eval(nlm, valid_reader, e, gstep, args)

        print("\t> The average train loss/ppl = %.3f/%.3f" % (tloss, math.exp(tloss)))
        print("\t> The average valid loss/ppl = %.3f/%.3f" % (vloss, math.exp(vloss)))
        if min[2] > vloss:
            min[0] = tloss
            min[1] = math.exp(tloss)
            min[2] = vloss
            min[3] = math.exp(vloss)

        if True:
            model_file_name = ""
            if args.save:
                model_file_name = args.mdir + ('model_ep%02d_gs%05d_.pt' % (e, gstep))
                print("[SAVE] Saving the model as %s" % model_file_name)
                torch.save(nlm, model_file_name)
                print("[SAVE] Saved the model...")

            """ whether need to decay """
            if best_valid_loss is not None and (math.exp(best_valid_loss) - math.exp(vloss) < args.decay_when):
                print("[WARNING]")
                print("[DECAY] Decaying the learning rate")
                #cur_lr = opt.defaults['lr']
                cur_lr = lr
                print("[DECAY] The cur_lr_rate: %f" % cur_lr)
                #new_lr = opt.defaults['lr'] * args.lr_decay
                lr = lr * args.lr_decay
                if lr < 1.e-5:
                    print("[WARNING] The learning rate is tool small, stopping now!")
                    reachMaxStep = True
                    break
                #opt.defaults['lr'] = new_lr
                print("[DECAY] The new_lr_rate: %f" % lr)

            else:
                best_valid_loss = vloss

        epoch_time = time.time() - e_t_b
        total_time += epoch_time
        print("[EPOCH] End of an epoch, costs %.3f mins" % (epoch_time / 60))
        if reachMaxStep:
            break
    print("[DONE] Training done, costs %.3f secs totally!" % (total_time))
    print("=" * 89)
    print("=" * 89)
    print("\t> The minimum train loss = %f, train ppl = %f" % (min[0], min[1]))
    print("\t> The minimum valid loss = %f, valid ppl = %f" % (min[2], min[3]))
	

    print('=' * 89)
    test_loss, _ = eval(nlm, test_reader, 0, gstep, args)
    print("\t> The test loss = %f, test ppl = %f" % (test_loss, math.exp(test_loss)))
    print('=' * 89) 
