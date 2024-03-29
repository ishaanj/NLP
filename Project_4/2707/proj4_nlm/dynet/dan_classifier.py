"""Simple deep averaging net classifier"""
import os
import pickle
from time import clock

import dynet_config
dynet_config.set(random_seed=42, autobatch=1, mem=1024)
dynet_config.set_gpu(True)
import dynet as dy
from matplotlib import pyplot as plt
from matplotlib import patches

MAX_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIM = 32
VOCAB_SIZE = 4748


def make_batches(data, batch_size):
    batches = []
    batch = []
    for pair in data:
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []

        batch.append(pair)

    if batch:
        batches.append(batch)

    return batches


class DANClassifier(object):
    def __init__(self, params, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = params.add_lookup_parameters((vocab_size, hidden_dim))

        self.W_hid = params.add_parameters((hidden_dim, hidden_dim))
        self.b_hid = params.add_parameters((hidden_dim))

        self.w_clf = params.add_parameters((1, hidden_dim))
        self.b_clf = params.add_parameters((1))

    def _predict(self, batch, train=True):

        # load the network parameters
        W_hid = dy.parameter(self.W_hid)
        b_hid = dy.parameter(self.b_hid)
        w_clf = dy.parameter(self.w_clf)
        b_clf = dy.parameter(self.b_clf)

        probas = []
        # predict the probability of positive sentiment for each sentence
        for _, sent in batch:

            sent_embed = [dy.lookup(self.embed, w) for w in sent]
            sent_embed = [dy.dropout(w,0.5) for w in sent_embed]
            sent_embed = dy.average(sent_embed)

            # hid = tanh(b + W * sent_embed)
            # but it's faster to use affine_transform in dynet
            hid = dy.affine_transform([b_hid, W_hid, sent_embed])
            hid = dy.tanh(hid)

            y_score = dy.affine_transform([b_clf, w_clf, hid])
            y_proba = dy.logistic(y_score)
            probas.append(y_proba)

        return probas

    def batch_loss(self, sents, train=True):
        probas = self._predict(sents, train)

        # we pack all predicted probas into one vector of length batch_size
        probas = dy.concatenate(probas)

        # we make a dynet vector out of the true ys
        y_true = dy.inputVector([y for y, _ in sents])

        # classification loss: we use the logistic loss
        # this function automatically sums over all entries.
        total_loss = dy.binary_log_loss(probas, y_true)

        return total_loss

    def num_correct(self, sents):
        probas = self._predict(sents, train=False)
        probas = [p.value() for p in probas]
        y_true = [y for y, _ in sents]

        correct = 0
        predicted = [p > 0.5 for p in probas]
        correct+=  sum([predicted[i] == y_true[i] for i in range(len(predicted))])
        # i = 0
        # for i in range(len(predicted)):
        #     if predicted[i] == y_true[i]:
        #         correct+=1

        # FIXME: count the number of correct predictions here
        return correct


if __name__ == '__main__':

    with open(os.path.join('..\processed', 'train_ix.pkl'), 'rb') as f:
        train_ix = pickle.load(f)

    with open(os.path.join('..\processed', 'valid_ix.pkl'), 'rb') as f:
        valid_ix = pickle.load(f)

    with open(os.path.join('..\processed', 'test_ix.pkl'), 'rb') as f:
        test_ix = pickle.load(f)

    # initialize dynet parameters and learning algorithm
    params = dy.ParameterCollection()
    trainer = dy.AdadeltaTrainer(params)
    clf = DANClassifier(params, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    clf.embed.populate('embeds_baseline_lm_3gram',"/embed")
    train_batches = make_batches(train_ix, BATCH_SIZE)
    # valid_batches = make_batches(valid_ix, BATCH_SIZE)
    test_batches = make_batches(test_ix, BATCH_SIZE)

    red_patch = patches.Patch(color='green', label='Test')

    for it in range(MAX_EPOCHS):
        tic = clock()

        # iterate over all training batches, accumulate loss.
        total_loss = 0
        for batch in train_batches:
            dy.renew_cg()
            loss = clf.batch_loss(batch, train=True)
            loss.backward()
            trainer.update()
            total_loss += loss.value()

        # iterate over all validation batches, accumulate # correct pred.
        valid_acc = 0
        for batch in test_batches:
            dy.renew_cg()
            valid_acc += clf.num_correct(batch)

        valid_acc /= len(valid_ix)

        toc = clock()

        print(("Epoch {:3d} took {:3.1f}s. "
               "Train loss: {:8.5f} "
               "Valid accuracy: {:8.2f}").format(
            it,
            toc - tic,
            total_loss / len(train_ix),
            valid_acc * 100
            ))
        plt.scatter(x=it, y=valid_acc*100, color='green', s=100)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.show()
