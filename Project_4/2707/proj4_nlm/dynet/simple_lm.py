"""Simplest possible neural language model:
    use word w_i to predict word w_(i + 1)
"""
import os
import pickle
from time import clock
from math import exp
from matplotlib import pyplot as plt
from matplotlib import patches
import dynet_config
dynet_config.set(random_seed=42, autobatch=1)
dynet_config.set_gpu(True)
import dynet as dy

N_GRAM_SIZE = 2

MAX_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIM = 32
USE_UNLABELED = False
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


class SimpleNLM(object):

    def __init__(self, params, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embed = params.add_lookup_parameters((vocab_size, hidden_dim))

        self.W_hid = params.add_parameters((hidden_dim, N_GRAM_SIZE * hidden_dim))
        self.b_hid = params.add_parameters((hidden_dim))

        self.W_out = params.add_parameters((vocab_size, hidden_dim))

    def batch_loss(self, batch, train=True):

        # load the parameters
        W_hid = dy.parameter(self.W_hid)
        b_hid = dy.parameter(self.b_hid)

        W_out = dy.parameter(self.W_out)

        losses = []
        for _, sent in batch:
            for i in range(N_GRAM_SIZE, len(sent)):

                prev_words = []
                for j in range(N_GRAM_SIZE):
                    prev_words.append(sent[j+i-1])
                curr_word_ix = sent[i]

                ctx = dy.lookup(self.embed, prev_words[0])
                for ix, k in enumerate(prev_words):
                    if ix == 0:
                        continue
                    ctx_2 = dy.lookup(self.embed, k)
                    ctx = dy.concatenate([ctx, ctx_2])

                # hid is the hidden layer output, size=hidden_size
                # compute b_hid + W_hid * ctx, but faster
                hid = dy.affine_transform([b_hid, W_hid, ctx])
                hid = dy.tanh(hid)

                # out is the prediction of the next word, size=vocab_size
                out = W_out * hid

                # Intepretation: The model estimates that
                # log P(curr_word=k | prev_word) ~ out[k]
                # in other words,
                # P(curr_word=k | prev_word) = exp(out[k]) / sum_j exp(out[j])
                #                            = softmax(out)[k]

                # We want to maximize the probability of the correct word.
                # (equivalently, minimize the negative log-probability)

                loss = dy.pickneglogsoftmax(out, curr_word_ix)
                losses.append(loss)

        # esum simply adds up the expressions in the list
        return dy.esum(losses)


if __name__ == '__main__':

    red_patch = patches.Patch(color='red', label='Training')
    blue_patch = patches.Patch(color='blue', label='Validation')
    plt.legend(handles=[red_patch, blue_patch])

    with open(os.path.join('..\processed', 'train_ix.pkl'), 'rb') as f:
        train_ix = pickle.load(f)

    if USE_UNLABELED:
        with open(os.path.join('..\processed', 'unlab_ix.pkl'), 'rb') as f:
            train_ix.extend(pickle.load(f))


    with open(os.path.join('..\processed', 'valid_ix.pkl'), 'rb') as f:
        valid_ix = pickle.load(f)

    # initialize dynet parameters and learning algorithm
    params = dy.ParameterCollection()
    trainer = dy.AdadeltaTrainer(params)
    lm = SimpleNLM(params, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)

    train_batches = make_batches(train_ix, batch_size=BATCH_SIZE)
    valid_batches = make_batches(valid_ix, batch_size=BATCH_SIZE)

    n_train_words = sum(len(sent) for _, sent in train_ix)
    n_valid_words = sum(len(sent) for _, sent in valid_ix)

    for it in range(MAX_EPOCHS):
        tic = clock()

        # iterate over all training batches, accumulate loss.
        total_loss = 0
        for batch in train_batches:
            dy.renew_cg()
            loss = lm.batch_loss(batch, train=True)
            loss.backward()
            trainer.update()
            total_loss += loss.value()

        # iterate over all validation batches, accumulate loss.
        valid_loss = 0
        for batch in valid_batches:
            dy.renew_cg()
            loss = lm.batch_loss(batch, train=False)
            valid_loss += loss.value()

        toc = clock()

        print(("Epoch {:3d} took {:3.1f}s. "
               "Train perplexity: {:8.3f} "
               "Valid perplexity: {:8.3f}").format(
            it,
            toc - tic,
            exp(total_loss / n_train_words),
            exp(valid_loss / n_valid_words)
            ))

        plt.scatter(x=it, y=exp(total_loss / n_train_words), color='red', s=100)
        plt.scatter(x=it, y=exp(valid_loss / n_valid_words), color='blue', s=100)
    plt.show()
    # FIXME: make sure to update filenames when implementing ngram models
    fn = "embeds_baseline_lm_" + str(N_GRAM_SIZE+1) + "gram"
    if USE_UNLABELED:
        fn += "_unlabeled"

    print("Saving embeddings to {}".format(fn))
    lm.embed.save(fn, "/embed")
