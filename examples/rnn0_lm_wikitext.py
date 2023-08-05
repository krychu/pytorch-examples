# Author: https://github.com/krychu
#
# Problem:   Language modeling. Predict next word.
# Dataset:   wikitext-2 (wikitext-103)
# Solution:  Vanilla RNN network with RNN cell implemented from scratch.
#
# wikitext datasets can be found at:
# https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
#
# Download and extract wikitext-2 dataset into ./datasets/wikitest-2.
#
# wikitext-2 contains 600 train, 60 validation and 60 test articles. This
# amounts to ~2M tokens and vocabulary of 33k words.
#
# wikitext-103 contains 28k train, 60 validation and 60 test articles. This
# gives ~103M tokens and vocabulary of 268k words.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Vocabulary is a container class. While building vocabulary each words is
# added and can later be queried for its index when turning words into int64.
class Vocabulary():
    def __init__(self):
        self._word2idx = {}
        self._idx2word = []

    def add(self, word):
        if word not in self._word2idx:
            self._idx2word.append(word)
            word_idx = len(self._idx2word) - 1
            self._word2idx[word] = word_idx
        return self.word2idx(word)

    def word2idx(self, word):
        return self._word2idx[word]

    def idx2word(self, idx):
        return self._idx2word[idx]

    def __len__(self):
        return len(self._word2idx)

# Build word indices for train, val, and test files given the vocabulary.
#
# INPUT
#
# paths: {
#   "train_file": ...
#   "val_file": ...
#   "test_file": ...
# }
#
# OUTPUT
#
# {
#   "train_tokens"    (TRAIN_WORD_CNT), int64
#   "val_tokens"      (VAL_WORD_CNT), int64
#   "test_tokens"     (TEST_WORD_CNT), int64
# }
#
# e.g., tensor([0, 1, .., 7])
def load_text(paths, vocabulary):
    ret = {}

    for return_key, path in [
            ("train_tokens", paths["train_file"]),
            ("val_tokens", paths["val_file"]),
            ("test_tokens", paths["test_file"])
    ]:
        # Encode text as a sequence of word indexes. Line by line, each
        # finished with <eos> token.
        #
        # word_idxs is an array of tensors, each for one line.
        word_idxs = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                line_word_idxs = [vocabulary.word2idx(word) for word in words]
                word_idxs.append(torch.tensor(line_word_idxs, dtype=torch.int64))

        # word_idxs is an array of tensors, we concatenate them together
        # "horizontally"
        ret[return_key] = torch.cat(word_idxs)

    return ret

def create_vocabulary(path):
    vocabulary = Vocabulary()
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            words = line.split() + ["<eos>"]
            for word in words:
                vocabulary.add(word)
    return vocabulary

# Turn 1D tensor with a single sequence into 2D tensor where each child 1D
# tensor holds a subsequent step of `batch_size` sequences. Having data
# prepared this way makes it easy to pick (L, N) slices for training RNN.
# Doing so simply equates to taking L first elements of the 2d tensor
# returned.
#
# INPUT
#
# tokens        (TRAIN_WORD_CNT), int64
# batch_size    Number of sequences in a batch we use for training
#
# OUTPUT
#
# (CNT, batch_size), int64
#
# e.g., tensor([[0, 1, 3], [2, 3, 7], ...]) -- this is for batch_size = 3
#               step 0     step 1
#
# CNT is irrelevant, it is a result of cutting TRAIN_WORD_CNT into `batch_size`
# equal pieces.
#
# Here is a pictorial depiction of what batchify does:
# ----------------------------------------------
#
# (batch_size, -1)
# ---------  ---------  ---------  ---------  ---------
# seq        seq        seq        seq        seq
#
# (-1, batch_size)
# ----- ----- ----- ----- ----- ----- ----- ----- -----
# step  step  step  step  step  step  step  step  step
#
# s s s s s
# e e e e e
# q q q q q
# | | | | | step
# | | | | | step
# | | | | | step
# | | | | | step

def batchify(tokens, batch_size, device):
    length = (tokens.shape[0] // batch_size) * batch_size;
    tokens = torch.narrow(tokens, 0, 0, length);
    # 1. We split a sequence into `batch_size` pieces
    #    torch.Size([batch_size, -1])
    #
    # 2. We transposed the above to get some count of steps of `batch_size`
    #    sequences
    #    torch.Size([-1, batch_size])
    batchified = tokens.view(batch_size, -1).t().contiguous() # int64
    return batchified.to(device)

# Return specific batch for traning: batch sequences and targets. We're still
# operating on word indices (int64)!
#
# INPUT
#
# batchified    Output of batchify(). (-1, batch_size)
# i             Which batch to return
# seq_len       Length of the batch
#
# OUTPUT
#
# [0]           Batch. (seq_len, batch_size), int64
# [1]           Targets. (seq_len * batch_size), int64
#
# Note that we're not specifying the size of the batch. This information is
# present in the shape of `batchified` tensor.
#
# Also note how we return targets. We need a target for each step of each batch
# sequence. This would suggest a 2D tensor, but in fact we return 1D tensor
# where targets for each step are placed one after another. This is because our
# loss function operates on 1D tensor and so we prepare it accordingly. See
# comments inside the function for more detail.
#
# `get_batch` allows you to take overlapping batches. But you will see that
# train() makes sure to request non-overlapping batches when training.
def get_batch(batchified, i, seq_len):
    # Preferably get seq_len, unless there are not enought steps left. In that
    # case get what's left.
    seq_len = min(seq_len, len(batchified) - 1 - i)

    # batchified -> (-1, batch_size)

    x = batchified[i:i+seq_len]

    # .view(-1) might be the confusing part. Essentially, our RNN network will
    # return probabilities of each word index, for each step of each batch
    # sequence. This gives us a 3D output. But the CrossEntropyLoss expects 2D
    # input, where we have the aforementioned probabilities per index, but not
    # as a 2D grid, just a 1D tensor.
    #
    # For this reason we are "flattening" both, the network output as well as
    # the targets here.
    #
    # Example for seq_len=10, batch_size=5, and vocab_size=33278
    # (10, 5, 33278) -> (50, 33278)
    y = batchified[i+1:i+1+seq_len].view(-1)
    # x -> (seq_len, batch_size)
    # y -> (seq_len * batch_size)

    return x, y

# This is our RNN module. The module can be designed in different ways. For
# example: 1) The module could take a single step of a batch of sequences and
# returns the next hidden state for each sequence, 2) The module could take a
# batch of full sequences, iterate over the steps internally and return hidden
# states for all steps of all batch sequences. We could also design the module
# in such a way that it returns the actual prediction instead of the hidden
# states. In this case it would be more like a full network rather than a
# module. We take the approach outline as 2) to follow PyTorch design as
# closely as possible.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    # The module takes a batch of sequences and returns hidden state for each
    # step of each sequence. Each step of each sequence is represented as a
    # tensor of `input_size`.
    #
    # INPUT
    #
    # x        (L, N, input_size)
    # h_t      (N, hidden_size)
    #
    # OUTPUT
    #
    # y        (L, N, hidden_state)
    # h_t      (N, hidden_state)
    def forward(self, x, h_t):
        seq_len = x.shape[0]

        # This is where we collect outputs (hidden states) at every step for
        # all sequences. We will concatenate this at the end. It's important to
        # return it back to the caller so that they can calculate the loss on
        # each step and back propagate.
        hs = []

        # In RNN, we process input sequence step by step. This is because
        # output of the first step is one of the inputs of the second step etc.
        for step_idx in range(seq_len):
            # We extract a single step across all sequences

            # x -> (L, N, input_size)

            x_t = x[step_idx, :]
            # x_t -> (N, input_size)

            # Now we transform x_t (input) and h_t (hidden state) into the
            # hidden-space and then calculate the new hidden state in that
            # space.

            xh_t = self.i2h(x_t)
            # xh_t -> (N, hidden_size)
            # h_t -> (N, hidden_size)

            hh_t = self.h2h(h_t)
            # hh_t -> (N, hidden_state)

            h_t = self.relu(xh_t + hh_t)
            # h_t -> (N, hidden_state)

            # Finally we collect the new hidden states for this step for all
            # sequences in the batch
            hs.append(h_t)

        # Stack hidden states from all steps
        # hs -> [(N, hidden_size), (N, hidden_size), ...]
        #
        # return
        # hs  -> (L, N, hidden_size)
        # h_t -> (N, hidden_size)
        return torch.stack(hs), h_t

    def init_hidden(self, batch_size, device):
        # TODO: init nicely
        return torch.zeros(batch_size, self.hidden_size).to(device)

class Model(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size):
        super(Model, self).__init__()

        # Word embeddings will be learned as part of the training. Consequently
        # they will be geared towards the specific task and dataset used in
        # this tutorial. Other approaches are possible, such as encoding words
        # as one-hot vectors or using pre-trained embeddings.
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        # init?

        self.rnn = RNN(embedding_size, hidden_size)

        self.ho = nn.Linear(hidden_size, vocabulary_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.ho.bias)
        nn.init.uniform_(self.ho.weight, -initrange, initrange)

    # The model takes a batch of sequences and returns raw predictions for each
    # step of each sequence. Each step of each sequence is an int64, an index
    # of a word in vocabulary.
    #
    # INPUT
    #
    # x        (L, N), int64
    # h_t      (N, hidden_size)
    #
    # OUTPUT
    #
    # y        (L, N, vocabulary_size)
    # h_t      (N, hidden_state)
    def forward(self, x, h_t):
        # (L, N), int64

        x = self.embedding(x)
        # (L, N, embedding_size)

        x, h_t = self.rnn(x, h_t)
        # (L, N, hidden_size)

        x = self.ho(x)
        # (L, N, vocabulary_size) <- raw scores

        # Warning! No Softmax at the end
        #
        # Normally you'd expect to see Softmax or LogSoftmax as the last layer
        # of such a network. But in PyTorch CrossEntropyLoss, which we will be
        # using in training, combines LogSoftmax and NLLLoss in one class, so
        # we don't need to add such layer here. Also note that we return raw
        # scores (logits), just like CrossEntropyLoss expects it.

        return x, h_t

    def init_hidden(self, batch_size, device):
        return self.rnn.init_hidden(batch_size, device)

# Do a single epoch on train_batchified
def train_epoch(cfg, epoch_idx):
    cfg["model"].train()

    total_loss = 0
    interval_loss = 0

    h_t = cfg["model"].init_hidden(cfg["batch_size"], cfg["device"])
    # h_t -> (N, hidden_state)

    # We iterate over batches with non-overlapping sequences
    for batch_idx, batchified_idx in enumerate(range(0, len(cfg["train_batchified"])-1, cfg["seq_len"])):
        #x, y = get_batch(batchified, batch_idx, seq_len) # what about shuffle
        x, y = get_batch(cfg["train_batchified"], batchified_idx, cfg["seq_len"])
        # x -> (L, N), int64
        # y -> (L*N), int64

        # Keep values of the hidden state between the batches because the new
        # batches are the continuation of the previous ones. Disconnect,
        # however, the hidden state from the computational graph.
        h_t = h_t.detach()

        y_pred, h_t = cfg["model"](x, h_t)
        # y_pred -> (L, N, vocabulary_size)
        # h_t -> (N, hidden_state)

        # "Flatten" 3D into 2D output
        #
        # Just like in the case of batchify() we prepare the output to be
        # 2-dimensional, where steps from all batch sequences are arranged into
        # a tensor.
        y_pred = y_pred.view(-1, cfg["vocabulary_size"])
        # (L*N, vocabulary_size)

        batch_loss = cfg["criterion"](y_pred, y)
        # tensor(loss_value)

        total_loss += batch_loss.item()
        interval_loss += batch_loss.item()

        cfg["optimizer"].zero_grad()
        batch_loss.backward()
        cfg["optimizer"].step()

        # Clip gradient to help with gradient explosion
        if cfg["clip"]:
            torch.nn.utils.clip_grad_norm_(cfg["model"].parameters(), cfg["clip"])

        if (batch_idx+1) % cfg["log_interval"] == 0:
            total_loss_per_batch = total_loss / (batch_idx+1)
            #interval_loss_per_batch = interval_loss / (batch_idx+1)
            interval_loss_per_batch = interval_loss / cfg["log_interval"]
            interval_loss = 0
            print(" epoch: {:3d}/{:d}, batch: {:5d}/{:5d}, loss: {:5.3f}, ppl: {:8.2f}".format(
                epoch_idx+1,
                cfg["epoch_cnt"],
                batch_idx+1,
                len(cfg["train_batchified"]) // cfg["seq_len"],
                interval_loss_per_batch,
                math.exp(interval_loss_per_batch)
            ))

def train(cfg):
    best_val_loss = None

    tt = time.time()

    for epoch_idx in range(cfg["epoch_cnt"]):
        t = time.time()
        train_epoch(cfg, epoch_idx)

        if cfg["generate_in_training"]:
            print()
            generate(cfg["model"], cfg["vocabulary"], "Oakland School is located", 10, cfg["device"])
            generate(cfg["model"], cfg["vocabulary"], "He was born in", 10, cfg["device"])
            generate(cfg["model"], cfg["vocabulary"], "Sun and moon", 10, cfg["device"])

        val_loss = evaluate(cfg, cfg["val_batchified"])

        print()
        print("end of epoch: {:3d}, val loss: {:5.3f}, val ppl: {:8.2f}, time: {:5.2f}s".format(
            epoch_idx+1,
            val_loss,
            math.exp(val_loss),
            time.time() - t
        ))

        print()
        if not best_val_loss or val_loss < best_val_loss:
            print(f'  -> New best model, saved to "{cfg["best_model_filename"]}"')
            with open(cfg["best_model_filename"], "wb") as f:
                torch.save(cfg["model"], f)
            best_val_loss = val_loss
        print(f'  -> Last model saved to "{cfg["last_model_filename"]}"')
        print()
        with open(cfg["last_model_filename"], "wb") as f:
            torch.save(cfg["model"], f)

    test_loss = evaluate(cfg, cfg["test_batchified"])

    print()
    print("end of training, test loss: {:5.3f}, test ppl: {:8.2f}, time: {:5.2f}s".format(
        test_loss,
        math.exp(test_loss),
        time.time() - tt
    ))

def evaluate(cfg, data_batchified):
    # eval mode disables things like dropout
    cfg["model"].eval()

    total_loss = 0
    batch_cnt = 0

    h_t = cfg["model"].init_hidden(cfg["batch_size"], cfg["device"])

    with torch.no_grad():
        for batch_idx, batchified_idx in enumerate(range(0, len(data_batchified)-1, cfg["seq_len"])):
            x, y = get_batch(data_batchified, batch_idx, cfg["seq_len"])
            h_t = h_t.detach()
            y_pred, h_t = cfg["model"](x, h_t)

            y_pred = y_pred.view(-1, cfg["vocabulary_size"])

            batch_loss = cfg["criterion"](y_pred, y).item()
            total_loss += batch_loss

            batch_cnt += 1

    return total_loss / batch_cnt

def generate(model, vocabulary, start_sequence, word_cnt, device):
    model.eval()

    words = start_sequence.split()
    sequence = [vocabulary.word2idx(word) for word in words]
    x = torch.tensor(sequence).view(len(sequence), 1).contiguous()
    # x -> (L, 1)

    print(f"{start_sequence} ", end="")
    with torch.no_grad():
        h_t = model.init_hidden(1, device) # 1 sequence in a batch
        # h_t -> (1, hidden_size)

        for i in range(word_cnt):
            # x -> (L, 1)

            y_pred, h_t = model(x, h_t)
            # y_pred -> (L, 1, vocabulary_size)
            # h_t -> (1, hidden_size)

            # Take prediction from the last step.
            o = y_pred[-1]
            # o -> (1, vocabulary_size)

            o = o.squeeze()
            # o -> (vocabulary_size)

            # The RNN network outputs logits (raw scores). We turn them into
            # probabilities which we can later sample from.

            o = F.softmax(o, dim=0)

            # If RNN network returned LogSoftmax instead of logits, we would
            # use o = o.exp()

            word_idx = torch.multinomial(o, 1).item()
            #word_idx = torch.argmax(o).item()

            word = vocabulary.idx2word(word_idx)
            print(f"{word} ", end="")

            x = torch.tensor([[word_idx]])

    print()

def get_data_paths():
    wikitext_version = 2
    # wikitext_version = 103

    return {
        "train_file": f"./datasets/wikitext-{wikitext_version}/wiki.train.tokens",
        "val_file": f"./datasets/wikitext-{wikitext_version}/wiki.valid.tokens",
        "test_file": f"./datasets/wikitext-{wikitext_version}/wiki.test.tokens"
    }

# Create training config
def create_config():
    data_paths = get_data_paths()
    vocabulary = create_vocabulary(data_paths["train_file"])
    tokens = load_text(data_paths, vocabulary)

    cfg = {
        "hidden_size": 100,
        "embedding_size": 100,
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),

        "device": "cpu",
        "seed": None,
        "seq_len": 30,
        "batch_size": 16,
        "epoch_cnt": 3,
        "lr": 0.001,
        "clip": None, # 0.25
        "criterion": nn.CrossEntropyLoss(),

        "log_interval": 200, # number of batches
        "generate_in_training": True,

        "init_model_filename": None,
        # "init_model_filename": "last-model",

        "best_model_filename": "best-model",
        "last_model_filename": "last-model",
    }

    cfg["train_batchified"] = batchify(tokens["train_tokens"], cfg["batch_size"], cfg["device"])
    cfg["val_batchified"] = batchify(tokens["val_tokens"], cfg["batch_size"], cfg["device"])
    cfg["test_batchified"] = batchify(tokens["test_tokens"], cfg["batch_size"], cfg["device"])

    if cfg["init_model_filename"]:
        print()
        print(f'  -> Init model with "{cfg["init_model_filename"]}"')
        print()
        with open(cfg["init_model_filename"], "rb") as f:
            cfg["model"] = torch.load(f)
    else:
        cfg["model"] = Model(cfg["vocabulary_size"], cfg["embedding_size"], cfg["hidden_size"])

    cfg["optimizer"] = torch.optim.Adam(cfg["model"].parameters(), lr=cfg["lr"])
    return cfg

if __name__ == "__main__":
    mode = ["train", "generate"][0]
    if mode == "train":
        cfg = create_config()
        if cfg["seed"]:
            torch.manual_seed(cfg["seed"])
        train(cfg)
    elif mode == "generate":
        with open("last-model", "rb") as f:
            model = torch.load(f)
            device = "cpu"
            data_paths = get_data_paths()
            vocabulary = create_vocabulary(data_paths["train_file"])
            generate(model, vocabulary, "Oakland School is located", 10, device)
            generate(model, vocabulary, "He was born", 10, device)
