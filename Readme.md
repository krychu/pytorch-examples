# PyTorch examples

This is a collection of PyTorch examples which I'm developing while learning and
transitioning to PyTorch. The examples demonstrate different model architectures
and tasks. Each example is commented rather heavily to make them useful to
others fellow learners. Each example is also self-contained by design. There is
no shared code. The datasets are not synthetic. I try to use popular, real-world
datasets that hopefully make the examples more relevant. Lastly, the models are
not optimized for performance but focus primarily on demonstration and teaching.

# Examples by model

| File | Model | Task | Dataset |
|-----|----|-----|------|
| `mlp_imgcls_mnist.py` | MLP | Image classification | Mnist (torchvision) |
| `mlp_tblbcls_space.py` | MLP | Table binary classification | Spaceship Titanic (Kaggle) |
| `rnn0_lm_wikitext.py` | RNN (from scratch) | Language modeling | wikitext (txt files) |
| `rnn_lm_wikitext.py` | RNN | Languge modeling | wikitext (txt files) |
| `lstm_lm_wikitext.py` | LSTM | Language modeling | wikitext (txt files) |
| `lstm_ts_` | LSTM | Timeseries prediction | |
| `lstm_ts_` | LSTM | Timeseries prediction (multivariate) | |

# Contributions

My intent is that these examples are useful to others, and that they demonstrate
broad range of concepts, architectures, and tasks. It's of course tricky to
decide what and how much to comment. The important goal to keep in mind is that
the examples here shouldn't be the only resource while learning PyTorch so they
don't have to explain everything.

If you think some comments or code is unclear, can be improved, or added please
feel free to submit an issue.
