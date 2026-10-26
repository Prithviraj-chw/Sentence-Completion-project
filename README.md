# Sentence Completion with RNN & LSTM

A next-word prediction model trained on a dataset of famous quotes. Given the start of a sentence, the LSTM suggests the most likely next words and can extend a seed phrase into a longer sentence.

Built with TensorFlow/Keras and trained on Google Colab (T4 GPU).

## Overview

| | |
|---|---|
| **Task** | Next-word prediction (sentence completion) |
| **Data** | 3,038 quotes from 1,005 authors (`qoute_dataset.csv`) |
| **Training samples** | 85,271 (prefix → next word pairs) |
| **Vocabulary** | 8,978 unique words, capped at 10,000 |
| **Models** | SimpleRNN (baseline) vs LSTM |

## How it works

1. **Preprocess**: lowercase the quotes and strip punctuation.
2. **Tokenize**: map each word to an integer id with the Keras `Tokenizer` (`num_words=10000`).
3. **Build training pairs**: for every quote, each prefix becomes an input and the word that follows becomes the label.
   For `the world as we ...`: `[the] → world`, `[the, world] → as`, `[the, world, as] → we`, and so on.
4. **Pad**: pre-pad every input to a fixed length of `max_len = 50`.
5. **Train**: `Embedding(50) → LSTM(128) → Dense(10000, softmax)` with `sparse_categorical_crossentropy`, so no 85k × 10k one-hot matrix is needed.
6. **Predict**: tokenize and pad the user's text, run the model, and read the top next-word candidates.

### Why `max_len = 50`?

The longest quote has about 746 words, but the median is 18 and the 75th percentile is 29. Padding everything to 746 steps meant most of every sequence was zeros and training was slow. Capping at 50 steps keeps nearly all real context (longer inputs keep their most recent 50 words) and cut epoch time from about 30s to about 6s.

## Model

```
Embedding(input_dim=10000, output_dim=50)
LSTM(128)
Dense(10000, activation="softmax")
```

About 1.88M trainable parameters. Optimizer: Adam. Metrics: accuracy and top-5 accuracy.

## Results

Next-word prediction over a 10k-word vocabulary on ~3k quotes is a hard, high-entropy task: many different words can be correct, and the dataset is small. Low top-1 accuracy is expected, so top-5 accuracy and perplexity are the more informative metrics.

| LSTM run | Train acc | Val loss | Val top-1 | Val top-5 | Val perplexity |
|---|---|---|---|---|---|
| **Early stopping** (best epoch = 5, restored by `val_loss`) | ~10.5% | **6.43** | **9.7%** | **23.4%** | **~620** |
| 100 epochs, no early stopping | 74.1% | 11.89 | 8.0% | 18.3% | ~146,000 |

### Key takeaway: overfitting

Training the LSTM for the full 100 epochs pushes **training accuracy above 70%**, but validation loss bottoms out around epoch 5 and then climbs steadily. The model is memorizing the quotes, not learning to generalize: validation top-1 accuracy actually drops and validation loss nearly doubles. Early stopping on `val_loss` gives a model that is better on unseen text, which is the one to use.

The SimpleRNN baseline shows the same pattern (train accuracy climbs, validation loss rises after the first epochs). The LSTM learns faster and reaches a lower validation loss.

## Repository structure

```
├── sentence_completion.ipynb   # preprocessing, training, evaluation, text generation
├── qoute_dataset.csv           # quotes dataset (quote, Author)
├── lstm_model.h5               # trained LSTM (best-validation weights)
├── tokenizer.pkl               # fitted Keras Tokenizer
└── max_len.pkl                 # sequence length used for padding (50)
```

## Quick start

```bash
pip install tensorflow numpy
```

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("lstm_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
max_len = pickle.load(open("max_len.pkl", "rb"))
index_to_word = {i: w for w, i in tokenizer.word_index.items()}

def suggest_next(text, k=5):
    seq = tokenizer.texts_to_sequences([text.lower()])
    seq = pad_sequences(seq, maxlen=max_len, padding="pre")
    probs = model.predict(seq, verbose=0)[0]
    top = np.argsort(probs)[::-1]
    return [index_to_word[i] for i in top if i in index_to_word][:k]

print(suggest_next("the best way to"))
```

> Load the model with the same TensorFlow/Keras version family used to train it (Colab's current release), since the tokenizer is a pickled Keras object.

To retrain, open `sentence_completion.ipynb` in Google Colab, upload `qoute_dataset.csv`, and run all cells.

## Limitations

- Small dataset (~3k quotes), so the model reproduces common quote phrasing more than it understands language.
- Word-level vocabulary: unseen words are dropped, and there is no handling of out-of-vocabulary tokens.
- Greedy decoding (always taking the most likely word) leads to repetitive output; sampling with a temperature or top-k gives more varied text.

## Possible improvements

- Temperature / top-k sampling for text generation
- Stacked LSTM or GRU layers with dropout
- Pretrained embeddings (GloVe) or a subword tokenizer
- A small Gradio or Streamlit app, deployed on Hugging Face Spaces

## Acknowledgements

Built while following the Deep Learning course by Sheryians AI School, then extended with a `max_len` cap for faster training, sparse labels, top-5 accuracy and perplexity metrics, and an early-stopping vs full-training overfitting comparison.

## Author

Prithviraj Chowdhury
