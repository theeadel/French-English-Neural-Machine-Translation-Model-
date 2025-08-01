# 🧠 English-to-French Neural Machine Translation

This project implements a **Seq2Seq model with attention** to translate sentences from **English to French** using a custom dataset. It covers preprocessing, tokenization, model building, training, and translation, all in TensorFlow/Keras.

---

## 🚀 Features

- ✅ Cleaned bilingual dataset (English & French)
- ✅ Word-level tokenization and padding
- ✅ Baseline Seq2Seq and attention-based encoder-decoder models
- ✅ Custom translation function for inference
- ✅ Model weights and tokenizers export for deployment

---

## 📁 Project Structure

```
seq2seq-translation/
├── main.py                 # Or notebook version (.ipynb)
├── en.csv                 # English training data
├── fr.csv                 # French training data
├── eng_tokenizer.pkl      # Saved English tokenizer
├── fr_tokenizer.pkl       # Saved French tokenizer
├── model.weights.h5       # Trained model weights
└── README.md              # Project overview and instructions
```

---

## 📦 Requirements
Install required packages:
```bash
pip install pandas numpy seaborn matplotlib tensorflow
```

---

## 📊 Dataset
- en.csv — English sentences
- fr.csv — French equivalents
- Both files must be in the root directory with the script or notebook.

---

## 🧪 Training
Trains an encoder-decoder model with attention:
```python
model.fit(
  [en_padded_sequences, decoder_input_sequences],
  fr_padded_sequences,
  batch_size=64,
  epochs=10,
  validation_split=0.2
)
```

---

## 🌐 Translate Example
```python
translate_sentence("she is driving the truck")
```
Output:
```
elle conduit le camion
```

---

## 🧠 Possible Improvements
- Add start/end tokens and beam search
- Use pretrained embeddings (e.g., GloVe)
- Switch to Transformer architecture

---

## 📥 Export Artifacts
- Tokenizers saved as `.pkl`
- Weights saved as `.h5` file for reuse

---


> Built with TensorFlow, attention, and a love for languages. 🌍✨
