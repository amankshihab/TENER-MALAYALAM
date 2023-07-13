# Named Entity Recognition In Malayalam

This repo implements and compare two models, namely a Bidirectional LSTM and TENER (Transformer Encoder for Named Entity Recognition) on the [ai4bharat-IndicNER](https://huggingface.co/ai4bharat/IndicNER) dataset,

Find live demo at https://tener-malayalam-k2i4cw0x9c.streamlit.app/

## Pretrained Models

Please find the pretrained weights at: https://drive.google.com/drive/folders/13DQ7zTz8fiSTkwmScpd8ZuO5mVtC4y_C?usp=sharing

## File Structure

- modeling_TENER.ipynb has the code for training and contains some results as well.
- models/ contains the code definitions for both the models.
- malayalam_ner.py implements a helper class that makes it easier to predict with either of the models.
- predict.py contains code for running inference on a single string.

## Tokenization & Embedding

- Byte Pair Encoding has been used here.
- It was chosen after a comparison between it and fasttext.
- The tokens were also vectorized using it's vectorizer.
- It was taken from BPEmb, which has pretrained embedding models for over 275 languages.
- More details can be found [here](https://bpemb.h-its.org/) and for [the specific one I used](https://bpemb.h-its.org/ml/)

## The Models

1) Bidirectional LSTM
    - Uses 3 layers, with hidden size of 200.
    - Uses ReLU as the activation funcion.
    - Combines manually initialized weights and LayerNorm layers for numerical stability.

2) TENER
    - Employs an adaptation of [TENER](https://arxiv.org/abs/1911.04474)
    - Compared to the paper, the CRF layer at the end has been dropped.
    - Here, I have set ```d_model = 512``` and ```n_heads = 16```
    - A weight vector has been used in the loss function to address for the imbalance of tags in the dataset

## Results

- The highest f1-score obtained with BiLSTM is 0.96 and lowest val_loss of 0.09

- The highest f1-score obtained with TENER is 0.98 and lowest val_loss of 0.05.

### NOTE: This was on the test set provided with the dataset.

## Caveats

- While testing I have observed that the model performs better on sentences from the test that the headlines or the title.
