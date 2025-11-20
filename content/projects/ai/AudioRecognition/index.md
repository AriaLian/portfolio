+++
title = "Audio Recognition"
summary = ""
description = ""
featuredImage = ""
tags = [""]
categories = ["AI"]
collections = [""]
draft = true
+++

### Data Processing:
- Audio data from GTZAN dataset is processed into spectrograms (128×128 pixels).
- Data normalization applied, scaling features to [0,1].

### CNN Model (ResNet50):
- Load pre-trained ResNet50, freeze layers to maintain learned spatial features.
- Attach Global Average Pooling and dense classification layers for genre classification.
- Train model with Adam optimizer and categorical cross-entropy loss.

### Transformer Model (Wav2Vec 2.0):
- Leverage Wav2Vec 2.0 pre-trained transformer model with frozen parameters.
- Classification layers include dimensionality reduction and output layers.
- Utilize Adam optimizer and cross-entropy loss during training.

### Model Evaluation:
- Performance assessed by accuracy and confusion matrices.
- Training and validation loss curves plotted for both models.

Link to Dataset:
[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

<!-- {{< button href="" target="_self" >}}
{{< icon "link" >}} View on Google Colab
{{< /button >}} -->

<!-- ![](00.jpeg) -->

