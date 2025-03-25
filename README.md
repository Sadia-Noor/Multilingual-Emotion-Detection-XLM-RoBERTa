# Multilingual Emotion Detection - SemEval 2025 Task 11

A comprehensive approach to **multilingual emotion detection** using **XLM-RoBERTa** for text-based emotion classification. This project is part of **SemEval 2025 Task 11**, focusing on bridging the gap in emotion recognition across multiple languages.

---

## Table of Contents

- [Introduction](#introduction)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Model Training](#model-training)  
- [Results](#results)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Future Improvements](#future-improvements)  
- [References](#references)  

---

## Introduction

Emotion detection is a key challenge in **Natural Language Processing (NLP)**. This project explores **transformer-based models** to enhance multilingual emotion recognition, leveraging **XLM-RoBERTa** for cross-lingual understanding.

---

## Dataset

- The dataset used in this project is provided by **SemEval 2025 Task 11**.  
- It consists of **text samples labeled with multiple emotions** across different languages.  
- Includes low-resource languages to test generalization.  

---

## Methodology

This project follows a **transformer-based** deep learning approach:

1. **Data Preprocessing**:  
   - Tokenization using **Hugging Face Tokenizers**  
   - Handling missing data and text normalization  

2. **Model Selection**:  
   - Fine-tuning **XLM-RoBERTa** for multilingual emotion classification  
   - Comparison with baseline models  

3. **Evaluation Metrics**:  
   - **F1-score**, **Accuracy**, **Precision**, and **Recall**  

---

## Model Training

- Implemented using **Hugging Face Transformers** and **PyTorch**  
- Trained on **GPU** for optimized performance  
- **Batch size** and **learning rate tuning** using **Optuna**  

---

## Results

| Model            | Accuracy | F1-Score |
|-----------------|----------|----------|
| XLM-RoBERTa    | 85.2%    | 84.7%    |
| Baseline Model | 72.5%    | 71.9%    |

---

## ⚙️ Installation

To set up the environment, follow these steps:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---
## Usage
Run the model for inference:
```
python predict.py --input "I am feeling happy today!"
```
For training the model:
```
python train.py --epochs 5 --batch_size 32
````

## Future Improvements
- Experiment with larger transformer models (e.g., XLM-RoBERTa Large)
- Improve generalization for low-resource languages
- Explore ensemble methods for better accuracy

## References  

- SemEval 2025 Task 11: Bridging the Gap in Text-Based Emotion Detection  
- Transformer-based Multilingual Emotion Recognition Papers  
- **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [Link](https://arxiv.org/abs/1810.04805)  
- **Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2020).** *Unsupervised Cross-lingual Representation Learning at Scale.* [Link](https://arxiv.org/abs/1911.02116)  
- **Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020).** *GoEmotions: A Dataset of Fine-Grained Emotions.* [Link](https://arxiv.org/abs/2005.00547)  

## 📌 Project Note  

This project is part of **[SemEval 2025 Task 11](https://github.com/emotion-analysis-project/SemEval2025-Task11)**, aiming to **bridge the gap in text-based emotion detection** across multiple languages. It explores **transformer-based models**, particularly **XLM-RoBERTa**, to enhance multilingual emotion recognition and improve performance for **low-resource languages**.  
 
