# CS5720 - Neural Networks and Deep Learning  
### Home Assignment 3 – Summer 2025  
**Student Name:** Swapnil Mergu
**Student Id:** 700772464
**University of Central Missouri**  
**Course:** CS5720 Neural Networks and Deep Learning  

---

## Assignment Overview
ChatGPT said:
This assignment explores key NLP techniques such as tokenization, stopword removal, stemming, lemmatization, Named Entity Recognition (NER) with spaCy, Scaled Dot-Product Attention, text generation using LSTM-based RNNs, and sentiment analysis using HuggingFace Transformers.

### 1. **RNN (LSTM) for Text Generation**
- Dataset: Sample texts (e.g., Shakespeare Sonnets, The Little Prince)
- Preprocessing: Character tokenization
- Model: LSTM-based Recurrent Neural Network
- Output: (Generated text using temperature scaling)

<pre>That time of year in’d mj(bow,
 th  tut py meerd:
Bjy withes f fareye me ve than t t d  t   t gh ve  Mesort bize t  p’mery him:Ores ist fasunor chach Hand,
We I
Whthancarun t s  sth le ccunt st pise k, Thanostwh podey
</pre>

### Short Question Answers:
**1. Explain the role of temperature scaling in text generation and its effect on randomness.**

Temperature is a parameter that controls the randomness of predictions in the softmax output layer during sampling:
Low temperature (< 1) → model becomes more confident, chooses high-probability characters (less creative).
High temperature (> 1) → output probabilities become flatter, more randomness (creative but may produce gibberish).

Effects on output with respect to temperature:
0.2:	Repetitive, conservative
0.8: 	Balanced, creative
1.2	:	Very random, possibly incoherent

---

### 2. **NLP Preprocessing Pipeline**
- Tokenization
- Stopword Removal
- Stemming using NLTK
- Example: `"NLP techniques are used in virtual assistants like Alexa and Siri."`
- Output: 
<pre>Original Tokens: ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
Tokens Without Stopwords: ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri', '.']
Stemmed Words: ['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri', '.'] </pre>

### Short Question Answers:
**1. What is the difference between stemming and lemmatization? Provide examples with the word “running.”**

Stemming is a crude process that chops off word endings to reduce words to their base or root form. It may not produce a real word.
Lemmatization is more sophisticated and returns the base or dictionary form (lemma) of a word using vocabulary and morphological analysis.
Example with the word “running”:
Stemming: running → run (using PorterStemmer and sometime it may return runn)
Lemmatization: running → run (with correct part-of-speech tag, like verb)

**2. Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?**

Removing the commonly occurring words like "the", "is", "in", which do not add significant meaning helps to reduce noise and dimensionality in tasks like text classification, topic modeling, or information retrieval.
Harmful scenario:
In tasks where context or grammatical structure is important, such as machine translation, question answering, or sentiment analysis, removing stop words might remove useful signals (e.g., “not” in “not happy”).

---


### 3. **Named Entity Recognition (NER) with spaCy**
- Detects entities like names, locations, dates
- Prints: entity text, label (e.g., PERSON), and character range
- Input: `Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009.`
- Output: 
<pre>Named Entities:
Text: Barack Obama, Label: PERSON, Start: 0, End: 12
Text: 44th, Label: ORDINAL, Start: 27, End: 31
Text: the United States, Label: GPE, Start: 45, End: 62
Text: the Nobel Peace Prize, Label: WORK_OF_ART, Start: 71, End: 92
Text: 2009, Label: DATE, Start: 96, End: 100
</pre>

### Short Question Answers:
**1. How does NER differ from POS tagging in NLP?**

NER (Named Entity Recognition) identifies and classifies named entities in text such as people, organizations, locations, dates, etc.

In contrast, POS (Part-of-Speech) tagging assigns grammatical categories (e.g., noun, verb, adjective) to each word in a sentence.

Example sentence: "Barack Obama was elected president in 2008."
- **NER**:
Barack Obama → PERSON

2008 → DATE

- **POS tagging**:

Barack (NNP), Obama (NNP), was (VBD), elected (VBN), president (NN), in (IN), 2008 (CD)

So, POS tagging is about grammar, while NER is about semantic meaning.

**2. Describe two applications that use NER in the real world (e.g., financial news, search engines).**

- **Financial News Analysis:** NER is used to extract company names, stock tickers, economic indicators, and events from financial articles to assist in automated trading or market analysis.

- **Search Engines:** NER helps identify entities in user queries (like “restaurants in New York” or “movies by Christopher Nolan”) to improve search relevance and context-aware results.

---

### 4. **Scaled Dot-Product Attention**
- Manual implementation of attention mechanism
- Includes softmax, scaling by √d, and output matrix
- Inputs: 
<pre>Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
</pre>
- Output:
<pre>Attention Weights:
 [[0.73105858 0.26894142]
 [0.26894142 0.73105858]]

Output:
 [[2.07576569 3.07576569 4.07576569 5.07576569]
 [3.92423431 4.92423431 5.92423431 6.92423431]]</pre>

### Short Question Answers:
**1. Why do we divide the attention score by √d in the scaled dot-product attention formula?**

We divide the attention score by √d (where d is the dimension of the key vectors) to prevent very large values in the dot product of queries and keys.

**2. How does self-attention help the model understand relationships between words in a sentence?**

Self-attention allows each word in a sentence to "look at" and relate to every other word, including itself. Unlike traditional models that only read left-to-right or right-to-left, self-attention gives a global view — it lets the model capture dependencies between distant words, regardless of their position in the sentence.

---

### 5. **Sentiment Analysis with HuggingFace**
- Uses pre-trained sentiment analysis model
- Outputs label (e.g., POSITIVE) and confidence score
- Input: `Despite the high price, the performance of the new MacBook is outstanding.`
- Output:
<pre>Sentiment: POSITIVE
Confidence Score: 0.9998</pre>

### Short Question Answers:

**1. What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?**

BERT uses only the encoder part of the Transformer architecture. It reads the text bidirectionally, meaning it considers context from both left and right. It is mainly used for understanding tasks like classification, NER, and question answering.
GPT uses only the decoder part of the Transformer. It reads text left to right (unidirectionally). It is mainly used for generating text like writing essays, answering questions, or conversations.

**2. Explain why using pre-trained models (like BERT or GPT) is beneficial for NLP applications instead of training from scratch.**

- **Saves Time and Resources:** Pre-trained models have already learned general language patterns from large datasets, avoiding the need for massive computing power.
- **Better Performance:** These models often outperform models trained from scratch, especially on small or medium-sized datasets.
- **Transfer Learning:** They can be fine-tuned on specific tasks with relatively little data, adapting general language knowledge to specialized applications.


---

---

## 🛠️ Technologies Used

- Python 3
- Jupyter Notebook (Google Colab)
- TensorFlow & Keras
- NLTK
- spaCy
- HuggingFace Transformers
- NumPy

---

---
# How to Run

```bash
# 1. Clone the Repository
git clone neural-network-home-assignment-3
open n-n-home-asignment-1/home-assignment3.ipynb

# 2. Run the each Python Scripts cell
