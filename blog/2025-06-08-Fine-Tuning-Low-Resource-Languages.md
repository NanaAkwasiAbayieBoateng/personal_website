---
layout: post
title: Fine-tuning  and Parameter Efficient Fine-tuning Low Resource Language Models
subtitle:  An example using Twi Dataset
gh-repo: NanaAkwasiAbayieBoateng/NanaAkwasiAbayieBoateng.github.io
gh-badge: [star, fork, follow]
tags: [PEFT,LLM, Rogue,BLEU,METEOR, TER,chrF, COMET]
---



## Introduction



Fine-tuning large language models (LLMs) for low-resource languages (LRLs) is a critical area of research and development, as it helps to democratize AI and ensure that these powerful technologies are accessible to a wider range of linguistic communities. LRLs often face challenges like limited digital presence, scarcity of annotated data, and lack of specialized NLP tools.



### Challenges of Fine-tuning for Low-Resource Languages:

**Data Scarcity:** This is the most significant hurdle. LLMs require vast amounts of text data for pre-training. For LRLs such as twi  data is often unavailable, making it difficult to train models from scratch or fine-tune existing models effectively.

**Lack of annotated data:** Even when raw text exists, labeled datasets for specific NLP tasks (e.g., sentiment analysis, named entity recognition, machine translation) are extremely rare.
**Limited digital presence:** Many LRLs have a predominantly oral tradition or simply haven't been digitized extensively.
**Cultural and linguistic nuances**: Direct translation of data from high-resource languages might miss important cultural contexts, idioms, or linguistic structures unique to the LRL.
**Language Complexity:** Some LRLs have intricate morphological systems, unique syntax, or tone systems that are not easily captured by models primarily trained on more analytical languages (like English).

**Computational Resources:** Even with parameter-efficient fine-tuning, training LLMs can be computationally intensive, which can be a barrier for researchers or organizations with limited access to powerful GPUs.

**Evaluation Metrics:** Standard evaluation metrics (like BLEU for translation) might not adequately capture the quality or cultural appropriateness of outputs for LRLs, necessitating human evaluation or the development of language-specific metrics.

**Catastrophic Forgetting:** When fine-tuning a pre-trained model on a small, specific dataset, there's a risk of "catastrophic forgetting," where the model loses its general linguistic knowledge acquired during pre-training.

### Strategies and Techniques for Fine-tuning for Low-Resource Languages:

The core idea is to leverage knowledge from high-resource languages or from readily available unlabeled text, and then adapt it to the low-resource setting with minimal data.

#### 1. Leveraging Pre-trained Multilingual Models:

**Multilingual Language Models (MLLMs):** Start with models like mBERT, XLM-R, or mT5, which have been pre-trained on text from hundreds of languages. These models learn shared representations across languages, making them good starting points for LRLs, even if the LRL wasn't explicitly included in the pre-training data.

**Cross-lingual Transfer Learning:** The knowledge acquired from high-resource languages by MLLMs can be transferred to LRLs. This means the model can perform well on an LRL task even with limited fine-tuning data in that specific language.
#### 2. Parameter-Efficient Fine-Tuning (PEFT):
These techniques are crucial for reducing computational costs and mitigating catastrophic forgetting when fine-tuning large models with limited data.

**LoRA (Low-Rank Adaptation):** Instead of fine-tuning all model parameters, LoRA injects small, trainable matrices into the model. Only these low-rank matrices are updated during fine-tuning, drastically reducing the number of trainable parameters and memory footprint.
**QLoRA (Quantized LoRA):** Builds on LoRA by incorporating 4-bit quantization, further enhancing memory efficiency, making it possible to fine-tune very large models on consumer-grade GPUs.
Adapter Layers: Small neural modules are inserted between the layers of the pre-trained model. Only these adapter layers are updated during fine-tuning, while the original model weights remain frozen.
**Prompt Tuning/Prefix Tuning:** These methods add trainable "soft prompts" or prefixes to the input, allowing the model to adapt to a new task without modifying the model's core weights.

#### 3 Data Augmentation Techniques:

**Back-translation:** Translate existing LRL text into a high-resource language, then translate it back into the LRL using a machine translation system (even if imperfect). This can generate slightly varied sentences, expanding the training data.
**Synonym Replacement:** Replace words with their synonyms in the LRL (if a reliable thesaurus or word embedding space exists).
**Noise Injection:** Introduce small amounts of noise (e.g., typos, phonetic variations) to existing data to make the model more robust.
**Synthetic Data Generation:** Use a powerful LLM (possibly fine-tuned on a small amount of LRL data) to generate new, synthetic data for the LRL. This needs careful curation to ensure quality.

#### 4 Unsupervised and Semi-Supervised Learning:

**Distant Supervision:** Use external knowledge bases (e.g., Wikipedia lists) to automatically label data, though this can introduce noise.
**Self-training/Pseudo-labeling:** Train a model on a small labeled dataset, then use this model to predict labels for a larger unlabeled dataset. High-confidence predictions are then added to the training data for further fine-tuning.


#### 5 Cross-Lingual Data Sharing and Collaboration:

**Parallel Corpora:** Leverage existing parallel corpora (texts aligned across two or more languages) if available, or actively work on creating them.
**Crowdsourcing:** Engage native speakers and local communities to collect and annotate data. This also helps ensure cultural relevance.
**Community-driven initiatives:** Support projects that aim to digitize and document LRLs.

#### 6 Transfer Learning from Related Languages:

If a closely related, but more resource-rich, language exists, knowledge can be transferred from that language to the LRL. This is particularly effective for languages within the same family or with similar grammatical structures.


#### 7 Curating High-Quality Data:

Even with limited data, the quality of the fine-tuning data is paramount. Ensure the data is clean, relevant, and representative of the task and language. "Garbage in, garbage out" still applies.




Finetuning LLM's can be computationally demanding. For this purpose smaller
 models will be selected for this finetuning demonstration.



```python
#install libraries
!pip install  torch transformers datasets evaluate rouge_score

```


```python
#!pip install datasets --upgrade
```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


```python    
!pip install python-dotenv
from dotenv import load_dotenv
import os
import sys

import os
print(os.environ)
sys.path.append('/huggingface_api')


load_dotenv()  # This must be called BEFORE trying to access the variables

API_KEY = os.getenv('API_KEY')



```python
from huggingface_hub import login, InferenceClient
from google.colab import userdata

from google.colab import userdata
API_KEY  = userdata.get('HF_TOKEN')


#login("huggingface token")

#API_KEY = "huggingface token"
#client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct")

from huggingface_hub import login
login(token = API_KEY)
```

### Data

The Ghana NLP Twi to English dataset can be located [here](https://zenodo.org/records/4432117#.YF5rndKSk2y). It is a bilingual English and Akuapem Twi machine translation dataset. The verified_data.csv, contains 25,421 sentence pairs. These initial translations, generated by a transformer model, were refined by native speakers to ensure accuracy. The dataset's primary purpose is to train machine translation models for Akuapem Twi, but it can also be used for other NLP tasks like Named Entity Recognition and Part-of-Speech tagging with additional annotations. Additionally, it could facilitate training unsupervised embeddings for Akuapem Twi. A smaller, high-quality set of 697 crowdsourced sentences, crowdsourced_data.csv, is also included and recommended as an evaluation set for both English-to-Twi and Twi-to-English translation models.


```python
import os
print(os.listdir())
import transformers
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split
import pandas as pd
import random

crowd_sourced_data = pd.read_csv('/content/crowdsourced_data.csv')

Verified_data = pd.read_csv('/content/verified_data.csv')

crowd_sourced_data.head()

```

    ['.config', 'drive', 'sample_data']
    





  <div id="df-44e2730a-7f3e-4b9c-b879-73a722c5c044" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1. English Sentence/Phrase</th>
      <th>1. Twi Translation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is going on here?</td>
      <td>Ɛdeɛn na  ɛrekɔso wɔ aha?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wake up</td>
      <td>Sɔre</td>
    </tr>
    <tr>
      <th>2</th>
      <td>She comes here every Friday</td>
      <td>Ɔba ha Fiada biara</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Learn to be wise</td>
      <td>Sua nyansa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I didn’t think you would loose your way</td>
      <td>Mannwene da sɛ wo bɛyera</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-44e2730a-7f3e-4b9c-b879-73a722c5c044')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-44e2730a-7f3e-4b9c-b879-73a722c5c044 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-44e2730a-7f3e-4b9c-b879-73a722c5c044');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-1366f0f9-c9c5-4cf2-95b1-93c8b5cc45f1">
      <button class="colab-df-quickchart" onclick="quickchart('df-1366f0f9-c9c5-4cf2-95b1-93c8b5cc45f1')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-1366f0f9-c9c5-4cf2-95b1-93c8b5cc45f1 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
from dotenv import load_dotenv
#load_dotenv()  # This loads .env from current working directory

data = crowd_sourced_data.rename(columns={'1. English Sentence/Phrase': "english", '1. Twi Translation': "twi"})
#Verified_data.rename(columns={'1. English Sentence/Phrase': "en", '1. Twi Translation': "tw"}, inplace=True)

data.head()
```





  <div id="df-c7adab8b-e7b7-4b5d-bcdd-91229d51fd53" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>english</th>
      <th>twi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is going on here?</td>
      <td>Ɛdeɛn na  ɛrekɔso wɔ aha?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wake up</td>
      <td>Sɔre</td>
    </tr>
    <tr>
      <th>2</th>
      <td>She comes here every Friday</td>
      <td>Ɔba ha Fiada biara</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Learn to be wise</td>
      <td>Sua nyansa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I didn’t think you would loose your way</td>
      <td>Mannwene da sɛ wo bɛyera</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c7adab8b-e7b7-4b5d-bcdd-91229d51fd53')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c7adab8b-e7b7-4b5d-bcdd-91229d51fd53 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c7adab8b-e7b7-4b5d-bcdd-91229d51fd53');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d5afeeed-8f07-4274-80b7-5b8242a4c63f">
      <button class="colab-df-quickchart" onclick="quickchart('df-d5afeeed-8f07-4274-80b7-5b8242a4c63f')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d5afeeed-8f07-4274-80b7-5b8242a4c63f button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





### Model 1 : Google T5 Models

The T5 (Text-to-Text Transfer Transformer) model family, developed by Google, frames all NLP tasks as text-to-text problems, enabling a single model to perform diverse functions. These models are pre-trained on a large corpus and can be fine-tuned for specific tasks like translation, summarization, and question answering. The original T5 models come in various sizes, ranging from smaller, efficient versions to massive models like the 11B parameter variant. Among the T5 series, Flan-T5 generally achieves better performance, especially in few-shot learning and instruction following, as it's instruction-tuned. For multilingual tasks like language translation, mT5 (Multilingual T5) is often a better choice compared to the original T5, as it's trained on a multilingual dataset. Flan-T5 models also exhibit strong performance in many tasks and can be fine-tuned for translation, although mT5 may have a richer multilingual vocabulary. Ultimately, the best T5 variant depends on the specific task and available resources.
The T5-small model is a compact yet powerful variant of Google's Text-to-Text Transfer Transformer, designed to unify all NLP tasks into a single text-in, text-out framework. With only 60 million parameters, it offers a highly efficient solution for various language processing needs. Despite its smaller size, it leverages pre-training on the massive C4 dataset to understand diverse linguistic patterns. This enables T5-small to be effectively fine-tuned for tasks such as machine translation, summarization, question answering, and text classification. Its efficiency makes it an excellent choice for resource-constrained environments and accessible prototyping.




```python
# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Model and Tokenizer
#model_name = "t5-small"  # Choose a suitable pre-trained model
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


#model_name = "google/flan-t5-xxl"
model_name = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Tokenization function
def preprocess_function(examples):
    inputs = [f"translate Twi to English: {twi}" for twi in examples["twi"]]
    targets = [english for english in examples["english"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

# Apply tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)
```


    Map:   0%|          | 0/557 [00:00<?, ? examples/s]



    Map:   0%|          | 0/140 [00:00<?, ? examples/s]


Weights & Biases (W&B) is a comprehensive AI developer platform designed to streamline the entire machine learning lifecycle. It provides tools for tracking, visualizing, and managing AI experiments, helping teams build and iterate on models faster. W&B offers features like experiment tracking, hyperparameter optimization (Sweeps), model and dataset versioning (Artifacts), and collaborative reporting. Increasingly, it also provides specialized tools for Large Language Model (LLM) operations, such as W&B Weave for evaluating and debugging LLM applications. Trusted by numerous AI practitioners and organizations, Weights & Biases enables reproducibility, transparency, and efficient collaboration in AI development. To track this experiment, an account will have to be created at https://wandb.ai/home. An api key can then be obtained which is  entered to sign in from notbook to weights and biases.



```python
import evaluate
import numpy as np

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/twi_translation",
    eval_strategy="epoch", # Changed from evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=15,  # Adjust as needed
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), #use fp16 if cuda is available
    metric_for_best_model="rouge1", # add a metric to select best model.
)



# Metric
metric = evaluate.load("rouge")

# Postprocessing for metric computation
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# Compute metrics
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad_token_id before decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


# Apply padding and max_length during data collation instead
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding="max_length",  # Add padding here
    max_length=128,      # Add max_length here (optional)
)

# Update the Trainer with the data_collator
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # Pass the data_collator to the Trainer
)

# Fine-tuning
trainer.train()

# Example Inference
def translate_twi(twi_text, model, tokenizer):
    input_text = f"translate Twi to English: {twi_text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=128).to(model.device)
    # Access the input_ids tensor directly and get its shape
    input_shape = inputs["input_ids"].shape
    # You can now use input_shape if needed
    # Pass input_ids to generate instead of the entire dictionary
    outputs = model.generate(inputs["input_ids"], max_length=128)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Example usage after training:
test_twi_sentence = "Ɛyɛ ahe?"
translated = translate_twi(test_twi_sentence, model, tokenizer)
print(f"Twi: {test_twi_sentence}, Translated: {translated}")

#Save the model.
model.save_pretrained("/content/drive/MyDrive/GhanaNLP/final_model")
tokenizer.save_pretrained("/content/drive/MyDrive/GhanaNLP/final_model")
```

    <ipython-input-9-bff434ee6e0c>:64: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
      trainer = Seq2SeqTrainer(
    



    <div>

      <progress value='329' max='1050' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [ 329/1050 02:17 < 05:03, 2.37 it/s, Epoch 4.69/15]
    </div>


    ---
## Model Training Results





    <div>

      <progress value='1050' max='1050' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1050/1050 08:23, Epoch 15/15]
    </div>
   

   | Epoch | Training Loss | Validation Loss | Rouge1 | Rouge2 | Rougel | Rougelsum | Gen Len |
| :---- | :------------ | :-------------- | :----- | :----- | :----- | :-------- | :------ |
| 1     | No log        | 2.361397        | 15.1206 | 4.6898  | 14.8538 | 14.7921   | 7.6786  |
| 2     | No log        | 2.354005        | 16.0499 | 5.722   | 16.1016 | 16.0298   | 7.7071  |
| 3     | No log        | 2.358570        | 17.315  | 6.6872  | 17.2508 | 17.1557   | 7.7143  |
| 4     | No log        | 2.365293        | 17.2146 | 6.1433  | 17.1501 | 17.0498   | 7.8214  |
| 5     | No log        | 2.372702        | 17.6735 | 6.5794  | 17.4498 | 17.3300   | 7.7643  |
| 6     | No log        | 2.378848        | 18.4638 | 7.0539  | 18.3888 | 18.2505   | 7.5786  |
| 7     | No log        | 2.380730        | 18.0472 | 6.9778  | 18.0897 | 17.9488   | 7.6857  |
| 8     | 1.884000      | 2.392827        | 18.0102 | 6.8799  | 18.0320 | 17.9240   | 7.8143  |
| 9     | 1.884000      | 2.393994        | 19.6326 | 7.6444  | 19.6836 | 19.5272   | 7.8000  |
| 10    | 1.884000      | 2.382862        | 18.8077 | 7.222   | 18.7884 | 18.7365   | 7.7643  |
| 11    | 1.884000      | 2.381667        | 19.5084 | 7.2568  | 19.4134 | 19.3247   | 7.9571  |
| 12    | 1.884000      | 2.379954        | 18.8376 | 6.5393  | 18.6371 | 18.5043   | 7.9357  |
| 13    | 1.884000      | 2.386058        | 19.4781 | 6.9979  | 19.4984 | 19.3514   | 7.9214  |
| 14    | 1.884000      | 2.383623        | 19.9393 | 7.0939  | 19.6939 | 19.6140   | 7.9500  |
| 15    | 1.690400      | 2.382863        | 19.9393 | 7.0939  | 19.6939 | 19.6140   | 7.9571  |


    Twi: Ɛyɛ ahe?, Translated: What is it?
    




    ('/content/drive/MyDrive/GhanaNLP/final_model/tokenizer_config.json',
     '/content/drive/MyDrive/GhanaNLP/final_model/special_tokens_map.json',
     '/content/drive/MyDrive/GhanaNLP/final_model/spiece.model',
     '/content/drive/MyDrive/GhanaNLP/final_model/added_tokens.json',
     '/content/drive/MyDrive/GhanaNLP/final_model/tokenizer.json')




```python
import evaluate
import numpy as np

# ... (rest of the code)

# Compute metrics
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad_token_id before decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

# ... (rest of the code)
```


```python
print(f"Twi: {test_twi_sentence}, Translated: {translated}")
```

    Twi: Ɛyɛ ahe?, Translated: What is it?
    


```python
translated
```




    'What is it?'



Working with low-resource languages and limited compute (like Google Colab, which usually offers 16GB RAM and a T4/V100 GPU), you'll want a model that balances translation quality, language coverage, and resource efficiency. Some of the 3 best choices

- Top Choice: facebook/nllb-200-distilled-600M
- Runner-up: Helsinki-NLP/opus-mt-<lang-pair>
- Bonus: google/byt5-small

### Model 2 : Facebook Model


Facebook's NLLB (No Language Left Behind) model, developed by Meta AI, aims to provide high-quality machine translations for 200 languages, especially focusing on low-resource ones. Its primary goal is to enable direct translation between any pair of these languages, fostering more inclusive global communication. Built on the Transformer architecture, sometimes incorporating Mixture of Experts, NLLB is trained on vast amounts of mined parallel and monolingual data, including specialized datasets like NLLB-Seed. This technology significantly improves access to information and online services for diverse linguistic communities. Meta has open-sourced many NLLB models and benchmarks like FLORES-200 to encourage further research and development. Ultimately, NLLB represents a major advancement in breaking down language barriers and promoting a more interconnected digital world.


```python
pip install transformers datasets evaluate accelerate peft bitsandbytes sacrebleu sacremoses

```

    
    Downloading portalocker-3.1.1-py3-none-any.whl (19 kB)
    Installing collected packages: sacremoses, portalocker, colorama, sacrebleu, bitsandbytes
    Successfully installed bitsandbytes-0.46.0 colorama-0.4.6 portalocker-3.1.1 sacrebleu-2.5.1 sacremoses-0.1.1
    

#### 1. Convert DataFrame to Hugging Face Dataset


```python
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

crowd_sourced_data = pd.read_csv('/content/drive/MyDrive/GhanaNLP/Data/crowdsourced_data.csv')

Verified_data = pd.read_csv('/content/drive/MyDrive/GhanaNLP/Data/verified_data.csv')


train_df, temp_df = train_test_split(crowd_sourced_data, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.rename(columns={'1. English Sentence/Phrase': "en", '1. Twi Translation': "tw"})),
    "validation": Dataset.from_pandas(val_df.rename(columns={'1. English Sentence/Phrase': "en", '1. Twi Translation': "tw"})),
    "test": Dataset.from_pandas(test_df.rename(columns={'1. English Sentence/Phrase': "en", '1. Twi Translation': "tw"}))
})

print(dataset["train"][:5]) # Display the first 5 rows of the training dataset


```

    {'en': ['I am running to school.', 'It is true', 'let us go out for a drink', 'They had to start from scratch.', 'My parents are English, but they came to Brazil in 2001.'], 'tw': ['Meredwane akɔ sukuu.', 'Ɛyɛ ampa', 'ma yenkɔ pɛ biribi nnom', 'Na ɛsɛ sɛ wɔhyɛ ase firi mfitiase', "M'awofo yɛ Enyiresifo, nanso wɔbaa Brazil afe 2001"], '__index_level_0__': [82, 51, 220, 558, 451]}
    


```python
from transformers import AutoTokenizer

model_id = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Language tokens
src_lang = "eng_Latn"
tgt_lang = "aka_Latn"

tokenizer.src_lang = src_lang

def preprocess(example):
    # Ensure 'en' and 'tw' are strings and handle potential missing values
    en_text = str(example["en"]) if example["en"] is not None else ""
    tw_text = str(example["tw"]) if example["tw"] is not None else ""

    inputs = tokenizer(en_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(tw_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
```


    Map:   0%|          | 0/557 [00:00<?, ? examples/s]



    Map:   0%|          | 0/70 [00:00<?, ? examples/s]



    Map:   0%|          | 0/70 [00:00<?, ? examples/s]


#### 2. PEFT + LoRA Setup


**1. Imports:**

- `from peft import get_peft_model, LoraConfig, TaskType:` Imports necessary functions and classes from the PEFT library. `get_peft_model` applies the PEFT method to the model, `LoraConfig` defines LoRA configuration parameters, and `TaskType` specifies the type of task.
- `from transformers import AutoModelForSeq2SeqLM`, AutoTokenizer: Imports classes from the Hugging Face Transformers library to load a pre-trained sequence-to-sequence model and its tokenizer.

**2. Loading the Pre-trained Model and Tokenizer:**

- `model_id = "facebook/nllb-200-distilled-600M"`: Specifies the model identifier for the pre-trained model from the Hugging Face model hub.
- `tokenizer = AutoTokenizer.from_pretrained(model_id)`: Loads the tokenizer associated with the specified pre-trained model.
- `model = AutoModelForSeq2SeqLM.from_pretrained(model_id)`: Loads the pre-trained sequence-to-sequence model.

**3. LoRA Configuration:**

- `peft_config = LoraConfig(...)`: Creates a LoRA configuration object using LoraConfig.
- `r=8`: Sets the rank of the low-rank matrices used in LoRA.
- `lora_alpha=32`: Specifies the scaling factor for the LoRA matrices.
- `task_type=TaskType.SEQ_2_SEQ_LM`: Indicates that the task is a sequence-to-sequence language modeling task.
- `lora_dropout=0.1`: Sets the dropout probability for the LoRA layers.
- `bias="none"`: Specifies that no bias terms are added to the LoRA layers.
- `target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]`: Specifies which layers in the model should be adapted using LoRA. This example targets the query, value, key, and output projection layers within the attention mechanism of the model.

**4. Applying LoRA:**

`model = get_peft_model(model, peft_config)`: Applies the LoRA configuration to the loaded pre-trained model using the get_peft_model function from the PEFT library.



```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.SEQ_2_SEQ_LM,
    lora_dropout=0.1,
    bias="none",
    # Specify the target modules for LoRA
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Example target modules - adjust as needed
)

model = get_peft_model(model, peft_config)

```

#### 3. Training with Accelerate




```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

training_args = Seq2SeqTrainingArguments(
    output_dir="/Data/twi_translation",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,
    logging_dir="/Data/twi_translation"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()

```

    <ipython-input-8-df2ef1be52cd>:16: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
      trainer = Seq2SeqTrainer(
    No label_names provided for model class `PeftModelForSeq2SeqLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
    [34m[1mwandb[0m: [33mWARNING[0m The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
    [34m[1mwandb[0m: Currently logged in as: [33mgucci148[0m ([33mgucci148-nice[0m) to [32mhttps://api.wandb.ai[0m. Use [1m`wandb login --relogin`[0m to force relogin
    


Tracking run with wandb version 0.19.11



Run data is saved locally in <code>/content/wandb/run-20250602_043830-zcm8vp5g</code>



Syncing run <strong><a href='https://wandb.ai/runs/zcm8vp5g' target="_blank">/Data/twi_translation</a></strong> to <a href='https://wandb.ai/huggingface' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/huggingface' target="_blank">https://wandb.ai/huggingface</a>



View run at <a href='https://wandb.ai/huggingface/runs/' target="_blank">https://wandb.ai/huggingface/runs/</a>


    Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
    



    <div>

      <progress value='15' max='15' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [15/15 00:32, Epoch 15/15]
    </div>
    
    | Epoch | Training Loss | Validation Loss |
| :---- | :------------ | :-------------- |
| 1     | No log        | 5.388194        |
| 2     | No log        | 5.378407        |
| 3     | No log        | 5.366781        |
| 4     | No log        | 5.354387        |
| 5     | No log        | 5.341895        |
| 6     | No log        | 5.329644        |
| 7     | No log        | 5.316280        |
| 8     | No log        | 5.304862        |
| 9     | No log        | 5.293172        |
| 10    | No log        | 5.282847        |
| 11    | No log        | 5.272978        |
| 12    | No log        | 5.265112        |
| 13    | No log        | 5.259263        |
| 14    | No log        | 5.254582        |
| 15    | No log        | 5.252716        |


    TrainOutput(global_step=15, training_loss=4.650517272949219, metrics={'train_runtime': 38.016, 'train_samples_per_second': 0.395, 'train_steps_per_second': 0.395, 'total_flos': 4083705446400.0, 'train_loss': 4.650517272949219, 'epoch': 15.0})



### Key Metrics for Language Translation Models

### BLEU (Bilingual Evaluation Understudy)

Concept: Measures the precision of n-grams (sequences of words) in the candidate (machine) translation compared to one or more human reference translations. It also includes a brevity penalty to penalize overly short translations. A higher BLEU score indicates a better translation. BLEU scores typically range from 0 to 1 (or 0 to 100). Higher scores indicate better similarity to the reference translations.
A score of 4.628308277061475 is a very low BLEU score. This indicates that the machine translation has very little n-gram overlap with the reference translations, suggesting poor quality.
Strengths: Widely adopted, easy to calculate, correlates reasonably well with human judgment at the corpus level.
Limitations: Does not directly capture semantic meaning, grammatical correctness, or fluency. Can be sensitive to reference translation variations.
### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

Concept: Addresses some of BLEU's limitations by considering precision and recall, as well as stemming and synonymy matching. It also includes a penalty for word order differences.
Strengths: Better correlation with human judgments than BLEU, particularly at the sentence level, as it accounts for synonyms and word reordering.
Limitations: More computationally intensive than BLEU and requires language-specific resources (e.g., stemming dictionaries).


### TER (Translation Edit Rate)

Concept: Measures the number of edit operations (insertions, deletions, substitutions, and shifts) required to transform the machine translation into a human reference translation, normalized by the length of the reference.
Strengths: Intuitive interpretation (lower score means fewer edits needed, thus better quality), useful for estimating post-editing effort.
Limitations: Primarily focuses on lexical and positional similarity; may not fully capture semantic equivalence.


### chrF (CHaRacter-level F-score)

Concept: Calculates the similarity between the machine translation and reference translation using character n-grams. This makes it less sensitive to word order and morphological variations, especially for highly agglutinative or morphologically rich languages. It's an F-score (harmonic mean of precision and recall) of character n-grams.
Strengths: Language-independent, robust to tokenization differences, shows good correlation with human judgments, especially for morphologically rich languages.
Limitations: May not capture higher-level syntactic or semantic issues as effectively as metrics that consider word meaning.

####  COMET (Cross-lingual Optimized Metric for Evaluation of Translation):

What it measures:

COMET is a more recent, neural network-based metric that aims to better correlate with human judgments of translation quality.
It considers various aspects of translation quality, including fluency, adequacy, and semantic similarity.
COMET utilizes trained models to compare the source text, and the translated text, to the reference text. Because of this, it has a much better correlation to human evaluation of translation.
Interpretation:
The interpretation of COMET scores has evolved. Newer COMET models produce scores that are scaled from 0 to 1, where 1 is a perfect translation.
Because of the nature of COMET, it is generally considered to be a much more reliable metric than BLEU.
Key takeaway:
To accurately interpret a COMET score, it is important to know which COMET model was used. Modern COMET scores are much more reliable than older versions.

### Human Evaluation

Concept: While not an automatic metric, human evaluation remains the gold standard. Professional translators or language experts rate the quality of translations based on criteria like fluency, adequacy, and overall quality.
Strengths: Provides the most accurate and nuanced assessment of translation quality.
Limitations: Time-consuming, expensive, and can be subjective. Often used for final validation or when automatic metrics are insufficient.


 Considerations for Finetuning Evaluation:
Dataset Split: Always evaluate your fine-tuned model on a separate test set that was not used during training or validation. This ensures an unbiased assessment of its generalization capabilities.
Multiple References: Whenever possible, use multiple human reference translations for each source sentence in your evaluation set. This accounts for the inherent variability in human translation and provides a more robust score.
Domain-Specific Evaluation: If your model is fine-tuned for a specific domain (e.g., medical, legal), ensure your evaluation set contains translations relevant to that domain. Generic benchmarks might not fully capture domain-specific improvements.
Human-in-the-Loop: While automatic metrics are convenient, they don't capture all nuances of human language. Incorporate human evaluation for critical assessments, especially after significant model improvements. This can involve A/B testing different model versions or having human annotators rate translations for fluency, adequacy, and specific error types.
Error Analysis: Don't just look at the scores. Analyze the types of errors your model makes (e.g., grammatical errors, lexical errors, fluency issues, factual inaccuracies, hallucinations). This qualitative analysis provides insights for further finetuning or model improvements.
Statistical Significance: When comparing different fine-tuned models, consider using statistical significance tests (e.g., bootstrap resampling) to determine if observed differences in metrics are truly meaningful or just due to random variation.
By combining these automatic metrics with careful dataset preparation and qualitative analysis, you can effectively evaluate and improve your fine-tuned language translation models



### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
ROUGE measures the overlap of n-grams (sequences of words), longest common subsequences (LCS), and skip-bigrams between a candidate (machine) translation and one or more human reference translations. It's "recall-oriented" because its primary focus is on how much of the information in the reference translation is captured by the candidate translation.

There are several variants of ROUGE:

- ROUGE-N: Measures the overlap of n-grams between the candidate and reference.
- ROUGE-1: Unigram (single word) overlap.
- ROUGE-2: Bigram (two-word sequence) overlap.
And so on (ROUGE-3, ROUGE-4).
- ROUGE-L: Based on the Longest Common Subsequence (LCS). This metric is good for capturing sentence-level structural similarity without requiring consecutive matches. It often comes with a precision, recall, and F1-score.

- ROUGE-W: A weighted LCS-based metric that favors consecutive matches more heavily.
- ROUGE-S: Based on skip-bigrams, allowing for gaps between words in the matching bigrams. This can be useful for capturing semantic similarity even if word order is slightly different.
Why is ROUGE useful for Translation?
While BLEU is generally the go-to metric for machine translation due to its precision-oriented nature and strong correlation with human judgment of fluency, ROUGE offers complementary insights:

Recall-Oriented: ROUGE emphasizes whether the key information from the reference is present in the machine translation. This can be crucial if you want to ensure that important concepts or terms are not missed.
Structural Similarity (ROUGE-L): ROUGE-L, by considering the longest common subsequence, can give an indication of how well the overall structure and flow of the sentence are preserved, even if the exact wording differs.
Flexibility with Word Order (ROUGE-S): ROUGE-S can be useful when slight variations in word order are acceptable, as it allows for "skip" matches.
Complementary to BLEU: BLEU is precision-focused, meaning it penalizes generated words that are not in the reference. ROUGE, being recall-focused, penalizes reference words that are not in the generated text. Using both provides a more holistic view of translation quality. A high BLEU score suggests good precision and fluency, while a high ROUGE score indicates good content coverage.
### ROUGE vs. BLEU: Key Differences
Feature	BLEU (Bilingual Evaluation Understudy)	ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Primary Focus	Precision (how much of the candidate is good?)	Recall (how much of the reference is captured?)
Core Idea	N-gram precision with brevity penalty	N-gram recall, LCS, skip-bigram
Typical Use	Machine Translation (main metric)	Text Summarization (main metric)
Sensitivity	Highly sensitive to exact word order and phrasing	More flexible with word order, good for content overlap
Interpretation	Higher score = better (closer to human reference)	Higher score = better (more content from reference)


### When to use ROUGE for Translation?
When you are particularly concerned about ensuring that the machine translation retains all the critical information from the source, even if it uses different phrasing.
In scenarios where some degree of paraphrasing is acceptable, and you want to measure the semantic overlap more than exact lexical matches.
As a complementary metric to BLEU, to get a more comprehensive evaluation, especially if your fine-tuning aims to improve content fidelity.
When evaluating less-resourced languages or domains where perfect word-for-word matches might be less common.
Python Code Example for ROUGE
You can use the rouge-score library in Python to calculate ROUGE scores.

First, install the library:



```python
pip install unbabel-comet
```

   
   

    


```python
import evaluate
from datasets import Dataset # Assuming 'dataset' is a Hugging Face Dataset object

# Load metrics
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
meteor = evaluate.load("meteor")
ter = evaluate.load("ter")
rouge = evaluate.load("rouge")

# Attempt to load COMET, handling potential errors
try:
    comet = evaluate.load("comet", module_type="metric", config_name="wmt20-comet-da")
except Exception as e:
    print(f"Error loading COMET: {e}")
    comet = None

# Assuming 'trainer', 'tokenized', 'tokenizer', and 'dataset' are already defined
prediction_output = trainer.predict(tokenized["test"])
predictions = prediction_output.predictions
labels = prediction_output.label_ids

# Access predictions and labels from the PredictionOutput object
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Ensure decoded_labels is a list of lists for metrics that expect it
# For metrics like BLEU, chrF, METEOR, TER, and ROUGE, references usually need to be a list of lists,
# where each inner list contains one or more reference translations for a given prediction.
# Here, we assume one reference per prediction.
formatted_decoded_labels = [[ref] for ref in decoded_labels]

# Get the source sentences from the test dataset.
# This assumes your 'dataset' object is a Hugging Face Dataset and has an 'en' column.
# Adjust 'en' if your source language column has a different name (e.g., 'source_text').
try:
    sources = dataset["test"]["en"]
    # Crucial: Ensure 'sources' is also a flat list of strings.
    # If dataset["test"]["en"] returns a list of lists, flatten it.
    if isinstance(sources[0], list): # Check if the first element is a list, indicating nested structure
        sources = [item for sublist in sources for item in sublist]
except KeyError:
    print("Warning: 'en' column not found in dataset['test']. COMET evaluation might fail if sources are missing.")
    sources = None # Set to None if sources cannot be found

# --- Evaluate Metrics ---

# BLEU Score
bleu_score = bleu.compute(predictions=decoded_preds, references=formatted_decoded_labels)
print(f"BLEU: {bleu_score['score']:.2f}")

# chrF Score
chrf_score = chrf.compute(predictions=decoded_preds, references=formatted_decoded_labels)
print(f"chrF: {chrf_score['score']:.2f}")

# METEOR Score
meteor_score = meteor.compute(predictions=decoded_preds, references=formatted_decoded_labels)
print(f"METEOR: {meteor_score['meteor']:.2f}")

# TER Score
ter_score = ter.compute(predictions=decoded_preds, references=formatted_decoded_labels)
print(f"TER: {ter_score['score']:.2f}")

# ROUGE Score
# ROUGE can return multiple scores (e.g., rouge1, rouge2, rougel).
# We'll print rougel for simplicity, but you can access others as needed.
rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels) # ROUGE usually expects flat lists for predictions and references
print(f"ROUGE-L: {rouge_score['rougeL']:.2f}")

# COMET Score (only if loaded successfully and sources are available)
if comet is not None and sources is not None:
    try:
        # COMET expects predictions, references, and sources as flat lists of strings
        # Make sure these are truly flat lists for COMET, not lists of lists.
        # decoded_preds and decoded_labels are already flat lists from batch_decode.
        # The key fix is ensuring 'sources' is also a flat list.
        comet_score = comet.compute(predictions=decoded_preds,
                                    references=decoded_labels, # COMET usually expects flat list for references too
                                    sources=sources)
        print(f"COMET: {comet_score['score']:.2f}")
    except Exception as e:
        print(f"Error computing COMET score: {e}")
elif comet is None:
    print("COMET score not computed because the metric could not be loaded.")
elif sources is None:
    print("COMET score not computed because source sentences could not be retrieved from the dataset.")
```

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package punkt_tab to /root/nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to /root/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!
    INFO:pytorch_lightning.utilities.migration.utils:Lightning automatically upgraded your loaded checkpoint from v1.3.5 to v2.5.1.post0. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../root/.cache/torch/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt`
    /usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/saving.py:195: Found keys that are not in the model state dict but in the checkpoint: ['encoder.model.embeddings.position_ids']
    





    BLEU: 4.63
    chrF: 4.74
    METEOR: 0.23
    TER: 100.00
    ROUGE-L: 0.00
  
    


```python
#Save the model.
model.save_pretrained("./twi_translation/final_model")
tokenizer.save_pretrained("./twi_translation/final_model")
```




    ('./twi_translation/final_model/tokenizer_config.json',
     './twi_translation/final_model/special_tokens_map.json',
     './twi_translation/final_model/sentencepiece.bpe.model',
     './twi_translation/final_model/added_tokens.json',
     './twi_translation/final_model/tokenizer.json')



#### Inference: Translate English → Twi with the Fine-Tuned Model


```python
from transformers import pipeline

# Load the fine-tuned model
model.eval()  # Set model to evaluation mode (optional)
tokenizer.src_lang = "eng_Latn"  # Source: English
tgt_lang = "aka_Latn"            # Target: Twi

# Example English sentence
english_sentences = [
    "Good morning, how are you?",
    "I will go to the market tomorrow.",
    "Thank you very much!"
]

# Tokenize and translate
inputs = tokenizer(english_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)

# Get the forced_bos_token_id directly using the target language
forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

# Pass forced_bos_token_id to model.generate
translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)

# Decode output
translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

# Print results
for en, tw in zip(english_sentences, translated_texts):
    print(f"EN: {en}")
    print(f"TW: {tw}")
    print("------")
```

    EN: Good morning, how are you?
    TW: Awia, dɛn na woayɛ?
    ------
    EN: I will go to the market tomorrow.
    TW: Mɛkɔ gua so ɔkyena.
    ------
    EN: Thank you very much!
    TW: Meda mo ase paa!
    ------
    
