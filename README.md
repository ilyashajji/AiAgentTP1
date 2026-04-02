# 🤖 Prompt Engineering for Multi-Agent Systems

> **Cours : IA et IA Distribuée — Ingénierie des Prompts**

Ce notebook couvre les fondamentaux du prompt engineering appliqué à différents LLMs (OpenAI, Ollama, Groq), ainsi qu'un cas d'usage complet d'analyse de sentiment sur des données IMDB.

---

## 📋 Table des matières

1. [Technologies utilisées](#technologies-utilisées)
2. [Tokenisation avec Tiktoken](#tokenisation-avec-tiktoken)
3. [Prompting des LLMs](#prompting-des-llms)
   - [OpenAI](#openai)
   - [Ollama (local)](#ollama-local)
   - [Groq](#groq)
4. [LLMs Multimodaux](#llms-multimodaux)
5. [Cas d'usage : Analyse de Sentiment](#cas-dusage--analyse-de-sentiment)
   - [Étape 1 : Objectifs & Métriques](#étape-1--objectifs--métriques)
   - [Étape 2 : Préparation des données](#étape-2--préparation-des-données)

---

## 🛠️ Technologies utilisées

| Bibliothèque | Utilisation |
|---|---|
| `langchain-openai` | Interface avec les modèles OpenAI (GPT) |
| `langchain-ollama` | Interface avec les modèles Ollama (local) |
| `langchain-groq` | Interface avec les modèles Groq |
| `tiktoken` | Tokenisation et comptage de tokens |
| `datasets` (HuggingFace) | Chargement du jeu de données IMDB |
| `scikit-learn` | Calcul des métriques (F1-score) |
| `pandas` | Manipulation des données tabulaires |

---

## 🔤 Tokenisation avec Tiktoken

Avant d'envoyer un texte à un LLM, il est utile de comprendre comment il est découpé en **tokens** (unités de traitement du modèle).

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")

system_message = """
Perform Sentiment analysis of the review presented in the user message.
The result should be positive or negative. Do not justify your response
"""

tokens = encoding.encode(system_message)
print(len(tokens))  # → 28 tokens
```

> 💡 **Fonction utilitaire** : compter les tokens d'une chaîne de caractères.

```python
def num_tokens_from_string(string: str, encoding_name: str = "o200k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

num_tokens_from_string("tiktoken is great!")  # → 6
```

---

## 💬 Prompting des LLMs

### OpenAI

```python
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4o", temperature=0)

response = model.invoke([
    SystemMessage("You are a helpful assistant. The output should be in Markdown"),
    HumanMessage("C'est quoi un Agent AI")
])
```

> ℹ️ Il est également possible d'interagir avec les modèles OpenAI via un client HTTP (ex. Postman) en ciblant directement l'endpoint REST.

---

### Ollama (local)

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

response = llm.invoke([
    SystemMessage("You are a helpful assistant. The output should be in Markdown"),
    HumanMessage("C'est quoi un Agent AI")
])
```

> ℹ️ Ollama permet de faire tourner des modèles **localement**, sans clé API. Consultable aussi via Postman.

---

### Groq

```python
from langchain_groq import ChatGroq

llm_groq = ChatGroq(model="openai/gpt-oss-120b")

response = llm_groq.invoke([
    SystemMessage("You are a helpful assistant. The output should be in Markdown"),
    HumanMessage("C'est quoi un Agent AI")
])
```

---

## 🖼️ LLMs Multimodaux

### Génération d'images

```python
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([
    {"type": "image_generation", "quality": "high"}
])

response = llm_with_tools.invoke([
    HumanMessage("Je veux une photo d'un chat qui code du java")
])

import base64
from IPython.display import Image
Image(base64.b64decode(response.content_blocks[0]['base64']))
```

### Description d'images

```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

img = encode_image("rag.png")

response = llm.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Qu'est ce que tu vois dans cette image"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
    ])
])
```

---

## 📊 Cas d'usage : Analyse de Sentiment

### Exemple : Aspect-Based Sentiment Analysis (ABSA)

Identifie la polarité (`positive`, `negative`, `neutral`) pour chaque aspect d'un avis produit (`screen`, `keyboard`, `pad`).

```python
system_message = """
Effectuez une analyse de sentiments basée sur les aspects des avis concernant les ordinateurs portables.
Aspects possibles : screen, keyboard, pad.
Pour chaque aspect, attribuez une polarité : positive, negative ou neutral.
Répondez uniquement en JSON avec le format :
  { "category": [...], "polarity": [...] }
"""

prompt = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "L'écran est très bon, mais je n'ai pas aimé la souris. Le clavier Ma fih Maytchaf"}
]

response = ChatOpenAI(model="gpt-4o").invoke(prompt)
```

**Sortie :**
```json
{
  "category": ["screen", "keyboard", "pad"],
  "polarity": ["positive", "neutral", "negative"]
}
```

---

### Étape 1 : Objectifs & Métriques

L'objectif est d'attribuer un sentiment (`positif` / `négatif`) aux avis clients Amazon (catégorie habillement).

#### Métriques d'évaluation

**Accuracy** : proportion de prédictions correctes sur l'ensemble des exemples.

**Micro-F1 Score** : métrique plus robuste, recommandée en pratique.

Exemple de calcul avec une matrice de confusion :

| Prédit / Réel | Positif | Négatif |
|---|---|---|
| **Positif** | 50 (TP) | 10 (FP) |
| **Négatif** | 5 (FN) | 100 (TN) |

```
Micro-Precision  = 50 / (50 + 10) = 0.833
Micro-Recall     = 50 / (50 + 5)  = 0.909
Micro-F1         = 2 × (0.833 × 0.909) / (0.833 + 0.909) ≈ 0.870
```

> 💡 On utilise `f1_score` de **scikit-learn** pour automatiser ce calcul lors de l'évaluation des prompts.

---

### Étape 2 : Préparation des données

Jeu de données : **IMDB Movie Reviews** (HuggingFace Datasets)

```python
from datasets import load_dataset

corpus = load_dataset("imdb")
# train: 25 000 exemples | test: 25 000 exemples | unsupervised: 50 000

train_df = corpus['train'].to_pandas()
print(train_df['label'].value_counts())
# 0 (négatif) : 12 500
# 1 (positif) : 12 500
```

> ✅ Le jeu de données est **parfaitement équilibré** (50% positif / 50% négatif), ce qui justifie l'utilisation du Micro-F1 comme métrique principale.

---

## ⚙️ Configuration

Créez un fichier `.env` à la racine du projet :

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

Chargement dans le notebook :

```python
from dotenv.ipython import load_dotenv
load_dotenv(override=True)
```

---

## 📁 Structure du projet

```
.
├── notebook.ipynb       # Notebook principal
├── rag.png              # Image utilisée pour la démo multimodale
├── .env                 # Variables d'environnement (non versionné)
└── README.md
```

---

## 📚 Références

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://docs.langchain.com)
- [Ollama](https://ollama.com)
- [Groq](https://groq.com)
- [IMDB Dataset – HuggingFace](https://huggingface.co/datasets/imdb)
- [Tiktoken](https://github.com/openai/tiktoken)
