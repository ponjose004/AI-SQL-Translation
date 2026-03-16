# 🧠 AI-SQL-TRANSLATOR — Natural Language to SQL Query Generator

> An AI-powered web application that converts plain English questions into SQL queries using a fine-tuned T5 transformer model.

---

## 🚀 Live Demo

🔗 **[https://job6742-ai-sql-translator.hf.space](https://job6742-ai-sql-translator.hf.space)**

> ⚠️ **Demo Notice:** This is a demo model trained on academic data. For best results, ask questions related to **students**, **classes**, **departments**, **employees**, or similar academic/institutional topics.
>
> Example queries:
> - *"What are the names of all male students?"*
> - *"How many students are there in the class?"*
> - *"List all departments with more than 10 employees."*

---

## 📦 Project Structure

```
EZOFIS/
├── app.py                    # Flask backend
├── Dockerfile                # Docker config for HF Spaces
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html            # Frontend UI
├── static/
│   ├── css/style1.css        # Styles
│   └── video/video.mp4       # Background video
└── t5/                       # Fine-tuned T5 model files
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── spiece.model
    └── pytorch_model.bin     # Hosted on Hugging Face Hub
```

---

## 🛠️ How I Deployed This (Simple Steps)

1. **Trained the model** locally using a custom academic dataset (described below)
2. **Uploaded model files** to [Hugging Face Hub](https://huggingface.co) (handles large files via Git LFS)
3. **Built a Flask app** (`app.py`) that loads the model and serves the UI
4. **Created a `Dockerfile`** so Hugging Face Spaces can run the app
5. **Created a `requirements.txt`** listing all Python packages needed
6. **Went to [huggingface.co/spaces](https://huggingface.co/spaces)** → New Space → chose **Docker** as SDK
7. **Uploaded all project files** to the Space via the Files tab
8. HF Spaces automatically **built and deployed** the app — live link generated instantly ✅

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Job6742/EZOFIS.git
cd EZOFIS

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open your browser at `http://localhost:5000`

---

## 🧪 Model Training Pipeline

The T5 model was fine-tuned on a custom academic dataset containing natural language questions paired with their corresponding SQL queries. Below is the complete step-by-step pipeline used.

---

### Step 1 — Dataset Preparation

A custom CSV dataset (`student.csv`) was created containing two columns:
- `question` — Natural language question (e.g., *"What are the names of all male students?"*)
- `sql` — Corresponding SQL query (e.g., *"SELECT name FROM Your_table_Name WHERE gender = 'male';"*)

The dataset covered academic and employee-related topics including student records, class details, department information, and staff data. In total, **395 question-SQL pairs** were used for training.

---

### Step 2 — Text Preprocessing

Each question and SQL query was cleaned before feeding into the model:

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()                                      # Lowercase
    text = re.sub(r'[^\w\s]', '', text)                      # Remove punctuation
    tokens = word_tokenize(text)                             # Tokenize
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in tokens if t not in stop_words]  # Remove stopwords
    return " ".join(filtered_tokens)
```

This step normalized the text by removing noise and unnecessary words, making the data cleaner for the model to learn from.

---

### Step 3 — Train/Test Split

The dataset was split into **80% training** and **20% testing** using scikit-learn:

```python
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```

This ensures the model is evaluated on unseen data to measure real-world performance.

---

### Step 4 — Base Model Selection

Instead of training from scratch, a pre-trained T5 model already fine-tuned on SQL tasks was used as the starting point:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-small-finetuned-wikiSQL")
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-small-finetuned-wikiSQL")
```

**Why this model?** `mrm8488/t5-small-finetuned-wikiSQL` is a T5-small model already trained on the WikiSQL dataset, meaning it already understands the structure of SQL. Fine-tuning it further on our custom academic data allows it to specialize in domain-specific queries without needing massive compute resources.

**T5 Architecture Highlights:**
- Encoder-Decoder architecture (512 hidden dimensions, 6 layers, 8 attention heads)
- Vocabulary size: 32,128 tokens
- Treats every NLP task as a text-to-text problem
- Input format: `"translate English to SQL: <question> </s>"`

---

### Step 5 — Tokenization and Encoding

The preprocessed questions and SQL queries were tokenized and padded to a fixed length of 128 tokens:

```python
train_encodings = tokenizer(
    train_questions,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=128
)

train_labels = tokenizer(
    train_queries,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
    max_length=128
)
```

Padding ensures all sequences in a batch are the same length, which is required for efficient GPU/CPU computation.

---

### Step 6 — Custom Dataset Class

A PyTorch `Dataset` class was implemented to properly structure the training data:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels.input_ids[idx]
        return item

    def __len__(self):
        return len(self.encodings.input_ids)
```

This wraps the tokenized data in a format compatible with the HuggingFace `Trainer` API.

---

### Step 7 — Training Configuration

The model was trained using the HuggingFace `Trainer` API with the following hyperparameters:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,             # 50 full passes over the training data
    per_device_train_batch_size=8,   # 8 samples per training step
    per_device_eval_batch_size=16,   # 16 samples per evaluation step
    warmup_steps=500,                # Gradually increase LR for first 500 steps
    weight_decay=0.01,               # Regularization to prevent overfitting
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
```

**Key training choices explained:**
- **50 epochs** — The dataset is relatively small (395 samples), so more epochs help the model learn the patterns well
- **Batch size 8** — Balances memory usage and training stability
- **Warmup steps 500** — Prevents large gradient updates at the start of training
- **Weight decay 0.01** — L2 regularization to reduce overfitting on the small dataset

Total optimization steps: **2,500** (395 samples ÷ 8 batch size × 50 epochs ≈ 2,500 steps)

---

### Step 8 — Model Saving

After training, the fine-tuned model and tokenizer were saved locally:

```python
output_dir = "text2sql"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
```

This produces the model files (`pytorch_model.bin`, `config.json`, tokenizer files) that were later uploaded to Hugging Face Hub for deployment.

---

### Step 9 — Inference (Query Generation)

The trained model generates SQL from natural language using this function:

```python
def get_sql(query):
    input_text = "translate English to SQL: %s </s>" % query
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(
        input_ids=features['input_ids'],
        attention_mask=features['attention_mask']
    )
    return tokenizer.decode(output[0])
```

The output is then post-processed to clean up special tokens:

```python
cleaned = output
    .replace("<pad>", "")
    .replace("</s>", " ;")
    .replace("<unk>", "=")
    .replace("table ", "Your_table_Name ")
```

---

### Training Pipeline Summary

```
Custom CSV Dataset (395 Q&A pairs)
        ↓
Text Preprocessing (lowercase, remove punctuation, stopword removal)
        ↓
Train/Test Split (80/20)
        ↓
Base Model: mrm8488/t5-small-finetuned-wikiSQL
        ↓
Tokenization & Encoding (max_length=128, padding)
        ↓
Fine-tuning (50 epochs, batch_size=8, AdamW optimizer)
        ↓
Saved Model (pytorch_model.bin + tokenizer files)
        ↓
Flask Web App → Hugging Face Spaces → Live URL
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | T5-small (fine-tuned) |
| ML Framework | PyTorch + HuggingFace Transformers |
| Backend | Flask (Python) |
| Frontend | HTML, CSS |
| Deployment | Hugging Face Spaces (Docker) |
| Model Hosting | Hugging Face Hub |

---

## 📋 Requirements

```
flask
transformers
torch
sentencepiece
huggingface_hub
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Job6742**
- Hugging Face: [@Job6742](https://huggingface.co/Job6742)
- GitHub: [@Job6742](https://github.com/Job6742)

---

> 💡 *This project was developed as a demo to showcase fine-tuning a T5 transformer model on a custom domain-specific NLP-to-SQL task using academic and employee data.*
