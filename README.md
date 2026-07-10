

## DistilBERT Model

### Architecture

Fine-tuned **DistilBERT-base-uncased** (66.9M parameters) for binary sexism classification using HuggingFace's `Trainer`. The model takes tokenized text (max 128 tokens) and outputs a probability for two classes: `not sexist` and `sexist`.

### Training

The model is trained on the EDOS labelled dataset with ~14,000 training samples. Training runs for 5 epochs with batch size 16, learning rate 2e-5, and AdamW optimizer.

### Class Imbalance

The dataset has a **76/24 split** (not sexist / sexist), addressed through:

- **Balanced class weights** — computed via `sklearn.utils.class_weight.compute_class_weight`
- **Focal Loss** — replaces cross-entropy with a gamma parameter that down-weights easy examples and focuses on hard-to-classify samples

### Experiments & Results

We compared standard CrossEntropy (baseline) against Focal Loss with different gamma values:

| Version | Gamma | F1 (sexist) | Precision | Recall | FN | FP | Accuracy |
|---------|:-----:|:-----------:|:---------:|:------:|:-:|:-:|:--------:|
| Baseline (CE) | — | 0.72 | 0.68 | 0.76 | 234 | 352 | 85% |
| Focal Loss | **1.0** | **0.72** | **0.69** | **0.76** | **234** | **333** | **86%** |
| Focal Loss | 1.5 | 0.71 | 0.66 | 0.78 | 210 | 399 | 85% |
| Focal Loss | 2.0 | 0.59 | 0.43 | 0.94 | — | — | 82% |
| Focal Loss | 3.0 | ~0.50 | ~0.30 | ~0.98 | — | — | ~78% |

**Gamma=1.0** was the best performer — matching baseline F1 while reducing false positives by **19 posts** (333 vs 352) and achieving the highest overall accuracy (86%). Higher gamma values (2.0, 3.0) caused precision to collapse as the model over-predicted the minority class.

### Threshold Tuning

After training, `precision_recall_curve` on the dev set found the optimal decision threshold. For gamma=1.0 the tuned threshold was 0.498 — effectively equivalent to the default 0.5, indicating no custom threshold is needed.

### GPU Training via Google Colab

The training script is GPU-ready — HuggingFace's `Trainer` automatically detects and uses CUDA when available. Training on CPU takes several hours, while a free **T4 GPU on Google Colab** completes 5 epochs in **~6 minutes** — roughly **40x faster**.

To train on Colab:

1. Upload `train_data.safetensors`, `dev_data.safetensors`, `test_data.safetensors`, and `config.pt` to Google Drive
2. Open `train_model_distilbert.py` in Colab and change `_base` to your Drive path
3. Set runtime to **T4 GPU** (Runtime → Change runtime type)
4. Run `!pip install transformers torch safetensors scikit-learn`
5. Execute the script

### Inference

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

model_path = "./results/checkpoint-XXXX"  # replace with actual checkpoint
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def predict(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = (probs[:, 1] >= threshold).int().item()
    return "sexist" if pred else "not sexist"

print(predict("Women belong in the kitchen"))   # "sexist"
print(predict("Great weather today"))           # "not sexist"
Use threshold 0.5 (optimal for gamma=1.0) or adjust for precision/recall tradeoff