import os
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from pathlib import Path

_base = Path(__file__).resolve().parent.parent


class SexismDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.as_tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


def load_data():
    train_data = load_file(_base / "models" / "train_data.safetensors")
    dev_data = load_file(_base / "models" / "dev_data.safetensors")
    test_data = load_file(_base / "models" / "test_data.safetensors")
    config = torch.load(_base / "models" / "config.pt", weights_only=False)

    train_dataset = SexismDataset(
        encodings={
            "input_ids": train_data["input_ids"],
            "attention_mask": train_data["attention_mask"],
        },
        labels=train_data["labels"].tolist(),
    )
    dev_dataset = SexismDataset(
        encodings={
            "input_ids": dev_data["input_ids"],
            "attention_mask": dev_data["attention_mask"],
        },
        labels=dev_data["labels"].tolist(),
    )
    test_dataset = SexismDataset(
        encodings={
            "input_ids": test_data["input_ids"],
            "attention_mask": test_data["attention_mask"],
        },
        labels=test_data["labels"].tolist(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    return train_loader, dev_loader, test_loader, config


def load_model(num_labels=2):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    return model


if __name__ == "__main__":
    print("Loading data...")
    train_loader, dev_loader, test_loader, config = load_data()

    print("Loading model...")
    model = load_model(num_labels=2)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
# Prueba Dummy 
def test_with_dummy_data(model):
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones((2, 128)),
        "labels": torch.tensor([0, 1])
    }
    output = model(**dummy_input)
    print(f"Dummy test exitoso. Loss: {output.loss.item()}")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,            # Solo 1 época para la prueba
    max_steps=10,                  # Solo 10 pasos (iterations)
    per_device_train_batch_size=4, # Batch pequeño para ir rápido
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="steps",   # Evaluar por pasos
    eval_steps=5,                  # Evaluar a los 5 pasos
    save_strategy="no",            # No guardar nada todavía
    load_best_model_at_end=False,
)

# Calcular métricas 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    # Métricas con sklearn
    f1 = f1_score(labels, predictions, average="binary")
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")
    
    return {"f1": f1, "precision": precision, "recall": recall}

# Entrenamiento y Validación (Custom Trainer)
class CustomTrainer(Trainer):
    # Añadimos num_items_in_batch=None para evitar el error
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Usamos class_weights aquí
        loss_fct = torch.nn.CrossEntropyLoss(weight=config["class_weights"])
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=dev_loader.dataset,
    compute_metrics=compute_metrics,
)

print("Iniciando entrenamiento...")
trainer.train()    
    

    

