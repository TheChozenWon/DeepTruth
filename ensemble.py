import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import ray

from ray import tune
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Define the Models and Tokenizers

# Models (HuggingFace)
MODEL_NAME_BERT = "google/bert-base-multilingual-cased"
MODEL_NAME_DISTIL = "distilbert-base-uncased"
MODEL_NAME_NLI = "roberta-large-mnli"

# Store "best" checkpoints after tuning
BEST_CHECKPOINT_BERT = "./best_ckpt_bert"
BEST_CHECKPOINT_DISTIL = "./best_ckpt_distil"
BEST_CHECKPOINT_NLI = "./best_ckpt_nli"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperperameter Search & Metric Functions

# Compute Metrics for Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}


"""Return a fresh BERT model for HF Trainer to initialize."""
def model_init_bert():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=2)


"""Return a fresh DistilBERT model for HF Trainer to initialize."""
def model_init_distil():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_DISTIL, num_labels=2)


"""Return a fresh NLI model (RoBERTa MNLI)."""
def model_init_nli():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_NLI, num_labels=3)


# Hyperparameter Space for bert
def ray_hp_space_bert(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

# Hyperperamater Space for DistillBERT
def ray_hp_space_distil(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

# Hyperperameter Space for NLI
def ray_hp_space_nli(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

# Data Loading and Preprocessing

# Load the datasets and tokenized it
def load_and_prepare_data(train_file, test_file, model_name, num_labels=2):
    raw_datasets = load_dataset("csv", data_files={
        "train": train_file,
        "test": test_file
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    # Apply tokenization
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # Rename label column if needed
    if "label" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set PyTorch format
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_datasets


# Train with Ray Tune

"""
1) Load and preprocess data
2) Create a Trainer with model_init
3) Perform hyperparameter search with Ray as backend
4) Return the best checkpoint path
"""
def train_with_raytune(model_init_func, model_name, train_file, test_file, hp_space_func, num_labels=2, output_dir="./results"):
    # Load data
    tokenized_data = load_and_prepare_data(train_file, test_file, model_name, num_labels=num_labels)
    train_dataset = tokenized_data["train"]
    eval_dataset = tokenized_data["test"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        disable_tqdm=False, 
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model_init=model_init_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        n_trials=20, 
        hp_space=hp_space_func,
        compute_objective=lambda metrics: metrics["eval_f1"],  # or eval_accuracy
    )

    print(f"Best run hyperparameters: {best_run.hyperparameters}")

    # Update Trainer with best hyperparams
    for k, v in best_run.hyperparameters.items():
        setattr(trainer.args, k, v)

    # Re-train with best hyperparams
    trainer.train()

    # Save the best model
    best_ckpt_path = os.path.join(output_dir, "best_model")
    trainer.save_model(best_ckpt_path)
    print(f"Saved best model checkpoint to {best_ckpt_path}")

    return best_ckpt_path

# Ensemble Inference

"""
1) Load each best checkpoint
2) Predict probabilities on the input texts
3) Average the softmax probabilities across models
4) Return final predictions

Args:
    texts (list of str): input sentences
    checkpoint_paths (list of str): paths to each best checkpoint
    label_counts (list of int): number of labels for each model (2 or 3)
"""
def ensemble_inference(texts, checkpoint_paths, label_counts, device=device):
    model_probs = []
    for ckpt_path, num_labels in zip(checkpoint_paths, label_counts):
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=num_labels)
        model.to(device)
        model.eval()

        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape [batch_size, num_labels]
            probs = F.softmax(logits, dim=-1)  # shape [batch_size, num_labels]
        model_probs.append(probs)

    # Stack probabilities from all models: shape [num_models, batch_size, num_labels]
    stacked_probs = torch.stack(model_probs, dim=0)
    # Average across num_models: shape [batch_size, num_labels]
    avg_probs = torch.mean(stacked_probs, dim=0)
    # Predicted class = argmax
    preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

    return preds, avg_probs.detach().cpu().numpy()


## Main Function

def main():
    parser = argparse.ArgumentParser(description="Train and ensemble multiple models with Ray Tune.")
    parser.add_argument("--train_file", type=str, default="balanced_train_data.csv", help="Path to training CSV")
    parser.add_argument("--test_file", type=str, default="balanced_test_data.csv", help="Path to testing CSV")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training + hyperparam search")
    parser.add_argument("--do_inference", action="store_true", help="Whether to run ensemble inference")
    parser.add_argument("--texts", nargs="+", default=["This is a test sentence."],
                        help="List of texts to run ensemble inference on")
    args = parser.parse_args()

    # Initialize Ray if using Ray backend
    if args.do_train:
        if not ray.is_initialized():
            ray.init()

        # 1) Train DistilBERT with RayTune
        distil_ckpt = train_with_raytune(
            model_init_func=model_init_distil,
            model_name=MODEL_NAME_DISTIL,
            train_file=args.train_file,
            test_file=args.test_file,
            hp_space_func=ray_hp_space_distil,
            num_labels=2,
            output_dir="./distil_results"
        )

        # 2) Train BERT with a separate or shared hp_space
        bert_ckpt = train_with_raytune(
            model_init_func=model_init_bert,
            model_name=MODEL_NAME_BERT,
            train_file=args.train_file,
            test_file=args.test_file,
            hp_space_func=ray_hp_space_bert, 
            num_labels=2,
            output_dir="./bert_results"
        )

        # 3) Train NLI with a separate or shared hp_space 
        #    If using 3 labels, set num_labels=3
        nli_ckpt = train_with_raytune(
            model_init_func=model_init_nli,
            model_name=MODEL_NAME_NLI,
            train_file=args.train_file,
            test_file=args.test_file,
            hp_space_func=ray_hp_space_nli,
            num_labels=3,
            output_dir="./nli_results"
        )

        print("Training complete. Best checkpoints:")
        print("DistilBERT:", distil_ckpt)
        print("BERT:", bert_ckpt)
        print("NLI:", nli_ckpt)

    if args.do_inference:
        # Example: if you already trained and have best checkpoints
        # DistilBERT is 2-label, BERT is 2-label, NLI is 3-label
        #checkpoint_paths = [BEST_CHECKPOINT_DISTIL, BEST_CHECKPOINT_BERT, BEST_CHECKPOINT_NLI]
        label_counts = [2, 2, 3]  # adjust if your final NLI model is 2-label

        # If you just trained them in the same run, you might do:
        checkpoint_paths = ["./distil_results/best_model", "./bert_results/best_model", "./nli_results/best_model"]

        preds, probs = ensemble_inference(args.texts, checkpoint_paths, label_counts, device=device)
        for text, pred, prob_vec in zip(args.texts, preds, probs):
            print(f"\nInput: {text}")
            print(f"Ensemble prediction: {pred}")
            print(f"Probabilities: {prob_vec}")

if __name__ == "__main__":
    main()