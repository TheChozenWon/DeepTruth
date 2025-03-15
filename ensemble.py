import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

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

# ------------------------------------------------------------------------------
# 1. Metric Function
# ------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# ------------------------------------------------------------------------------
# 2. Model Init Functions
# ------------------------------------------------------------------------------
def model_init_bert():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_BERT, num_labels=2)

def model_init_distil():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_DISTIL, num_labels=2)

def model_init_nli():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_NLI, num_labels=3)

# ------------------------------------------------------------------------------
# 3. Hyperparameter Space (Originally named "ray_hp_space_*" but works for Optuna)
# ------------------------------------------------------------------------------
def ray_hp_space_bert(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

def ray_hp_space_distil(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

def ray_hp_space_nli(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

# ------------------------------------------------------------------------------
# 4. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
def load_and_prepare_data(train_file, test_file, model_name, num_labels=2):
    raw_datasets = load_dataset("csv", data_files={
        "train": train_file,
        "test": test_file
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        # If your CSV has "title" as the text column
        return tokenizer(examples["title"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # Rename label column if needed
    if "label" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_datasets

# ------------------------------------------------------------------------------
# 5. Train with Optuna (Replacing Ray)
# ------------------------------------------------------------------------------
def train_with_optuna(model_init_func, model_name, train_file, test_file, hp_space_func, num_labels=2, output_dir="./results"):
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

    # Switch to "optuna" backend
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=20,  # Adjust how many trials you want
        hp_space=hp_space_func,
        compute_objective=lambda metrics: metrics["eval_f1"],  # or "eval_accuracy"
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

# ------------------------------------------------------------------------------
# 6. Ensemble Inference
# ------------------------------------------------------------------------------
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
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
        model_probs.append(probs)

    stacked_probs = torch.stack(model_probs, dim=0)
    avg_probs = torch.mean(stacked_probs, dim=0)
    preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

    return preds, avg_probs.detach().cpu().numpy()

# ------------------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------------------
def main():
    import optuna  # Make sure optuna is installed
    parser = argparse.ArgumentParser(description="Train and ensemble multiple models with Optuna.")
    parser.add_argument("--train_file", type=str, default="balanced_train_data.csv", help="Path to training CSV")
    parser.add_argument("--test_file", type=str, default="balanced_test_data.csv", help="Path to testing CSV")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training + hyperparam search")
    parser.add_argument("--do_inference", action="store_true", help="Whether to run ensemble inference")
    parser.add_argument("--texts", nargs="+", default=["This is a test sentence."],
                        help="List of texts to run ensemble inference on")
    args = parser.parse_args()

    # 1) Train DistilBERT
    if args.do_train:
        distil_ckpt = train_with_optuna(
            model_init_func=model_init_distil,
            model_name=MODEL_NAME_DISTIL,
            train_file=args.train_file,
            test_file=args.test_file,
            hp_space_func=ray_hp_space_distil,
            num_labels=2,
            output_dir="./distil_results"
        )

        # 2) Train BERT
        bert_ckpt = train_with_optuna(
            model_init_func=model_init_bert,
            model_name=MODEL_NAME_BERT,
            train_file=args.train_file,
            test_file=args.test_file,
            hp_space_func=ray_hp_space_bert,
            num_labels=2,
            output_dir="./bert_results"
        )

        # 3) Train NLI
        nli_ckpt = train_with_optuna(
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
        label_counts = [2, 2, 3]
        checkpoint_paths = ["./distil_results/best_model", "./bert_results/best_model", "./nli_results/best_model"]
        preds, probs = ensemble_inference(args.texts, checkpoint_paths, label_counts, device=device)
        for text, pred, prob_vec in zip(args.texts, preds, probs):
            print(f"\nInput: {text}")
            print(f"Ensemble prediction: {pred}")
            print(f"Probabilities: {prob_vec}")

if __name__ == "__main__":
    main()
