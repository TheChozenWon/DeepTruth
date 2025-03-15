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

model = AutoModelForSequenceClassification.from_pretrained("google/bert-base-multilingual-cased")
print(model.config.num_labels)
