from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Path to the directory containing config.json and pytorch_model.bin
model_path = 'E:/DeepTruth/backend/distill_results/best_model'  # Replace with your directory

# Load the tokenizer (assumes tokenizer files are in the same directory)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load the model (requires pytorch_model.bin in the directory)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Function to analyze a claim and return a confidence score
def analyze_claim(title: str) -> float:
    # Tokenize the input
    inputs = tokenizer(
        title,
        return_tensors="pt",  # PyTorch tensors
        truncation=True,      # Truncate to max length
        max_length=512,       # Max sequence length
        padding=True          # Pad to max length
    )
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run the model in evaluation mode (no gradient computation)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence_score = probabilities[0][1].item()  # Assuming 1 is the positive class
    return confidence_score

# Example usage
claim = "This is a test claim."
score = analyze_claim(claim)
print(f"Confidence score: {score}")