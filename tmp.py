import torch
from transformers import BertModel, BertTokenizer


import faulthandler
faulthandler.enable()


# Choose a pre-trained BERT variant. This downloads weights automatically.
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Prepare some example input
text = "Hello, how are you today?"
inputs = tokenizer(text, return_tensors="pt").to(device)


# Run the model without compilation
with torch.no_grad():
    outputs = model(**inputs)
print("Without compile:", outputs.last_hidden_state.shape)


# Execute on CPU
inputs = inputs.to("cpu")
model = model.to("cpu")

compiled_model = torch.compile(model, backend="inductor")

# Run the compiled model
with torch.no_grad():
    compiled_outputs = compiled_model(**inputs)
print("With compile:", compiled_outputs.last_hidden_state.shape)

assert torch.allclose(outputs.last_hidden_state.cpu(), compiled_outputs.last_hidden_state.cpu(), atol=1e-3)
print("All good!")

