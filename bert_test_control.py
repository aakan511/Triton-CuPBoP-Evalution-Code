import timeit

SETUP_CODE = '''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import torch
from transformers import BertModel, BertTokenizer

#eliminating nondeterminstic behavior
import random
random.seed(0)
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
torch.use_deterministic_algorithms(True) # Eliminating random (non-deterministic) elements
# torch.backends.cudnn.benchmark = False # Does this need to be set in Triton as well because it would be calling the CUDA C?


model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

device = 'cpu'

text = "Hello, how are you today?"
inputs = tokenizer(text, return_tensors="pt") #.to(device)
model = model.to(device)

compiled_model = torch.compile(model, backend="onnxrt")
'''


#other potential compilers:  'onnxrt', 'tvm'
TEST_CODE = '''
with torch.no_grad():
    outputs = compiled_model(inputs)
'''

numTrials = 100
time = (timeit.timeit(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          number=numTrials)) / numTrials

# printing minimum exec. time
print(f'Time: {time}')
#print("With compile:", compiled_outputs.last_hidden_state.shape)
