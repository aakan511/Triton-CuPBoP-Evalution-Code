import timeit

SETUP_CODE = '''
from transformers import AutoImageProcessor, ResNetModel
import torch
from datasets import load_dataset

import random
random.seed(0)
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
torch.use_deterministic_algorithms(True) # Eliminating random (non-deterministic) elements
torch.backends.cudnn.benchmark = False
 
dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]
 
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetModel.from_pretrained("microsoft/resnet-50")
 
inputs = image_processor(image, return_tensors="pt")
 
device = 'cpu'
model = model.to(device)
 
# Run the compiled model
with torch.no_grad():
    outputs = model(**inputs)
'''

TEST_CODE = '''
with torch.no_grad():
    outputs = model(**inputs)
'''
numTrials = 100
time = (timeit.timeit(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          number=numTrials)) / 100

    # printing minimum exec. time
print(f'Time: {time}')
#print("With compile:", compiled_outputs.last_hidden_state.shape)
