from typing import Tuple
import os
import sys
import torch
# import fire
import time
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import json


from transformers import AutoModelForCausalLM

def pretty_print(s):
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(s, skip_special_tokens=True))


model = AutoModelForCausalLM.from_pretrained("gpt2")
print("model",model)

device = xm.xla_device()
model = model.to(device)



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
inputs = tokenizer(["Sky","Earth"],return_tensors='pt', padding="longest")
print(f"inputs = {inputs}" )
inputs = inputs.to(device)

outputs=model.generate(**inputs, max_new_tokens=80, do_sample=False)
print("outputs=")
print(outputs)   

for output in outputs:
    pretty_print(output)

