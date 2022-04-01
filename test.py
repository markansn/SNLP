# from datasets import load_dataset
#
# dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0")

import torch
foo = torch.tensor([1,2,3])
foo = foo.to('cuda')