import torch
from torch.autograd import Variable
from utee import selector

import numpy as np
from PIL import Image
from PIL import ImageOps

model_raw, ds_fetcher, is_imagenet = selector.select('mnist')
ds_val = ds_fetcher(batch_size=10, train=False, val=True)

img = Image.open('/tmp/image.png')
x = torch.tensor(np.array(img))
x = x.unsqueeze(0)
x = x.type(torch.FloatTensor).cuda()
x = 1 - x / 255.0
print(x.size())
output = model_raw(x)
_, predicted = torch.max(output.data, 1)
print("output", output)
print(predicted.item())
