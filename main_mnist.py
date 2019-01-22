import torch
from torch.autograd import Variable
from utee import selector

model_raw, ds_fetcher, is_imagenet = selector.select('mnist')
ds_val = ds_fetcher(batch_size=10, train=False, val=True)

def validate():
    correct = 0
    total = 0

    X = []
    Ypred = []
    Ytarget = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(ds_val):
            data = Variable(torch.FloatTensor(data)).cuda()
            target = target.cuda()
            output = model_raw(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print("Accuracy", float(correct) / total)

validate()
