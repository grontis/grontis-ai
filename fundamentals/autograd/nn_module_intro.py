import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__() #DONT FORGET PARENT INIT!
        #Layers defined as attributes
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        #define how data flows through layers
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
    
model = MyModel()

for name, p in model.named_parameters():
    print(name, tuple(p.shape))

model = MyModel()
x = torch.randn(4, 10)
preds = model(x)
# preds = model.forward(x) #DONT DO THIS. hooks called by model(x) are not called