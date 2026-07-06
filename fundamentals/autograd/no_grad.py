import torch

w = torch.tensor(5.0, requires_grad=True)

#in place edit that does not error or corrupt the graph
#saves memory and time since nothing needs to be recorded
with torch.no_grad():
    w -= 0.1 * 2.0 #manual update, not tracked
    # w is still a leaf with requires_grad=True, just edited in place

print(w)

