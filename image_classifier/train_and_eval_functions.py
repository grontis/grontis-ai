def train_one_epoch(model, loss_fn, optimizer, device):
    model.train() # training mode
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        preds = model(images)           # forward -> (B,10) Logits
        loss = loss_fn(preds, labels)   # measure

        optimizer.zero_grad()           # clear old grads
        loss.backward()                 # compute grads
        optimizer.step()                # update weights

        running_loss += loss.item() * images.size(0) #sum, weighted by batch size
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct, total # avg loss, accuracy

def evaluate(model, loader, loss_fn, device): 
    model.eval()        # eval mode: dropout off, batchnorm frozen
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)

            running_loss += loss.item() * images.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total