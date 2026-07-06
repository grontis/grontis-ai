from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/fasion_cnn") # runs is gitignored

for epoch in range(num_epochs):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)

    writer.add_scalar("loss/train", tr_loss, epoch)
    writer.add_scalar("loss/test", te_loss, epoch)
    writer.add_scalar("acc/train", tr_acc, epoch)
    writer.add_scalar("acc/train", te_acc, epoch)

writer.close()