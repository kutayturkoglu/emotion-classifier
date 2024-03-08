import torch
from tqdm import tqdm
def train_loop(epoches, model, train_loader, device, optimizer, criterion, val_loader):
    train_accuracy_arr = []
    val_accuracy_arr = []

    for epoch in range(epoches):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epoches}', leave=False)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _ , predicted = torch.max(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / train_total
        train_accuracy_arr.append(train_accuracy)
        # print("Train_loss: ", train_loss, "Train_accuracy: ", train_accuracy)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()*images.size(0)
            _,predicted = torch.max(outputs,1)
            val_total+=labels.size(0)
            val_correct += (predicted==labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * val_correct / val_total
        val_accuracy_arr.append(val_accuracy)
        # print("Val_loss: ", val_loss, "Val_accuracy: ", val_accuracy)