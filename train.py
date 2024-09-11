import torch


def train_step(num_epochs, train_loader, test_loader, model, criterion, optimizer, pixel_accuracy, accuracy_use_softmax, one_hot, use_softmax=False):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        for images, masks in train_loader:

            outputs = model(images)
            if use_softmax:
                outputs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            acc = pixel_accuracy(
                outputs, masks, use_softmax=accuracy_use_softmax, one_hot=one_hot)
            train_acc += acc.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for images, masks in test_loader:

                outputs = model(images)
                if use_softmax:
                    outputs = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                acc = pixel_accuracy(
                    outputs, masks, use_softmax=accuracy_use_softmax, one_hot=one_hot)
                val_acc += acc.item()

        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_acc / len(test_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
