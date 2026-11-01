from dataset.dataloader import create_dataloaders

train_loader, val_loader, test_loader, classes = create_dataloaders()

print()

print("Classes :", len(classes))

print("Train batches :", len(train_loader))

print("Validation batches :", len(val_loader))

print("Test batches :", len(test_loader))

images, labels = next(iter(train_loader))

print()

print("Image batch :", images.shape)

print("Label batch :", labels.shape)