from load_dataset import train_ds, val_ds
from model import model

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("cats_dogs_mobilenetv2.keras")

print("Training completed!")