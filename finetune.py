
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from load_dataset import train_ds, val_ds

# 1. Load previously trained model
model = tf.keras.models.load_model(
    "cats_dogs_mobilenetv2.keras"
)

# 2. Access pretrained MobileNetV2
base_model = model.layers[0]

# 3. Unfreeze the pretrained model
base_model.trainable = True

# 4. Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 5. Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# 6. Recompile with a low learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 7. Configure early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# 8. Fine-tune
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[early_stopping]
)

# 9. Save fine-tuned model
model.save("cats_dogs_finetuned.keras")

print("Fine-tuning completed!")
