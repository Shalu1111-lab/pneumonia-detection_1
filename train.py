# train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load npz file
data = np.load('pneumoniamnist.npz')
print("Files in npz:", data.files)

# Extract images and labels
images = data['train_images']   # shape: (4708, 28, 28)
labels = data['train_labels']   # shape: (4708,)

# Preprocess images: add channel dim, resize, convert to RGB
images = np.expand_dims(images, -1)   # shape: (4708, 28, 28, 1)
images = tf.image.resize(images, (128,128))
images = tf.image.grayscale_to_rgb(images)   # shape: (4708, 128, 128, 3)
images = images.numpy()
# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
# Load pretrained InceptionV3
base_model = tf.keras.applications.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)
base_model.trainable = False  # Freeze the base model

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_val, y_val),
    batch_size=32
)
# Compute F1-score
y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype(int)
f1 = f1_score(y_val, y_pred)
print("F1-score on validation set:", f1)
# Save the model in modern Keras format
model.save('pneumonia_inceptionv3.keras')
print("✅ Training complete!")
import matplotlib.pyplot as plt

# Plot a few validation images with predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for img, true_label, ax in zip(X_val[:10], y_val[:10], axes):
    pred_prob = model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_label = int(pred_prob > 0.5)
    ax.imshow(img.astype("uint8"))
    ax.set_title(f'True: {int(true_label)}, Pred: {pred_label}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('examples_inceptionv3.png')
plt.close()
print("🖼 Saved prediction sample as examples_inceptionv3.png")
