
    #I. PREPARING ESSENTIAL LIBRARIES
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, f1_score
import numpy as np

    #II.DATA COLLECTION AND PREPROCESSING
# Paths to data directories containing images
train_dir = r"C:/Users/levuo/PycharmProjects/african-wildlife/train"
validation_dir = r"C:/Users/levuo/PycharmProjects/african-wildlife/valid"

# Create ImageDataGenerators for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

    #III. BUILDING THE CNN MODEL
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.AveragePooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Function to adjust learning rate
def scheduler(epoch, lr):
    if epoch > 0 and epoch % 5 == 0:
        return lr * 0.5
    return lr

# Callback for Learning Rate Scheduling
lr_scheduler = LearningRateScheduler(scheduler)

    #IV. TRAINING THE MODEL
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[lr_scheduler]
)

    #V. EVALUATING THE MODEL
loss, acc = model.evaluate(validation_generator, verbose=2)
print(f"Validation accuracy: {acc*100:.2f} %")

# Plotting Loss and Accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Generating predictions
predictions = model.predict(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Generating classification report
print("Classification Report:")
if 'class_labels' not in locals() or len(class_labels) != len(set(true_classes)):
    class_labels = [f"Class {i}" for i in range(len(set(true_classes)))]
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Calculating F1 Score
f1 = f1_score(true_classes, predicted_classes, average='weighted')
print(f"Weighted F1 Score: {f1:.2f}")

# Visualizing learning rates
learning_rates = [scheduler(epoch, model.optimizer.learning_rate.numpy()) for epoch in range(20)]
plt.plot(range(20), learning_rates, marker='o')
plt.title('Learning Rate Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()

# Calculate Bias and Variance
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Bias as validation loss
bias = np.mean(val_loss)  # Average validation loss over epochs

# Variance as the gap between training and validation loss
variance = np.mean([val - train for val, train in zip(val_loss, train_loss)])

print(f"Estimated Bias: {bias:.4f}")
print(f"Estimated Variance: {variance:.4f}")

# Visualizing Bias and Variance
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.axhline(y=bias, color='r', linestyle='--', label=f"Bias (Mean Val Loss)")
plt.fill_between(epochs, train_loss, val_loss, color='gray', alpha=0.2, label="Variance (Gap)")
plt.title('Bias and Variance Visualization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

