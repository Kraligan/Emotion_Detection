from utils import create_mlp_model, plot_training_curves, plot_confusion_matrix
import pickle
import numpy as np
import joblib

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight


# Create basic tensorflow model (3 dense layer?)
# model_fer2013 = create_mlp_model(num_classes=7)
# model_fer2013.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print(model_fer2013.summary())


model_affectnet = create_mlp_model(num_classes=6)
optimizer = Adam(learning_rate=0.00001)
model_affectnet.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model_affectnet.summary())

# load data 
data = np.loadtxt("data/dataset_Affectnet_balanced.txt")
X = data[:, :-1]  # landmarks (shape: [n_samples, 1404])
y = data[:, -1]   # labels (shape: [n_samples])

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split train/test/val
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# Convert to tf.data.Dataset
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(500).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# train model
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

history = model_affectnet.fit(
    train_ds,
    validation_data = val_ds,
    epochs=50,
    callbacks = [
        ModelCheckpoint('model/affectnet_opti/best_model.h5', save_best_only=True),
        EarlyStopping(patience=5, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ], 
    class_weight=dict(enumerate(class_weights))
)

joblib.dump(scaler, 'model/affectnet_opti/scaler_affectnet.pkl')

# Plot training curves
plot_training_curves(history=history)

# Evaluate on test set
test_loss, test_acc = model_affectnet.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

plot_confusion_matrix(model_affectnet, X_test, y_test, class_names=[
    'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
])



