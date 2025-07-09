import numpy as np
import tensorflow as tf
import joblib

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from keras.models import Sequential, Model

from utils import create_mlp_model, plot_training_curves, load_model, build_mlp_functional

# Load
data = np.loadtxt("data/processed/dataset_personal.txt")
X = data[:, :-1]
y = data[:, -1]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# TF Datasets
batch_size = 16
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(200).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# # 1. Reconstruis le modèle avec la Functional API
# base_model = build_mlp_functional(input_dim=1404, num_classes=4)

# # 2. Charge les poids d'AffectNet (attention, même architecture et même nombre de classes requis ici)
# base_model.load_weights("model/2025-07-09:10/4classes_model.h5")

# Load the previously fine-tuned model (already has correct 4-class output)
finetune_model = load_model("model/fine_tune/affectnet_finetuned_on_self.h5")

# Geler toutes les couches (sauf les deux derniere couches)
for layer in finetune_model.layers[:-2]:
    layer.trainable = False

# # Récupère la sortie de la dernière couche cachée
# last_features = base_model.layers[-2].output  # <- avant softmax

# # Ajoute une nouvelle tête dense (4 classes)
# new_output = Dense(4, activation='softmax')(last_features)

# # Crée le nouveau modèle
# finetune_model = Model(inputs=base_model.input, outputs=new_output)

finetune_model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

history = finetune_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6, verbose=1)
    ]
)

finetune_model.save("model/fine_tune2/affectnet_finetuned_on_self2.h5")
joblib.dump(scaler, "model/fine_tune2/scaler_finetune.pkl")