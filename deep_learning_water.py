"""
deep_learning_water.py
Tugas: Pertemuan 11 - Deep Learning (INDIVIDU)
Nama : Nabila Risha Juliana
NIM  : 434241017
Dataset: Water Potability (dari Kaggle)

Program ini membangun model Neural Network (Sequential MLP)
untuk klasifikasi biner kelayakan air minum (Potability).
"""

# ========================================================================
# 1. IMPORT LIBRARY
# ========================================================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf

# Set random seed untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ========================================================================
# 2. LOAD DATASET
# ========================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "water_potability.csv")
df = pd.read_csv(data_path)

print("=" * 65)
print("DEEP LEARNING - KLASIFIKASI KUALITAS AIR (Water Potability)")
print("=" * 65)
print(f"\n[1] Dataset dimuat")
print(f"    Shape: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"    Kolom: {list(df.columns)}")
print(f"\n    Preview 5 baris pertama:")
print(df.head())


# ========================================================================
# 3. PISAHKAN FITUR (X) DAN LABEL (y)
# ========================================================================
# Variabel Input  : 9 fitur numerik (pH, Hardness, Solids, dst)
# Variabel Output : Potability (0 = tidak layak, 1 = layak minum)

X = df.drop(columns=['Potability'])
y = df['Potability']

print(f"\n[2] Pemisahan Fitur (X) dan Label (y)")
print(f"    X (fitur) : {X.shape} -> 9 fitur numerik")
print(f"    y (label) : {y.shape} -> target biner (0/1)")
print(f"\n    Distribusi kelas y:")
print(f"    {dict(y.value_counts())}")
print(f"    Ratio: {y.value_counts(normalize=True).mul(100).round(2).to_dict()} %")


# ========================================================================
# 4. (SKIP) ENCODING - Dataset Water Potability tidak punya fitur kategorikal
# ========================================================================
# Catatan: Step encoding 'Umur' di modul tidak berlaku untuk water potability,
# karena seluruh fitur sudah bertipe numerik kontinu.
print(f"\n[3] Encoding: SKIP (semua fitur sudah numerik, tidak perlu encoding)")


# ========================================================================
# 5. PREPROCESSING DATA
# ========================================================================
print(f"\n[4] Preprocessing Data")

# a. Cek Missing Value
print(f"\n    a. Cek Missing Value:")
missing = X.isnull().sum()
print(missing[missing > 0])

# Isi missing value dengan median
X = X.fillna(X.median())
print(f"\n    -> Missing value diisi dengan MEDIAN tiap kolom")
print(f"    -> Total missing value sekarang: {X.isnull().sum().sum()}")

# b. Handle Outlier dengan IQR Clipping
print(f"\n    b. Handle Outlier (IQR Clipping):")
outlier_summary = {}
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outlier = ((X[col] < lower) | (X[col] > upper)).sum()
    outlier_summary[col] = n_outlier
    X[col] = X[col].clip(lower, upper)

for col, n in outlier_summary.items():
    print(f"       {col:25s}: {n} outlier di-clip")

# c. Feature Scaling dengan StandardScaler
print(f"\n    c. Feature Scaling (StandardScaler):")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"    -> Semua fitur di-standardize (mean=0, std=1)")
print(f"    -> Shape X_scaled: {X_scaled.shape}")


# ========================================================================
# 6. SPLIT DATA (80% train, 20% test)
# ========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n[5] Split Data 80:20 (stratified)")
print(f"    X_train : {X_train.shape}")
print(f"    X_test  : {X_test.shape}")
print(f"    y_train : {y_train.shape} -> {dict(y_train.value_counts())}")
print(f"    y_test  : {y_test.shape} -> {dict(y_test.value_counts())}")


# ========================================================================
# 7. BANGUN MODEL KLASIFIKASI (Sequential MLP)
# ========================================================================
print(f"\n[6] Bangun Arsitektur Neural Network")

model = Sequential([
    Input(shape=(X_train.shape[1],)),           # Input layer: 9 fitur
    Dense(16, activation='relu'),               # Hidden Layer 1: 16 neuron, ReLU
    Dense(8, activation='relu'),                # Hidden Layer 2: 8 neuron, ReLU
    Dense(1, activation='sigmoid')              # Output Layer: 1 neuron, Sigmoid
])

print(f"\n    Arsitektur Model:")
model.summary()


# ========================================================================
# 8. COMPILE DAN TRAIN
# ========================================================================
print(f"\n[7] Compile Model")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(f"    Optimizer : adam")
print(f"    Loss      : binary_crossentropy")
print(f"    Metrik    : accuracy")

print(f"\n[8] Training Model (100 epochs, validation_split=0.2)")
print(f"    (Proses ini bisa memakan waktu 1-3 menit)")
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1,
    batch_size=32
)


# ========================================================================
# 9. EVALUASI MODEL
# ========================================================================
print(f"\n[9] Evaluasi Model pada Data Testing")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"    Loss     : {loss:.4f}")
print(f"    Akurasi di data uji: {acc * 100:.2f}%")


# ========================================================================
# 10. CONFUSION MATRIX
# ========================================================================
# Prediksi probabilitas dan konversi ke kelas (threshold 0.5)
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)
print(f"\n[10] Confusion Matrix:")
print(f"                    Predicted 0   Predicted 1")
print(f"     Actual 0         {cm[0][0]:5d}         {cm[0][1]:5d}")
print(f"     Actual 1         {cm[1][0]:5d}         {cm[1][1]:5d}")


# ========================================================================
# 11. VISUALISASI CONFUSION MATRIX
# ========================================================================
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Layak (0)', 'Layak (1)'],
            yticklabels=['Tidak Layak (0)', 'Layak (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Deep Learning Water Potability')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'viz_confusion_matrix.png'), dpi=120, bbox_inches='tight')
print(f"\n[11] Confusion Matrix disimpan: viz_confusion_matrix.png")
plt.show()


# ========================================================================
# 12. CLASSIFICATION REPORT
# ========================================================================
print(f"\n[12] Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Tidak Layak (0)', 'Layak (1)']
))


# ========================================================================
# 13. VISUALISASI TRAINING HISTORY (BONUS)
# ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2E7D32')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#C62828')
axes[0].set_title('Model Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(history.history['loss'], label='Train Loss', color='#2E7D32')
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='#C62828')
axes[1].set_title('Model Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'viz_training_history.png'), dpi=120, bbox_inches='tight')
print(f"\n[13] Training History disimpan: viz_training_history.png")
plt.show()


# ========================================================================
# RINGKASAN AKHIR
# ========================================================================
print(f"\n" + "=" * 65)
print(f"  RINGKASAN HASIL DEEP LEARNING")
print(f"=" * 65)
print(f"  Dataset         : Water Potability ({df.shape[0]} sampel x {df.shape[1]} kolom)")
print(f"  Arsitektur      : Sequential MLP (9 -> 16 -> 8 -> 1)")
print(f"  Optimizer       : Adam")
print(f"  Loss Function   : Binary Crossentropy")
print(f"  Epochs          : 100")
print(f"  Akurasi Testing : {acc * 100:.2f}%")
print(f"\n  File Output     :")
print(f"    - viz_confusion_matrix.png")
print(f"    - viz_training_history.png")
print(f"=" * 65 + "\n")