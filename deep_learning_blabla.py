"""
deep_learning_blabla.py
Tugas: Pertemuan 11 - Deep Learning (INDIVIDU)
Nama : Nabila Risha Juliana
NIM  : 434241017
Dataset: BlaBla.xlsx

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
from sklearn.metrics import confusion_matrix, classification_report

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
data_path = os.path.join(base_dir, "data/BlaBla.xlsx")
df = pd.read_excel(data_path)

print("=" * 65)
print("DEEP LEARNING - KLASIFIKASI BINER (Dataset BlaBla)")
print("=" * 65)
print(f"\n[1] Dataset dimuat")
print(f"    Shape: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"    Kolom: {list(df.columns)}")
print(f"\n    Preview 5 baris pertama:")
print(df.head())


# ========================================================================
# 3. PISAHKAN FITUR DAN LABEL
# ========================================================================
# Variabel Input  : Umur (hasil encoding) + B sampai M = 13 fitur
# Variabel Output : N (target biner 0/1)
print(f"\n[2] Pisahkan Fitur dan Label")
print(f"    Variabel Input  : Umur, B, C, D, E, F, G, H, I, J, K, L, M (13 fitur)")
print(f"    Variabel Output : N")


# ========================================================================
# 4. ENCODING UMUR SESUAI MODUL
# Rules:
#   1 = <= 20
#   2 = >= 21 dan <= 30
#   3 = >= 31 dan <= 40
#   4 = >= 41 dan <= 50
#   5 = > 50
# ========================================================================
print(f"\n[3] Encoding UMUR_TAHUN ke Kode Umur")

def encode_umur(umur):
    """Encoding umur sesuai ketentuan modul Pertemuan 11."""
    umur = int(umur)
    if umur <= 20:
        return 1
    elif umur <= 30:
        return 2
    elif umur <= 40:
        return 3
    elif umur <= 50:
        return 4
    else:
        return 5

# Pastikan UMUR_TAHUN bertipe numerik, isi NaN dengan median kalau ada
df['UMUR_TAHUN'] = pd.to_numeric(df['UMUR_TAHUN'], errors='coerce')
if df['UMUR_TAHUN'].isna().any():
    n_bad = df['UMUR_TAHUN'].isna().sum()
    median_umur = df['UMUR_TAHUN'].median()
    df['UMUR_TAHUN'] = df['UMUR_TAHUN'].fillna(median_umur)
    print(f"    Catatan: ditemukan {n_bad} nilai non-numerik di UMUR_TAHUN, diisi median ({median_umur})")

df['Umur_Encoded'] = df['UMUR_TAHUN'].apply(encode_umur)

print(f"    Rules Encoding:")
print(f"        1 = umur <= 20")
print(f"        2 = umur 21-30")
print(f"        3 = umur 31-40")
print(f"        4 = umur 41-50")
print(f"        5 = umur > 50")

print(f"\n    Contoh hasil encoding (5 sampel):")
print(df[['UMUR_TAHUN', 'Umur_Encoded', 'A']].head())

print(f"\n    Distribusi kode umur:")
print(df['Umur_Encoded'].value_counts().sort_index().to_string())

# Validasi: pastikan kolom Umur_Encoded sama dengan kolom A (jika ada)
print(f"\n    Validasi dengan kolom A: {(df['Umur_Encoded'] == df['A']).sum()}/{len(df)} baris cocok")


# ========================================================================
# 5. PREPROCESSING DATA
# ========================================================================
# Pisahkan input dan output
X = df[['Umur_Encoded', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']].copy()
y = df['N']

print(f"\n[4] Preprocessing Data")
print(f"    X (input) : {X.shape} -> 13 fitur")
print(f"    y (output): {y.shape} -> target biner (0/1)")
print(f"\n    Distribusi kelas target N:")
print(f"    {dict(y.value_counts())}")
print(f"    Rasio: {y.value_counts(normalize=True).mul(100).round(2).to_dict()} %")

# a. Cek Missing Value
print(f"\n    a. Cek Missing Value:")
total_nan = X.isnull().sum().sum() + y.isnull().sum()
print(f"       Total missing di X : {X.isnull().sum().sum()}")
print(f"       Total missing di y : {y.isnull().sum()}")
if total_nan > 0:
    X = X.fillna(X.median())
    print(f"       -> Missing diisi dengan median")
else:
    print(f"       -> Tidak ada missing value")

# b. Cek Outlier dengan IQR (hanya untuk fitur Umur_Encoded karena B-M sudah 0/1)
print(f"\n    b. Cek Outlier (IQR):")
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outlier = ((X[col] < lower) | (X[col] > upper)).sum()
    if n_outlier > 0:
        print(f"       {col:15s}: {n_outlier} outlier (di-clip ke batas IQR)")
        X[col] = X[col].clip(lower, upper)
    else:
        print(f"       {col:15s}: 0 outlier")

# c. Feature Scaling dengan StandardScaler
print(f"\n    c. Feature Scaling (StandardScaler):")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"       -> Semua fitur di-standardize (mean=0, std=1)")
print(f"       -> Shape X_scaled: {X_scaled.shape}")


# ========================================================================
# 6. SPLIT DATA (80% training, 20% testing)
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
    Input(shape=(X_train.shape[1],)),           # Input: 13 fitur
    Dense(16, activation='relu'),               # Hidden Layer 1
    Dense(8, activation='relu'),                # Hidden Layer 2
    Dense(1, activation='sigmoid')              # Output: biner
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
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)
print(f"\n[10] Confusion Matrix:")
print(f"                    Predicted 0   Predicted 1")
print(f"     Actual 0         {cm[0][0]:5d}         {cm[0][1]:5d}")
print(f"     Actual 1         {cm[1][0]:5d}         {cm[1][1]:5d}")


# ========================================================================
# 11. TAMPILKAN CONFUSION MATRIX (HEATMAP)
# ========================================================================
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0, 1],
            yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Deep Learning BlaBla')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'viz_confusion_matrix.png'), dpi=120, bbox_inches='tight')
print(f"\n[11] Confusion Matrix disimpan: viz_confusion_matrix.png")
plt.show()


# ========================================================================
# 12. LAPORAN KLASIFIKASI
# ========================================================================
print(f"\n[12] Classification Report:")
print(classification_report(y_test, y_pred))


# ========================================================================
# 13. VISUALISASI TRAINING HISTORY (BONUS)
# ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2E7D32')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#C62828')
axes[0].set_title('Model Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

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
print(f"  Dataset         : BlaBla.xlsx ({df.shape[0]} sampel)")
print(f"  Jumlah Fitur    : 13 (Umur_Encoded + B-M)")
print(f"  Arsitektur      : Sequential MLP (13 -> 16 -> 8 -> 1)")
print(f"  Optimizer       : Adam")
print(f"  Loss Function   : Binary Crossentropy")
print(f"  Epochs          : 100")
print(f"  Akurasi Testing : {acc * 100:.2f}%")
print(f"\n  File Output     :")
print(f"    - viz_confusion_matrix.png")
print(f"    - viz_training_history.png")
print(f"=" * 65 + "\n")