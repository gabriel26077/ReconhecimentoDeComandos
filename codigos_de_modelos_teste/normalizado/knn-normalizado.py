import os
import librosa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Lista de diretórios dos arquivos de áudio
audio_dirs = [
    'dataset/',
    # Adicione mais diretórios se necessário
]

# Combinar todos os arquivos de áudio em uma lista única
audio_files = []
for dir in audio_dirs:
    audio_files.extend([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.wav')])

# Dividir os arquivos de áudio em treino e teste (80% treino e 20% teste)
train_files, test_files = train_test_split(audio_files, test_size=0.2, random_state=42)

# Função para extrair características de um arquivo de áudio
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Extrair características dos arquivos de treino e criar os rótulos
X_train = []
y_train = []

for file_path in train_files:
    features = extract_features(file_path)
    X_train.append(features)
    label = os.path.basename(file_path).split('-')[0]  # Considerando que a cor é o rótulo
    y_train.append(label)

# Normalizar os dados de treino
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Treinar um modelo k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Extrair e normalizar características dos arquivos de teste
X_test = []
y_test = []

for file_path in test_files:
    features = extract_features(file_path)
    X_test.append(features)
    label = os.path.basename(file_path).split('-')[0]
    y_test.append(label)

# Normalizar os dados de teste com o mesmo escalador usado nos dados de treino
X_test = scaler.transform(X_test)

# Prever e calcular métricas
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

# Resultados
print(set(y_train))
print(f'Acurácia: {accuracy * 100:.2f}%')
print(f'Precisão: {precision * 100:.2f}%')
print(f'Revocação: {recall * 100:.2f}%')

# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_train), yticklabels=set(y_train))
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
