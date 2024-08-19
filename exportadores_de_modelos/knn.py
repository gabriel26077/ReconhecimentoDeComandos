import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Lista de diretórios dos arquivos de áudio
audio_dirs = ['dataset/', 'dataset_extra/']

# Combinar todos os arquivos de áudio em uma lista única
audio_files = []
for dir in audio_dirs:
    audio_files.extend([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.wav')])

# Função para extrair características de um arquivo de áudio
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Extrair características dos arquivos de treino e criar os rótulos
X_train = []
y_train = []

for file_path in audio_files:
    features = extract_features(file_path)
    X_train.append(features)
    label = os.path.basename(file_path).split('-')[0]  # Considerando que a cor é o rótulo
    y_train.append(label)

# Normalizar os dados de treino
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Treinar um modelo KNN
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Exportar o modelo treinado e o escalador
nome_modelo = input("Nome do modelo: ")

# Cria os diretórios se não existirem
os.makedirs('modelos_exportados', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

# Salvar o modelo e o escalador
joblib.dump(model, f'modelos_exportados/{nome_modelo}.pkl')
joblib.dump(scaler, f'scalers/scaler_{nome_modelo}.pkl')

print("Modelo KNN e escalador exportados com sucesso!")
