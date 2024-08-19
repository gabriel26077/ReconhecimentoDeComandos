import os
import joblib
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Caminho para o diretório dos arquivos de áudio de teste e modelos
audio_test_dir = 'dataset_testes/'

model_path = 'modelos_exportados/knn_modelo_01_todos_os_dados.pkl'
scaler_path = 'scalers/scaler_knn_modelo_01_todos_os_dados.pkl'
#model_path = 'modelos_exportados/svm_modelo_01_todos_os_dados.pkl'
#scaler_path = 'scalers/scaler_svm_modelo_01_todos_os_dados.pkl'

# Carregar o modelo e o scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")
    exit()

# Listar todos os arquivos de áudio de teste
test_files = [f for f in os.listdir(audio_test_dir) if f.endswith('.wav')]

# Função para extrair características de um arquivo de áudio
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Extrair e normalizar características dos arquivos de teste
X_test = []
y_test = []

for file_name in test_files:
    file_path = os.path.join(audio_test_dir, file_name)
    features = extract_features(file_path)
    X_test.append(features)
    label = file_name.split('-')[0]  # Considerando que a cor é o rótulo
    y_test.append(label)

# Normalizar os dados de teste com o mesmo escalador usado nos dados de treino
X_test = scaler.transform(X_test)

# Prever as classes
y_pred = model.predict(X_test)

# Calcular e mostrar as métricas de desempenho
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Acurácia: {accuracy * 100:.2f}%')
print(f'Precisão: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-score: {f1 * 100:.2f}%')

# Calcular e imprimir a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
print("Matriz de Confusão:")
print(conf_matrix)

# Exibir a matriz de confusão usando um heatmap para facilitar a visualização
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
