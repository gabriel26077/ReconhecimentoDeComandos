import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configurações ---
# ATENÇÃO: Aponte para a pasta com os ESPECTROGRAMAS de teste
TEST_IMG_DIR = 'dataset_testes/espectros_mel/'
MODEL_PATH = 'modelos_exportados/cnn_modelo_01_dataset_1.keras' # <-- Altere para o nome do seu modelo

# Configurações da imagem (DEVEM SER AS MESMAS DO TREINAMENTO)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Mapeamento de classes (A ORDEM DEVE SER A MESMA DO TREINAMENTO)
# Verifique a saída do script de treino para a ordem correta (ex: ['azul', 'verde', 'vermelho'])
CLASS_NAMES = ['azul', 'verde', 'vermelho'] # <-- Altere para suas classes, na ordem correta

# --- Script Principal ---

# 1. Carregar o modelo treinado
print(f"Carregando modelo de: {MODEL_PATH}")
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# 2. Preparar os dados de teste
X_test = []
y_true_labels = [] # Rótulos verdadeiros como strings

print(f"Carregando imagens de teste de: {TEST_IMG_DIR}")
test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')]

for file_name in tqdm(test_files):
    # Carrega e processa a imagem
    file_path = os.path.join(TEST_IMG_DIR, file_name)
    img = keras.preprocessing.image.load_img(
        file_path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    X_test.append(img_array)
    
    # Extrai o rótulo verdadeiro do nome do arquivo
    label = file_name.split('-')[0]
    y_true_labels.append(label)

# Converte a lista de imagens para um array NumPy e normaliza
X_test = np.array(X_test, dtype="float32") / 255.0

# 3. Fazer as previsões
predictions_prob = model.predict(X_test)
# A predição da CNN é um array de probabilidades. Pegamos o índice com maior valor.
y_pred_indices = np.argmax(predictions_prob, axis=1)

# Converte os rótulos verdadeiros (string) para índices para comparação
y_true_indices = [CLASS_NAMES.index(label) for label in y_true_labels]

# 4. Calcular e mostrar as métricas de desempenho
accuracy = accuracy_score(y_true_indices, y_pred_indices)
precision = precision_score(y_true_indices, y_pred_indices, average='weighted')
recall = recall_score(y_true_indices, y_pred_indices, average='weighted')
f1 = f1_score(y_true_indices, y_pred_indices, average='weighted')

print(f'\nAcurácia: {accuracy * 100:.2f}%')
print(f'Precisão: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-score: {f1 * 100:.2f}%')

# 5. Calcular e exibir a matriz de confusão
conf_matrix = confusion_matrix(y_true_indices, y_pred_indices)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Previsão do Modelo')
plt.ylabel('Rótulo Real')
plt.title('Matriz de Confusão')
plt.show()