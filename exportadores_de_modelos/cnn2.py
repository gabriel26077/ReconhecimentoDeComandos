import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Configurações ---
SPECTROGRAM_DIR = 'dataset/espectros_mel'
IMG_HEIGHT = 128      # Altura da imagem para redimensionar
IMG_WIDTH = 128       # Largura da imagem para redimensionar
EPOCHS = 25           # Número de épocas para o treinamento
BATCH_SIZE = 32       # Tamanho do lote
TEST_SIZE = 0.15      # Parcela dos dados para o conjunto de teste
VALIDATION_SIZE = 0.15 # Parcela dos dados para o conjunto de validação

def carregar_e_processar_dados(diretorio):
    """Carrega as imagens, processa e retorna os dados e o codificador de rótulos."""
    print("Iniciando carregamento e pré-processamento dos dados...")

    imagens = []
    rotulos_str = []

    arquivos_png = [f for f in os.listdir(diretorio) if f.endswith('.png')]

    for nome_arquivo in tqdm(arquivos_png, desc="Processando Imagens"):
        caminho_img = os.path.join(diretorio, nome_arquivo)

        img = keras.preprocessing.image.load_img(
            caminho_img, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        imagens.append(img_array)

        rotulo = nome_arquivo.split('-')[0]
        rotulos_str.append(rotulo)

    imagens = np.array(imagens, dtype="float32")
    rotulos_str = np.array(rotulos_str)

    # Normaliza os pixels antes da augmentation
    imagens /= 255.0

    # Codifica os rótulos de string para números
    le = LabelEncoder()
    rotulos_int = le.fit_transform(rotulos_str)

    # Converte para one-hot
    num_classes = len(le.classes_)
    rotulos_cat = keras.utils.to_categorical(rotulos_int, num_classes=num_classes)

    print(f"\nDados carregados. Encontradas {num_classes} classes: {list(le.classes_)}")

    # Retorna também o 'le' para usarmos os nomes das classes no final
    return imagens, rotulos_cat, num_classes, le

def construir_modelo_cnn(input_shape, num_classes):
    """Cria e compila a arquitetura da CNN com Data Augmentation."""
    
    # Camadas para aumento de dados (Data Augmentation)
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ],
        name="data_augmentation",
    )
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        # Aplica o aumento de dados como a primeira camada.
        # Essas camadas só ficam ativas durante o treinamento.
        data_augmentation,
        
        # O resto da arquitetura da CNN
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plotar_historico_treinamento(history):
    """Plota os gráficos de acurácia e perda do treinamento e validação."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 6))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia de Treino')
    plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Evolução da Acurácia', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')


    # Gráfico de Perda (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perda de Treino')
    plt.plot(epochs_range, val_loss, label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Evolução da Perda (Loss)', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Perda')

    plt.suptitle('Desempenho do Modelo ao Longo das Épocas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta para o supertítulo não sobrepor
    plt.show()


# --- Script Principal ---
if __name__ == '__main__':
    # 1. Carregar e preparar os dados
    X, y, num_classes, label_encoder = carregar_e_processar_dados(SPECTROGRAM_DIR)

    # 2. Dividir os dados em treino, validação e TESTE
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    val_size_relative = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative, random_state=42, stratify=y_temp
    )

    print(f"\nDivisão dos dados:")
    print(f"  - Treino:    {len(X_train)} amostras")
    print(f"  - Validação: {len(X_val)} amostras")
    print(f"  - Teste:     {len(X_test)} amostras")

    # 3. Construir o modelo
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
    model = construir_modelo_cnn(input_shape, num_classes)

    print("\n--- Arquitetura do Modelo ---")
    model.summary()
    print("---------------------------\n")

    # 4. Treinar o modelo
    print(f"Iniciando treinamento por {EPOCHS} épocas...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )

    # 5. PLOTAR GRÁFICOS DE TREINAMENTO
    print("\n--- Exibindo gráficos de desempenho do treinamento ---")
    plotar_historico_treinamento(history)

    # --- AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ---
    print("\n--- Avaliando o modelo no conjunto de teste ---")
    
    # 6. Fazer previsões no conjunto de teste
    y_pred_probs = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    y_true_indices = np.argmax(y_test, axis=1)

    # 7. Calcular e mostrar as métricas de desempenho
    accuracy = accuracy_score(y_true_indices, y_pred_indices)
    precision = precision_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)
    recall = recall_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)
    f1 = f1_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)

    print(f'\nAcurácia: {accuracy * 100:.2f}%')
    print(f'Precisão:  {precision * 100:.2f}%')
    print(f'Recall:    {recall * 100:.2f}%')
    print(f'F1-score:  {f1 * 100:.2f}%')

    # 8. Calcular e exibir a matriz de confusão
    conf_matrix = confusion_matrix(y_true_indices, y_pred_indices)
    class_names = label_encoder.classes_

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Previsão do Modelo', fontsize=12)
    plt.ylabel('Rótulo Real', fontsize=12)
    plt.title('Matriz de Confusão', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 9. Salvar o modelo treinado
    nome_modelo = input("\nDigite o nome do modelo para salvar (ex: meu_modelo_cnn): ")
    if nome_modelo:
        os.makedirs('modelos_exportados', exist_ok=True)
        model.save(f'modelos_exportados/{nome_modelo}.keras')
        print(f"\n✅ Modelo '{nome_modelo}.keras' salvo com sucesso em 'modelos_exportados/'!")