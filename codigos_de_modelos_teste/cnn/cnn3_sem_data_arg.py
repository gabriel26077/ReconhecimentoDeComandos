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

# --- Importações para os Callbacks ---
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configurações ---
SPECTROGRAM_DIR = 'dataset/espectros_mel'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# ==============================================================================
# A função carregar_e_processar_dados continua a mesma
# ==============================================================================

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
    imagens = np.array(imagens, dtype="float32") / 255.0
    rotulos_str = np.array(rotulos_str)
    le = LabelEncoder()
    rotulos_int = le.fit_transform(rotulos_str)
    num_classes = len(le.classes_)
    rotulos_cat = keras.utils.to_categorical(rotulos_int, num_classes=num_classes)
    print(f"\nDados carregados. Encontradas {num_classes} classes: {list(le.classes_)}")
    return imagens, rotulos_cat, num_classes, le

# ==============================================================================
# FUNÇÃO MODIFICADA: construir_modelo_cnn SEM DATA AUGMENTATION
# ==============================================================================

def construir_modelo_cnn(input_shape, num_classes):
    """
    Cria e compila a arquitetura da CNN, SEM a etapa de Data Augmentation.
    """
    
    # --- Construir o modelo usando a API Funcional do Keras ---
    
    # Camada de entrada
    inputs = keras.Input(shape=input_shape)
    
    # A primeira camada de convolução agora se conecta DIRETAMENTE à entrada.
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    
    # Camada de saída
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Cria o modelo final
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compila o modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==============================================================================
# O callback customizado e o resto do script continuam os mesmos
# ==============================================================================

class LivePlotCallback(keras.callbacks.Callback):
    """Callback para plotar a perda e a acurácia em tempo real."""
    def on_train_begin(self, logs={}):
        self.epochs = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        print("\nIniciando treinamento... Feche a janela do gráfico para parar antecipadamente.")

    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)
        for key, value in logs.items():
            self.history[key].append(value)
        
        self.axes[0].clear()
        self.axes[1].clear()
        
        self.axes[0].plot(self.epochs, self.history['loss'], 'b-', label='Perda de Treino')
        self.axes[0].plot(self.epochs, self.history['val_loss'], 'r-', label='Perda de Validação')
        best_epoch_loss = np.argmin(self.history['val_loss'])
        self.axes[0].axvline(x=best_epoch_loss, color='g', linestyle='--', label=f'Melhor Época ({best_epoch_loss+1})')
        self.axes[0].set_title('Perda (Loss) do Modelo')
        self.axes[0].set_xlabel('Época')
        self.axes[0].set_ylabel('Perda')
        self.axes[0].legend()
        self.axes[0].grid(True)
        
        self.axes[1].plot(self.epochs, self.history['accuracy'], 'b-', label='Acurácia de Treino')
        self.axes[1].plot(self.epochs, self.history['val_accuracy'], 'r-', label='Acurácia de Validação')
        best_epoch_acc = np.argmax(self.history['val_accuracy'])
        self.axes[1].axvline(x=best_epoch_acc, color='g', linestyle='--', label=f'Melhor Época ({best_epoch_acc+1})')
        self.axes[1].set_title('Acurácia do Modelo')
        self.axes[1].set_xlabel('Época')
        self.axes[1].set_ylabel('Acurácia')
        self.axes[1].legend()
        self.axes[1].grid(True)
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.1)

    def on_close(self, event):
        print("\nJanela do gráfico fechada! Interrompendo o treinamento...")
        self.model.stop_training = True
        
    def on_train_end(self, logs={}):
        plt.ioff()
        plt.show()

# --- Script Principal ---
if __name__ == '__main__':
    # 1. Carregar dados
    X, y, num_classes, label_encoder = carregar_e_processar_dados(SPECTROGRAM_DIR)

    # 2. Divisão Tripla dos Dados (Treino, Validação, Teste)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    val_size_relative = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative, random_state=42, stratify=y_temp
    )
    print(f"\nDivisão dos dados: Treino({len(X_train)}), Validação({len(X_val)}), Teste({len(X_test)})")

    # 3. Construir o modelo
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
    model = construir_modelo_cnn(input_shape, num_classes) # Chamando a versão SEM augmentation
    model.summary()
    
    # 4. Configurar os Callbacks
    MAX_EPOCHS = 200
    PATIENCE = 15
    CHECKPOINT_DIR = 'modelos_checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model_no_aug.keras') # Mudei o nome para diferenciar
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE, 
            verbose=1, 
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        LivePlotCallback()
    ]

    # 5. Treinar o modelo
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # 6. Avaliação final no conjunto de TESTE
    print("\n--- Avaliando o melhor modelo no conjunto de teste ---")
    y_pred_probs = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    y_true_indices = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_indices, y_pred_indices)
    precision = precision_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)
    recall = recall_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)
    f1 = f1_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)

    print(f'\nAcurácia: {accuracy * 100:.2f}%')
    print(f'Precisão:  {precision * 100:.2f}%')
    print(f'Recall:    {recall * 100:.2f}%')
    print(f'F1-score:  {f1 * 100:.2f}%')

    # 7. Exibir a matriz de confusão final
    conf_matrix = confusion_matrix(y_true_indices, y_pred_indices)
    class_names = label_encoder.classes_
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Previsão do Modelo', fontsize=12)
    plt.ylabel('Rótulo Real', fontsize=12)
    plt.title('Matriz de Confusão Final (Conjunto de Teste)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()