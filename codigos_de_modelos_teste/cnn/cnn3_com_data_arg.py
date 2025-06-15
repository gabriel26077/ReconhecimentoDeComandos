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

# ==============================================================================
# AS FUNÇÕES carregar_e_processar_dados E construir_modelo_cnn CONTINUAM AS MESMAS
# VOU REPETI-LAS AQUI PARA O CÓDIGO FICAR COMPLETO
# ==============================================================================

# --- Configurações ---
SPECTROGRAM_DIR = 'dataset/espectros_mel'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

def carregar_e_processar_dados(diretorio):
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

def construir_modelo_cnn(input_shape, num_classes):
    """
    Cria e compila a arquitetura da CNN, incluindo uma etapa de Data Augmentation.
    """
    
    # --- 1. Definir a camada de Aumento de Dados ---
    # Este é um "mini-modelo" que aplica transformações aleatórias
    data_augmentation = keras.Sequential(
        [
            # Nota: As camadas de aumento só ficam ativas durante o TREINAMENTO.
            # Elas são desativadas automaticamente durante a validação e o teste.
            
            # Aplica uma rotação aleatória de até 5% (aprox. 18 graus)
            layers.RandomRotation(factor=0.05),
            
            # Aplica um zoom aleatório de até 10% (para dentro ou para fora)
            layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            
            # Outras opções úteis para espectrogramas poderiam ser:
            # layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Desloca a imagem
        ],
        name="data_augmentation",
    )

    # --- 2. Construir o modelo principal usando a API Funcional do Keras ---
    # A API Funcional é mais flexível e ideal para inserir o bloco de aumento.
    
    # Camada de entrada
    inputs = keras.Input(shape=input_shape)
    
    # Etapa 1: Aplicar o aumento de dados nas imagens de entrada
    x = data_augmentation(inputs)
    
    # Etapa 2: O restante da sua arquitetura CNN original
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    
    # Etapa 3: Camada de saída
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

# --- Callback customizado para plotagem em tempo real ---
class LivePlotCallback(keras.callbacks.Callback):
    """
    Callback para plotar a perda e a acurácia em tempo real durante o treinamento.
    O treinamento pode ser interrompido fechando a janela do gráfico.
    """
    def on_train_begin(self, logs={}):
        self.epochs = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        plt.ion() # Ativa o modo interativo do Matplotlib
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        print("\nIniciando treinamento... Feche a janela do gráfico para parar antecipadamente.")

    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)
        for key, value in logs.items():
            self.history[key].append(value)
        
        # Limpa os eixos para redesenhar
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Plota a Perda (Loss)
        self.axes[0].plot(self.epochs, self.history['loss'], 'b-', label='Perda de Treino')
        self.axes[0].plot(self.epochs, self.history['val_loss'], 'r-', label='Perda de Validação')
        best_epoch_loss = np.argmin(self.history['val_loss'])
        self.axes[0].axvline(x=best_epoch_loss, color='g', linestyle='--', label=f'Melhor Época ({best_epoch_loss+1})')
        self.axes[0].set_title('Perda (Loss) do Modelo')
        self.axes[0].set_xlabel('Época')
        self.axes[0].set_ylabel('Perda')
        self.axes[0].legend()
        self.axes[0].grid(True)
        
        # Plota a Acurácia (Accuracy)
        self.axes[1].plot(self.epochs, self.history['accuracy'], 'b-', label='Acurácia de Treino')
        self.axes[1].plot(self.epochs, self.history['val_accuracy'], 'r-', label='Acurácia de Validação')
        best_epoch_acc = np.argmax(self.history['val_accuracy'])
        self.axes[1].axvline(x=best_epoch_acc, color='g', linestyle='--', label=f'Melhor Época ({best_epoch_acc+1})')
        self.axes[1].set_title('Acurácia do Modelo')
        self.axes[1].set_xlabel('Época')
        self.axes[1].set_ylabel('Acurácia')
        self.axes[1].legend()
        self.axes[1].grid(True)
        
        # Desenha o gráfico e pausa para que a GUI possa processar os eventos
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.1)

    def on_close(self, event):
        print("\nJanela do gráfico fechada! Interrompendo o treinamento...")
        self.model.stop_training = True
        
    def on_train_end(self, logs={}):
        plt.ioff() # Desativa o modo interativo
        plt.show() # Mantém a janela final aberta

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
    model = construir_modelo_cnn(input_shape, num_classes)
    model.summary()
    
    # 4. Configurar os Callbacks
    MAX_EPOCHS = 200 # Um número alto, pois o EarlyStopping irá parar antes
    PATIENCE = 15 # Número de épocas sem melhora para parar o treino
    CHECKPOINT_DIR = 'modelos_checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Salva o MELHOR modelo encontrado durante o treino, monitorando a perda na validação
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.keras')
    
    callbacks = [
        # Callback 1: Parar o treino se a 'val_loss' não melhorar por 'PATIENCE' épocas
        EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE, 
            verbose=1, 
            restore_best_weights=True # Restaura os pesos da melhor época ao final
        ),
        # Callback 2: Salva o melhor modelo em disco
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Callback 3: Nosso plot em tempo real
        LivePlotCallback()
    ]

    # 5. Treinar o modelo com os callbacks
    # O treinamento agora é dinâmico e vai parar quando for ideal
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks # Passamos nossa lista de callbacks aqui!
    )
    
    # Após o treino, 'model' já terá os melhores pesos graças a 'restore_best_weights=True'.
    # Alternativamente, poderíamos carregar o modelo salvo:
    # print(f"\nCarregando melhor modelo salvo de: {checkpoint_path}")
    # model = keras.models.load_model(checkpoint_path)

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