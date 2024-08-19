import joblib
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def main():
    #model_path = 'modelos_exportados/knn_modelo_01_todos_os_dados.pkl'
    #scaler_path = 'scalers/scaler_knn_modelo_01_todos_os_dados.pkl'
    model_path = 'modelos_exportados/svm_modelo_01_todos_os_dados.pkl'
    scaler_path = 'scalers/scaler_svm_modelo_01_todos_os_dados.pkl'
    
    # Carregar o modelo e o scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Configuração da janela de diálogo para selecionar arquivo
    root = tk.Tk()
    root.withdraw()  # Ocultar a janela principal
    file_path = filedialog.askopenfilename(
        title="Selecione um arquivo de áudio",
        filetypes=[("Arquivos de Áudio", "*.wav")]
    )
    
    if not file_path:
        print("Nenhum arquivo selecionado.")
        return
    
    # Extrair características do áudio selecionado
    features = extract_features(file_path)
    features = np.array([features])
    
    # Normalizar o dado
    features = scaler.transform(features)
    
    # Fazer a inferência
    prediction = model.predict(features)
    
    # Printar a classe na tela
    print(f'A classe do áudio selecionado é: {prediction[0]}')

if __name__ == "__main__":
    main()
