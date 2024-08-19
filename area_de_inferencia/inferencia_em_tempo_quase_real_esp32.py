import sounddevice as sd
import numpy as np
import librosa
import joblib
from pynput import keyboard
import serial

#model_path = 'modelos_exportados/knn_modelo_01_todos_os_dados.pkl'
#scaler_path = 'scalers/scaler_knn_modelo_01_todos_os_dados.pkl'
model_path = 'modelos_exportados/svm_modelo_01_todos_os_dados.pkl'
scaler_path = 'scalers/scaler_svm_modelo_01_todos_os_dados.pkl'

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Configuração da comunicação serial
serial_port = '/dev/ttyUSB0'  # Caminho da porta serial
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Variável global para controlar a gravação
is_recording = False

# Função para capturar áudio do microfone
def record_audio(duration=None, fs=44100):
    print("Pressione a tecla 'ctrl' para começar a gravar.")
    while not is_recording:
        pass  # Aguarda até que a gravação seja iniciada
    print("Gravando...")
    
    # Inicializa a gravação de áudio
    audio_data = []
    with sd.InputStream(samplerate=fs, channels=1, callback=lambda indata, frames, time, status: audio_data.append(indata.copy())):
        while is_recording:  # Enquanto a gravação estiver ativada
            sd.sleep(100)  # Espera um pouco para evitar uso excessivo da CPU
    
    audio_data = np.concatenate(audio_data, axis=0)
    print("Gravação terminada.")
    
    # Reproduzir o áudio gravado
    print("Reproduzindo áudio...")
    sd.play(audio_data, samplerate=fs)
    sd.wait()  # Aguarda a reprodução terminar
    
    return np.squeeze(audio_data)

# Função para extrair características do áudio
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Função para iniciar a gravação
def on_press(key):
    global is_recording
    if key == keyboard.Key.ctrl:
        is_recording = True

# Função para parar a gravação
def on_release(key):
    global is_recording
    if key == keyboard.Key.ctrl:
        is_recording = False

# Função para enviar comando para o ESP32
def send_command(command):
    if ser.is_open:
        ser.write(command.encode())
        print(f"Comando enviado: {command}")
    else:
        print("Porta serial não está aberta.")

# Configurações iniciais
fs = 44100  # Taxa de amostragem
classes = {'verde': 'G', 'azul': 'B', 'vermelho': 'R'}

# Inicia o listener do teclado
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Loop contínuo para gravação e predição
while True:
    audio = record_audio(fs=fs)  # Captura áudio enquanto a tecla 'ctrl' estiver pressionada
    features = extract_features(audio, sr=fs)
    features = scaler.transform([features])  # Normaliza as características

    prediction = model.predict(features)  # Realiza a predição

    # Verifica se a predição está entre as classes conhecidas
    if prediction[0] in classes:
        command = classes[prediction[0]]
        send_command(command)
    else:
        print("Nenhuma classe reconhecida")

# Para o listener do teclado quando o programa termina
listener.stop()
ser.close()
