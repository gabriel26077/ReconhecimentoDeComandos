import sounddevice as sd
import wave
from pynput import keyboard
import numpy as np
import threading

# Variável global para controlar a gravação
is_recording = False
audio_data = []

# Função para gravar áudio enquanto 'ctrl' está pressionado
def gravar_audio(taxa_amostragem=44100, nome_arquivo="audio_gravado"):
    global is_recording, audio_data
    
    def callback(indata, frames, time, status):
        if is_recording:
            audio_data.append(indata.copy())

    print("Pressione e segure 'ctrl' para começar a gravar...")
    
    # Inicializa a gravação de áudio
    with sd.InputStream(samplerate=taxa_amostragem, channels=1, dtype='int16', callback=callback):
        while True:
            if not is_recording and len(audio_data) > 0:
                print("Gravação finalizada!")
                break
            sd.sleep(100)  # Espera um pouco para evitar uso excessivo da CPU
    
    # Converte a lista de arrays em um único array
    audio_data = np.concatenate(audio_data, axis=0)

    # Salva o áudio em um arquivo WAV
    with wave.open(f'{nome_arquivo}.wav', 'w') as arquivo_wav:
        arquivo_wav.setnchannels(1)
        arquivo_wav.setsampwidth(2)  # 16 bits = 2 bytes
        arquivo_wav.setframerate(taxa_amostragem)
        arquivo_wav.writeframes(audio_data.tobytes())

    print(f"Áudio salvo como {nome_arquivo}.wav")

# Função para iniciar a gravação
def on_press(key):
    global is_recording
    if key == keyboard.Key.ctrl:
        is_recording = True
        print("Gravando...")

# Função para parar a gravação
def on_release(key):
    global is_recording
    if key == keyboard.Key.ctrl:
        is_recording = False
        print("Parando a gravação...")

# Configurações do listener de teclado
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Aguarda o usuário fornecer o nome do arquivo e então inicia a gravação
if __name__ == "__main__":
    nome_arquivo = input("Nome do arquivo: ")
    gravar_audio(nome_arquivo=nome_arquivo)

    # Para o listener do teclado quando o programa termina
    listener.stop()
