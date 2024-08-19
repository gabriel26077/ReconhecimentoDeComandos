import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Button
from scipy.io import wavfile
from tkinter import Tk, filedialog
import os

def load_audio(file_path):
    sample_rate, samples = wavfile.read(file_path)
    return sample_rate, samples

def plot_waveform(sample_rate, samples):
    if len(samples.shape) == 2:  # Se for estéreo, converte para mono
        samples = samples.mean(axis=1)
    
    # Cria o eixo do tempo em segundos
    time = np.arange(len(samples)) / sample_rate
    
    # Plota o waveform
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, samples, label='Waveform')
    
    return fig, ax, time

def onclick(event, time, ax, markers, segment_start_color='r', segment_end_color='b'):
    if event.inaxes is not None and event.inaxes != button_ax:
        x = event.xdata
        if x is not None:
            # Converte o tempo para milissegundos
            point_ms = int(x * 1000)
            
            if len(markers) % 2 == 0:  # Início de um segmento
                markers.append(point_ms)
                ax.axvline(x=x, color=segment_start_color, linestyle='--')
            else:  # Fim de um segmento
                markers.append(point_ms)
                ax.axvline(x=x, color=segment_end_color, linestyle='--')
            
            plt.draw()

def save_audio_segments(sample_rate, samples, markers):
    # Garante que tenhamos um número par de marcadores
    if len(markers) % 2 != 0:
        markers.append(len(samples) * 1000 // sample_rate)
    
    # Ordena os marcadores e cria os segmentos
    markers = sorted(markers)
    segments = [(markers[i], markers[i+1]) for i in range(0, len(markers)-1, 2)]
    
    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sample_rate / 1000)
        end_sample = int(end * sample_rate / 1000)
        segment = samples[start_sample:end_sample]
        
        # Salva o segmento como um arquivo WAV
        wavfile.write(f'segment_{i+1}.wav', sample_rate, segment)
        print(f'Segment {i+1} saved: {start}ms to {end}ms')

def on_save_click(event):
    if len(markers) >= 2:
        save_audio_segments(sample_rate, samples, markers)
        print(f'Segments saved: {len(markers) // 2}')
    else:
        print('Not enough markers defined.')

def main():
    global sample_rate, samples, markers, button_ax
    
    # Configura o Tkinter para seleção de arquivo
    root = Tk()
    root.withdraw()  # Esconde a janela principal

    file_path = filedialog.askopenfilename(
        filetypes=[("WAV files", "*.wav")],
        title="Select a WAV file"
    )

    if not file_path or not os.path.exists(file_path):
        print("No file selected or file does not exist.")
        return
    
    sample_rate, samples = load_audio(file_path)
    fig, ax, time = plot_waveform(sample_rate, samples)
    
    markers = []
    
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, time, ax, markers))
    
    # Adiciona um botão para salvar os segmentos de áudio
    button_ax = plt.axes([0.8, 0.05, 0.15, 0.075])  # [esquerda, inferior, largura, altura]
    save_button = Button(button_ax, 'Save Audio')
    save_button.on_clicked(on_save_click)
    
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    
    plt.show()

if __name__ == "__main__":
    main()
