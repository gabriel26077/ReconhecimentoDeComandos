import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm # Para uma barra de progresso bonita

def criar_espectrogramas_mel(diretorio_audio, diretorio_saida):
    """
    Percorre uma pasta de arquivos .wav, gera espectrogramas Mel para cada um
    e os salva como imagens .png na pasta de saída.
    
    Args:
        diretorio_audio (str): Caminho para a pasta com os arquivos .wav.
        diretorio_saida (str): Caminho para a pasta onde as imagens .png serão salvas.
    """
    # 1. Certifica-se de que o diretório de saída existe
    os.makedirs(diretorio_saida, exist_ok=True)
    
    # 2. Lista todos os arquivos .wav no diretório de áudio
    arquivos_wav = [f for f in os.listdir(diretorio_audio) if f.endswith('.wav')]
    
    print(f"Encontrados {len(arquivos_wav)} arquivos .wav. Iniciando a geração dos espectrogramas...")
    
    # 3. Itera sobre cada arquivo com uma barra de progresso
    for nome_arquivo in tqdm(arquivos_wav):
        try:
            # Constrói o caminho completo para o arquivo de áudio
            caminho_audio = os.path.join(diretorio_audio, nome_arquivo)
            
            # Carrega o arquivo de áudio
            y, sr = librosa.load(caminho_audio)
            
            # Gera o espectrograma Mel
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            
            # Converte para decibéis (dB) para melhor representação visual/aprendizado
            S_DB = librosa.power_to_db(S, ref=np.max)
            
            # --- Criação da imagem ---
            plt.figure(figsize=(4, 4)) # Define um tamanho quadrado para a imagem
            librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
            
            # Remove eixos, títulos e bordas para uma imagem limpa (essencial para CNN)
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Constrói o caminho de saída, trocando .wav por .png
            nome_base = os.path.splitext(nome_arquivo)[0]
            caminho_saida_img = os.path.join(diretorio_saida, f"{nome_base}.png")
            
            # Salva a imagem
            plt.savefig(caminho_saida_img, bbox_inches='tight', pad_inches=0)
            
            # Fecha a figura para liberar memória (MUITO IMPORTANTE EM LOOPS)
            plt.close()

        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo}: {e}")

    print("\nProcesso concluído com sucesso!")
    print(f"Todos os espectrogramas foram salvos em: '{diretorio_saida}'")


if __name__ == '__main__':
    # Define os diretórios com base na estrutura de pastas descrita
    DIRETORIO_AUDIOS = 'audios'
    DIRETORIO_ESPECTROS = 'espectros_mel'
    
    # Executa a função principal
    criar_espectrogramas_mel(DIRETORIO_AUDIOS, DIRETORIO_ESPECTROS)