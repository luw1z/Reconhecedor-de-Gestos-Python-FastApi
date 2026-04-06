import cv2
import csv
import os
import time
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIGURAÇÕES DO DATASET (DUAS MÃOS)
# ==========================================
TARGET_LABEL = "passaro"
CSV_FILENAME = "dataset_gestos_duas_maos.csv"
FRAMES_POR_SESSAO = 100 # Quantos frames capturar a cada sessão (100 = ~3 segundos de vídeo)

# Baixando e Configurando o Modelo
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Baixando o Cérebro do Hand Landmarker (Nova API)...")
    urllib.request.urlretrieve(model_url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# Configurando o CSV com 127 colunas
try:
    with open(CSV_FILENAME, "x", newline='') as f:
        writer = csv.writer(f)
        headers = ["label"]
        for hand in [1, 2]:
            for i in range(21):
                headers.extend([f"m{hand}_p{i}_x", f"m{hand}_p{i}_y", f"m{hand}_p{i}_z"])
        writer.writerow(headers)
except FileExistsError:
    pass 

# Tabela das linhas imaginárias entre as juntas que formam a teia da mão
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (5,9), (9,10), 
    (10,11), (11,12), (9,13), (13,14), (14,15), (15,16), (13,17), (17,18), 
    (18,19), (19,20), (0,17)
]

# ESTADOS DA GRAVAÇÃO
recording_stage = 0      # 0 = Parado, 1 = Contagem Regressiva, 2 = Gravando
frames_recorded = 0
countdown_start_time = 0

cap = cv2.VideoCapture(0)

print(f"===========================================================")
print(f" 🎬 GRAVADOR DE DATASET COM CONTAGEM REGRESSIVA AUTOMÁTICA")
print(f" 🎯 ALVO ATUAL (LABEL): '{TARGET_LABEL}'")
print(f" -> Pressione 'r' (R) para iniciar a contagem regressiva de 3s.")
print(f" -> O script vai capturar e salvar {FRAMES_POR_SESSAO} frames automáticos.")
print(f" -> Pressione 'q' (Q) para ENCERRAR.")
print(f"===========================================================")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1) # Espelho
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    row_data = None
    num_maos = 0

    if detection_result.hand_landmarks:
        row_data = [TARGET_LABEL]
        points_px = [] 
        num_maos = len(detection_result.hand_landmarks)

        for i in range(2):
            if i < num_maos:
                hand_landmarks = detection_result.hand_landmarks[i]
                for landmark in hand_landmarks:
                    row_data.extend([landmark.x, landmark.y, landmark.z])
                    x_px = int(landmark.x * frame.shape[1])
                    y_px = int(landmark.y * frame.shape[0])
                    points_px.append((x_px, y_px))
                    cv2.circle(frame, (x_px, y_px), 5, (255, 0, 255), -1)
            else:
                for _ in range(21):
                    row_data.extend([0.0, 0.0, 0.0])

        for m in range(num_maos):
            offset = m * 21 
            for conn in HAND_CONNECTIONS:
                pt1 = points_px[offset + conn[0]]
                pt2 = points_px[offset + conn[1]]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # ----------------------------------------------------
    # MAQUINA DE ESTADOS DA GRAVAÇÃO INTELIGENTE
    # ----------------------------------------------------
    if recording_stage == 1:
        # CONTAGEM REGRESSIVA
        elapsed_time = time.time() - countdown_start_time
        remains = 3 - int(elapsed_time)
        
        if remains > 0:
            cv2.putText(frame, f"PREPARE-SE: {remains}", (120, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
        else:
            print(f"[*] Iniciando gravação burst de {FRAMES_POR_SESSAO} quadros para '{TARGET_LABEL}'...")
            recording_stage = 2
            frames_recorded = 0

    elif recording_stage == 2:
        # MODO BURST DE GRAVAÇÃO CONTÍNUA
        cv2.putText(frame, f"GRAVANDO: {frames_recorded}/{FRAMES_POR_SESSAO}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 20) # Borda vermelha chamativa
        
        # Só gravar o fram se houver mãos na tela (evita lixo de frames vazios no Dataset gravando apenas ruído)
        if row_data:
            with open(CSV_FILENAME, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            frames_recorded += 1
            
            if frames_recorded >= FRAMES_POR_SESSAO:
                recording_stage = 0
                print(f"[!] Sucesso! Um lote de {FRAMES_POR_SESSAO} amostras de '{TARGET_LABEL}' foi gravado no CSV!")
    else:
        # ESTADO PARADO (IDLE)
        cv2.putText(frame, "Aperte 'R' p/ Comecar Gravar  |  'Q' p/ Sair", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    # Interface Fixa de Informação
    cv2.putText(frame, f"Label alvo: {TARGET_LABEL} | Maos Visiveis: {num_maos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Criador de Dataset DUAS MAOS Automato", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Comandos Globais
    if key == ord('r') and recording_stage == 0:
        recording_stage = 1
        countdown_start_time = time.time()
        print("[!] Contagem regressiva ativada.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sessão finalizada. Inspecione seu 'dataset_gestos_duas_maos.csv'.")
