import cv2
import joblib
import warnings
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings("ignore")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),
    (13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

class ReconhecedorGestos:
    def __init__(self, modelo_pkl='../classificador de gestos/meu_modelo_de_gestos.pkl', model_asset_path='hand_landmarker.task', gesture_asset_path='gesture_recognizer.task'):
        try:
            self.modelo_ia = joblib.load(modelo_pkl)
        except Exception as e:
            raise e

        base_options_hl = python.BaseOptions(model_asset_path=model_asset_path)
        options_hl = vision.HandLandmarkerOptions(base_options=base_options_hl, num_hands=2, min_hand_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(options_hl)

        try:
            base_options_gr = python.BaseOptions(model_asset_path=gesture_asset_path)
            options_gr = vision.GestureRecognizerOptions(base_options=base_options_gr, num_hands=2)
            self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options_gr)
        except Exception as e:
            raise e

    def processar_imagem(self, frame, desenhar=True):
        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        rec_result = self.gesture_recognizer.recognize(mp_image)
        builtin_g = "Nenhum"
        builtin_score = 0
        if hasattr(rec_result, 'gestures') and rec_result.gestures:
             builtin_g = rec_result.gestures[0][0].category_name
             builtin_score = int(rec_result.gestures[0][0].score * 100)
             
        mapa_nativos = {
            "Thumb_Up": "Joinha", "Thumb_Down": "Polegar pra Baixo", 
            "Open_Palm": "Mão Aberta", "Closed_Fist": "Punho Fechado",
            "Victory": "Vitória", "ILoveYou": "Sinal de Rock",
            "Pointing_Up": "Dedo pra Cima", "None": "Nenhum"
        }
        builtin_g = mapa_nativos.get(builtin_g, "Nenhum")

        row_data = []
        points_px = []
        num_maos = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0

        if num_maos == 0:
            return frame, "Nenhum Gesto Detectado"

        for i in range(2):
            if i < num_maos:
                hand_landmarks = detection_result.hand_landmarks[i]
                for landmark in hand_landmarks:
                    row_data.extend([landmark.x, landmark.y, landmark.z])
                    # OTIMIZAÇÃO: Apenas injetamos no openCV e fazemos os calculos numéricos 
                    # do px se o switch do HUD FrontEnd autorizar o display (salvando uso de CPU).
                    if desenhar:
                        x_px = int(landmark.x * frame.shape[1])
                        y_px = int(landmark.y * frame.shape[0])
                        points_px.append((x_px, y_px))
                        cv2.circle(frame, (x_px, y_px), 5, (200, 50, 255), -1)
            else:
                for _ in range(21):
                    row_data.extend([0.0, 0.0, 0.0])

        if desenhar:
            for m in range(num_maos):
                offset = m * 21 
                for conn in HAND_CONNECTIONS:
                    pt1 = points_px[offset + conn[0]]
                    pt2 = points_px[offset + conn[1]]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        X_live = [row_data]   
        gesto_custom = self.modelo_ia.predict(X_live)[0]
        taxa_certeza = int(max(self.modelo_ia.predict_proba(X_live)[0]) * 100)

        custom_lower = gesto_custom.lower().strip()
        if "aber" in custom_lower or "palma" in custom_lower:
            gesto_custom = "Mão Aberta"
        elif "fech" in custom_lower or "soco" in custom_lower or "punh" in custom_lower:
            gesto_custom = "Punho Fechado"
        elif "joi" in custom_lower:
            gesto_custom = "Joinha"

        gesto_final = gesto_custom
        confianca_final = taxa_certeza
        
        if taxa_certeza < 70 and builtin_g != "Nenhum" and builtin_score > taxa_certeza:
            gesto_final = builtin_g
            confianca_final = builtin_score

        return frame, f"{gesto_final} ({confianca_final}%)"
