import cv2
from reconhecedor import ReconhecedorGestos

def main():
    print("--- INICIALIZANDO O RECONHECEDOR DE GESTOS ---")
    reconhecedor = ReconhecedorGestos()

    cap = cv2.VideoCapture(0)
    print("\n--- INICIANDO AS LENTES DOS OLHOS DA INTELIGÊNCIA ARTIFICIAL ---")
    print("-> Pressione a tecla 'q' (Q) para Fechar o Software.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # ========================================================
        # Processando a imagem (Magia da IA extraída para o arquivo menor)
        # ========================================================
        frame_processado = reconhecedor.processar_imagem(frame)

        cv2.imshow("IA do Futuro Lendo Maos - Base Pessoal", frame_processado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✔️] Desconectado.")

if __name__ == "__main__":
    main()
