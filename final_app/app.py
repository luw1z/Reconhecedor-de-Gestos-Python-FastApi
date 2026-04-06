import cv2
import numpy as np
import base64
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from reconhecedor import ReconhecedorGestos

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
reconhecedor = ReconhecedorGestos()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/")
async def get_root():
    # Carrega agora o HTML diretamente do arquivo físico index.html
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload_front = json.loads(data)
            
            base64_img = payload_front.get("image")
            desenhar_esqueleto = payload_front.get("draw_landmarks", True)
            
            np_arr = np.frombuffer(base64.b64decode(base64_img), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                img_processada, gesture_final = reconhecedor.processar_imagem(img, desenhar_esqueleto)
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
                _, buffer = cv2.imencode('.jpg', img_processada, encode_param)
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    "image": img_str,
                    "finalGesture": gesture_final
                }
                
                await websocket.send_text(json.dumps(payload))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Erro no processamento: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
