const video = document.getElementById('video');
const outputCanvas = document.getElementById('output_canvas');
const ctxOutput = outputCanvas.getContext('2d');
const txtFinal = document.getElementById('final-gesture');
const drawCheckbox = document.getElementById('draw-checkbox');

const txtFps = document.getElementById('fps-counter');
const txtLatency = document.getElementById('latency-counter');

const qualitySlider = document.getElementById('quality-slider');
const qualityValue = document.getElementById('quality-value');

if (qualitySlider) {
    qualitySlider.addEventListener('input', (e) => {
        qualityValue.innerText = `${e.target.value}%`;
    });
}

let ws = new WebSocket("ws://localhost:8000/ws");

const offscreenCanvas = document.createElement('canvas');
const ctxOffscreen = offscreenCanvas.getContext('2d');
offscreenCanvas.width = 640;
offscreenCanvas.height = 480;

let lastFrameTime = Date.now();
let framesInThisSecond = 0;
let requestSentTime = 0;
let previousGesture = "";

ws.onopen = () => {
    if (video.readyState >= 2) {
        sendFrame();
    }
};

ws.onmessage = (event) => {
    // Calculando e exibindo a Latência da API/Websocket
    let latency = Date.now() - requestSentTime;
    if (txtLatency) txtLatency.innerText = `${latency}ms`;

    let payload = JSON.parse(event.data);
    let image = new Image();

    image.onload = function () {
        ctxOutput.drawImage(image, 0, 0, 640, 480);
        
        // Calculando e exibindo o FPS
        framesInThisSecond++;
        let now = Date.now();
        if (now - lastFrameTime >= 1000) {
            if (txtFps) txtFps.innerText = `${framesInThisSecond} FPS`;
            framesInThisSecond = 0;
            lastFrameTime = now;
        }

        requestAnimationFrame(() => sendFrame());
    };
    image.src = "data:image/jpeg;base64," + payload.image;

    let currentGesture = payload.finalGesture;

    // Atualiza o texto apenas se houver mudança
    if (currentGesture !== previousGesture) {
        txtFinal.innerText = currentGesture;
        previousGesture = currentGesture;
    }
};

ws.onclose = () => { console.log("Conexão WebSocket Interrompida."); }

navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
    .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
        };
        video.onplaying = () => {
            if (ws.readyState === WebSocket.OPEN) {
                sendFrame();
            }
        };
    })
    .catch(err => {
        console.error("Erro ao acessar webcam:", err);
    });

function sendFrame() {
    if (ws.readyState === WebSocket.OPEN && video.readyState >= 2) {
        ctxOffscreen.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

        let quality = qualitySlider ? qualitySlider.value / 100 : 0.5;
        let dataURL = offscreenCanvas.toDataURL("image/jpeg", quality);
        let partes = dataURL.split(',');
        if (partes.length < 2) {
            setTimeout(sendFrame, 100);
            return;
        }

        let base64 = partes[1];
        let msg = {
            image: base64,
            draw_landmarks: drawCheckbox.checked
        };
        
        // Marca o tempo de envio para cálculo da latência pura
        requestSentTime = Date.now();
        ws.send(JSON.stringify(msg));
    } else {
        setTimeout(sendFrame, 100);
    }
}
