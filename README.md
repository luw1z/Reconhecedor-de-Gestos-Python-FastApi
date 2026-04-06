# 🚀 Reconhecimento de Gestos (Next Level Week - Jornada Python)

Este projeto de **Visão Computacional e Machine Learning** é o produto final desenvolvido a partir do aprendizado adquirido durante o **Next Level Week (NLW) - Jornada Python** gerido pela Rocketseat. 

Nele, evoluímos a base para aplicar na prática a construção "end-to-end" de uma ferramenta focada na classificação de gestos baseados em mãos em tempo real. O código captura os frames de webcam, processa e extrai toda a topologia/coordenadas 3D dos dedos (landmarks) nativamente com a engine do **MediaPipe** da Google. Entregamos esses valores matemáticos a um modelo **Random Forest (Scikit-Learn)** e ele nos responde perfeitamente o gesto executado.

A arquitetura final roda sob um **WebApp servido com FastAPI e Uvicorn**. E, mais notavelmente, o streaming e telemetria funcionam sob **WebSockets** bi-direcionais que quebram o gargalo tradicional da internet garantindo que o vídeo flua pro frontend (nativo em HTML/JS com UI Premium limpa) numa latência incrível.

*(Nota: Nas pastas adjacentes também estruturei meus testes usando ecossistemas de arquitetura como YOLOv8, Clipseg e MobileNet para segmentar o mundo e detectar variados objetos!)*

## 🛠 Tecnologias Utilizadas

- **Backend / Streaming:** FastAPI, Uvicorn, WebSockets
- **Visão Computacional:** OpenCV, MediaPipe
- **Machine Learning:** Scikit-Learn (Random Forest), Pandas, NumPy, JobLib
- **Frontend:** Vanilla HTML/CSS/JS

## 🚀 Funcionalidades

- **Interface em Baixa Latência:** Uso do WebSocket para comunicação contínua entre frontend e backend. A UI conta com telemetria básica (FPS e Latência) em um painel dark mode construído apenas com CSS limpo.
- **Modelagem em Tempo Real:** Captura, inserção e extração de landmarks alimentando o modelo RandomForest frame a frame.
- **Script de Coleta (Dataset Creator):** Ferramenta inclusa para gerar o próprio conjunto de dados (CSV) e treinar gestos novos pelo painel da webcam usando contagem regressiva em formato Burst.

## ⚙ Como rodar o projeto

Você precisará de Python 3.12 (ou recente) e uma webcam integrada.

1. **Clone este repositório:**
```bash
git clone https://github.com/SEU-USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

3. **Inicie o servidor (WebApp):**
```bash
cd final_app
uvicorn app:app --reload
```

4. **Acesse no navegador:** Entrando em `http://127.0.0.1:8000` o sistema vai requisitar sua câmera de forma local e as inferências devem começar magicamente.

## 🧠 Como ensinar gestos novos?

A parte mais legal do repositório é que ele permite mudar as classes e gerar um modelo pkl do zero para você usar:

1. Acesse o script `classificador de gestos/coleta_dataset_gestos.py` e altere a string em `TARGET_LABEL` para o novo gesto.
2. Ao rodar o script, pressione `R`. A câmera começará um timer de 3 segundos e vai "bater fotos seguidas de dados" focadas na pose da sua mão, enchendo o arquivo `.csv`.
3. Abra o Jupyter Notebook `treinamento_gestos.ipynb` e apenas re-execute as células. Ele lerá os seus novos dados e compilará um novo aquivo `.pkl`. Basta copiar esse arquivo e substituir o atual dentro da pasta `final_app`.

---

*Coded by [Luiz Willian / @luw1z*
