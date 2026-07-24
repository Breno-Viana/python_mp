# Virtual Mouse — controle de mouse por gestos com MediaPipe

Controla o cursor do mouse, cliques, arraste e scroll usando as duas mãos, via webcam, com [MediaPipe HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) e [PyAutoGUI](https://pyautogui.readthedocs.io/).

## Como funciona

Cada mão tem um papel fixo:

- **Mão esquerda** → controla o **cursor**, o **arraste** (drag) e o **scroll para cima**
- **Mão direita** → controla o **clique esquerdo**, o **clique direito** e o **scroll para baixo**

> **Nota sobre o código**: internamente, o MediaPipe rotula as mãos de forma espelhada em relação ao que a pessoa vê na tela (efeito de câmera espelhada). Por isso, no código, o bloco `if hand_label == 'Left':` na verdade controla a mão **direita** física, e o bloco `if hand_label == 'Right':` controla a mão **esquerda** física. Essa documentação já descreve o comportamento **real**, não o rótulo interno.

## Gestos

### Mão esquerda

| Gesto | Ação |
|---|---|
| Mover o dedo indicador | Move o cursor pela tela |
| Pinçar polegar + dedo médio (segurar) | Arrasta o mouse (equivalente a segurar o botão esquerdo e mover) |
| Fechar anelar + mindinho | Scroll para cima, enquanto o gesto for mantido |
| Fechar a mão inteira (punho) | Contabiliza para o gesto de encerrar o programa |

### Mão direita

| Gesto | Ação |
|---|---|
| Pinçar polegar + indicador | Clique esquerdo (um clique por pinça) |
| Pinçar polegar + dedo médio | Clique direito (um clique por pinça) |
| Fechar anelar + mindinho | Scroll para baixo, enquanto o gesto for mantido |
| Fechar a mão inteira (punho) | Contabiliza para o gesto de encerrar o programa |

### Encerrar o programa

Feche **as duas mãos** ao mesmo tempo (punho fechado nas duas) para encerrar o script.

## Requisitos

- Python 3.10+
- Webcam funcional
- Linux com sessão **X11** (veja a seção [Limitações conhecidas](#limitações-conhecidas) sobre Wayland)

## Instalação

Crie um ambiente virtual (recomendado) e instale as dependências:

```bash
python3 -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python pyautogui
```

O modelo de detecção de mãos (`hand_landmarker.task`) já vem incluído no projeto, na pasta `landmarks/` — não é necessário baixar nada manualmente.

## Executando

```bash
python virtual_mouse.py
```

Posicione-se de frente para a webcam, com boa iluminação, e comece a usar os gestos descritos acima. O programa roda sem exibir uma janela de vídeo por padrão (câmera "invisível" em segundo plano).

## Configurações ajustáveis

No topo do script, algumas constantes controlam o comportamento e podem ser calibradas conforme a sua preferência:

| Constante | O que controla | Padrão |
|---|---|---|
| `SENSITIVITY_MARGIN` | Quanto da borda da imagem da webcam é ignorado ao mapear a posição da mão para a tela. Valores maiores = menos esforço físico para cobrir a tela inteira, porém mais sensível a tremores | `0.35` |
| `SMOOTHING` | Suavização do movimento do cursor (0 a 1). Valores menores = mais suave, porém mais "atraso"; valores maiores = mais responsivo, porém mais tremido | `0.25` |
| `CLICK_THRESHOLD` | Distância máxima (em pixels da webcam) entre dois dedos para considerar que estão "pinçados" | `30` |

## Limitações conhecidas

- **Wayland**: em sessões Wayland (padrão em versões recentes do GNOME/Ubuntu), o `pyautogui` pode não conseguir ler a posição real do cursor (`pg.position()` trava em um valor fixo), embora mover o cursor (`pg.moveTo`) continue funcionando. O script contorna isso rastreando a posição do cursor internamente (`prev_mouse_x`, `prev_mouse_y`) em vez de depender do sistema operacional. Se notar comportamento inconsistente do mouse, considere rodar em uma sessão **X11 nativa** (opção "GNOME on Xorg" na tela de login, se disponível na sua distro).
- **Distância da câmera**: a precisão da detecção piora conforme a mão se afasta da webcam. Recomenda-se uso a uma distância confortável de "uso de teclado" (40–60 cm).
- **Iluminação**: pouca luz reduz a confiabilidade da detecção de landmarks.