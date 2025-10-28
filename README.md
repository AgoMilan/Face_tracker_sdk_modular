# Canon Face Tracker (YOLO + InsightFace/dlib)

Tento projekt využívá Canon EDSDK pro získávání LiveView z kamery Canon a kombinuje jej s YOLO11 pro detekci obličejů a InsightFace (nebo dlib) pro rozpoznávání osob.

## 📁 Struktura projektu
```
face_tracker/
├── main.py
├── camera/
│   └── canon_sdk.py
├── recognition/
│   └── face_manager.py
├── models/
│   └── detector_yolo.py
├── utils/
│   ├── config.py
│   └── draw.py
└── requirements.txt
```

## ⚙️ Instalace
1. Vytvoř si virtuální prostředí (doporučeno)
2. Nainstaluj závislosti:
   ```bash
   pip install -r requirements.txt
   ```

3. Uprav cestu k Canon EDSDK DLL v souboru `utils/config.py`:
   ```python
   EDSDK_PATH = r"C:\Cesta\k\EDSDK.dll"
   ```

4. Ujisti se, že model `YOLO11face.pt` je ve stejném adresáři, nebo uprav cestu v `utils/config.py`.

## ▶️ Spuštění
```bash
python main.py
```

## 🧠 Ovládání během běhu
- **r** – přejmenuj aktuální osobu („unknown“ → jméno)
- **a** – přidej embedding pro aktuální osobu
- **c** – pročisti galerii (ponechá 10 nejlepších embeddingů)
- **ESC** – ukončení programu

## 📦 Galerie a snímky
- Galerie se ukládá do `gallery.json`
- Miniatury obličejů se ukládají do složky `thumbnails/`

---

Autor: Milan  
Verze: 3.15 (modulární)
