# preinstall

## For audio gen

```
npm i &&
npm run dev
```

(input file expected in root folder with the name input.mp4)

## For video translation

### one time setup

```
python3 -m venv venv
source venv/bin/activate
```

### Install deps

```
pip install -r requirements.txt
```

### Run command

```
python video_ocr_translate_overlay.py <input mp4 file>  ./output --lang <language>
```

e.g.

```
python video_ocr_translate_overlay.py dcm_gu.mp4  ./output --lang gu
```

<!-- ```
sudo apt install fonts-noto-core fonts-noto-unhinted fonts-noto-extra
```

# 1. Install PyTorch (choose CPU build)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install mmengine (required by all OpenMMLab projects)

pip install mmengine

# 3. Install mmcv (core computer vision lib)

pip install mmcv

# 4. Install MMOCR itself

pip install mmocr -->
