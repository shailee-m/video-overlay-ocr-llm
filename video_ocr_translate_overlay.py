#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_ocr_translate_overlay.py

Pipeline:
 1) Extract frames from input video (preserve FPS).
 2) OCR each frame (cached).
 3) Track text lines across frames (IOU-based).
 4) Classify each line as STATIC (appears fully formed) or HANDWRITTEN (grows).
 5) Compute timing metadata:
      STATIC: start_visible, end_frame (=start_visible), last_frame
      HANDWRITTEN (phrase): start_writing, end_writing, last_frame
      HANDWRITTEN (per word): start_frame (first char of word), end_frame (=phrase end_writing), last_frame
 6) Translate unique final line texts (cache).
 7) Overlay:
      STATIC: full translation across [start_visible, last_frame] in the median line bbox.
      HANDWRITTEN: progressive per-word reveal until end_writing, then solid until last_frame.
 8) Write:
      segments_static.json
      segments_handwritten.json
 9) Re-encode frames into video with original audio.

Notes:
 - OCR: uses PaddleOCR if available; optional MMOCR fallback (best-effort).
 - Translation: by default returns identity (no network). Provide your integration in `translate_texts`.
 - Boxes: (x1, y1, x2, y2) in pixel coords on original frame size.

Author: you :)
"""

import os, sys, math, json, shutil, hashlib
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

# Optional OCR engines
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from mmocr.apis import MMOCRInferencer as MMOCRInferencer
except Exception:
    MMOCRInferencer = None

# Video mux
from moviepy import ImageSequenceClip, AudioFileClip


# Optional: Google Gemini translate. If not configured, falls back to identity translation.
try:
    from google import genai
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False

# =========================
# Config
# =========================
OUTPUT_IMAGE_EXT = ".png"
MAX_WORKERS = max(1, os.cpu_count() or 1)

# Overlay appearance
BG_RGBA = (0, 0, 0, 150)  # translucent black background for overlay text
PAD_X = 6                 # horizontal padding inside bbox for text
PAD_Y = 3                 # vertical padding inside bbox for text
TEXT_MAX_W_FRAC = 0.98    # leave a tiny margin inside the box

# Fonts — adjust paths for your machine
DEFAULT_FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",  # Gujarati example
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
GEMINI_API_KEY = ""  # Replace with your actual Gemini API Key


# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_frames_sorted(frames_dir: str) -> List[str]:
    # Sort by numeric name if available; else lexicographic
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(OUTPUT_IMAGE_EXT)]
    def keyfn(x):
        n = os.path.splitext(x)[0]
        return int(n) if n.isdigit() else x
    files.sort(key=keyfn)
    return [os.path.join(frames_dir, f) for f in files]

def pick_font_path() -> Optional[str]:
    for p in DEFAULT_FONT_PATHS:
        if os.path.exists(p):
            return p
    return None

def draw_text_fit(draw: ImageDraw.ImageDraw, text: str, bbox: Tuple[int,int,int,int], font_path: Optional[str]):
    """Draw text centered within bbox with dynamic font size."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    # bg
    draw.rectangle([x1, y1, x2, y2], fill=BG_RGBA)

    if not text.strip():
        return

    # Fit font height to box
    # Basic heuristic: target ~70% of height for glyphs + padding
    target_h = max(1, int((h - 2 * PAD_Y) * 0.75))
    if target_h <= 0:
        return
    size = max(10, target_h)
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, size=size)
    else:
        font = ImageFont.load_default()

    # shrink font if too wide
    # PIL textbbox available in modern Pillow; fallback to textlength/size
    def text_size(fnt):
        try:
            tb = draw.textbbox((0, 0), text, font=fnt)
            return tb[2]-tb[0], tb[3]-tb[1]
        except Exception:
            return draw.textlength(text, font=fnt), fnt.size

    tw, th = text_size(font)
    while (tw > TEXT_MAX_W_FRAC * (w - 2 * PAD_X)) and size > 10:
        size -= 1
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=size)
        else:
            font = ImageFont.load_default()
        tw, th = text_size(font)

    tx = x1 + (w - tw) // 2
    ty = y1 + (h - th) // 2
    draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)

# =========================
# Video → Frames
# =========================
def extract_frames(video_path: str, frames_dir: str) -> Tuple[List[str], float, Tuple[int,int]]:
    """
    Extracts all frames. Returns (frame_paths, fps, (width, height)).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if os.path.exists(frames_dir):
        frames = []
        for filename in os.listdir(frames_dir):
            frames.append(os.path.join(frames_dir, filename))
        frames.sort(key=lambda x: x)
        return frames, float(fps), (w, h)

    ensure_dir(frames_dir)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    
    frame_paths = []
    idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc="Extracting frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outp = os.path.join(frames_dir, f"{idx:06d}{OUTPUT_IMAGE_EXT}")
        cv2.imwrite(outp, frame)
        frame_paths.append(outp)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return frame_paths, float(fps), (w, h)

# =========================
# OCR
# =========================
class OCREngine:
    def __init__(self, engine: str = "paddle", lang: str = "en"):
        self.engine = None
        self.kind = None
        if engine == "paddle" and PaddleOCR is not None:
            # Use English for latin handwriting; adjust as needed
            self.engine = PaddleOCR( use_angle_cls=True, lang='en')
            self.kind = "paddle"
        elif MMOCRInferencer is not None:
            # Best-effort fallback
            self.engine = MMOCRInferencer(det="DBNet", rec="SVTR")
            self.kind = "mmocr"
        else:
            raise RuntimeError("No OCR engine available. Install paddleocr or mmocr.")

    def ocr(self, image_path: str):
        if self.kind == "paddle":
            return self.engine.ocr(image_path)
        else:
            # MMOCR returns dict; convert to paddle-like for downstream
            out = self.engine(image_path, return_vis=False)
            results = []
            det = out.get("det_polygons", [[]])
            rec = out.get("rec_texts", [[]])
            # naive merge
            page = []
            for poly, txt in zip(det[0], rec[0]):
                # poly: Nx2 -> make 4-point box then convert to x1y1x2y2 later
                page.append([poly, (txt, 0.9)])
            results.append(page)
            return results

def parse_paddle_result(result):
    """
    Returns (texts, boxes, scores)
    boxes as (x1, y1, x2, y2) ints.
    """
    texts = []
    boxes = []
    scores = []
    result_json = result[0].json['res']
    res = result_json
    texts = res.get("rec_texts", [])
    boxes = res.get("rec_boxes", [])
    scores = res.get("rec_scores", [])
    for text, box, score in zip(texts, boxes, scores):
        # box shape: (4, 2) [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        if score >= 0.75:
            x1, y1, x2, y2 = map(int, box)
            texts.append(text)
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(score))
    return texts, boxes, scores


# =========================
# OCR Helpers
# =========================
def extract_texts_and_boxes(result) -> List[Tuple[str, Tuple[int,int,int,int], float]]:
    """
    Normalizes PaddleOCR result into:
      [(text, (x1,y1,x2,y2), score), ...]
    Handles both list-output and dict-output styles.
    """
    output = []
    result_json = result[0].json['res']
    res = result_json
    texts = res.get("rec_texts", [])
    boxes = res.get("rec_boxes", [])
    scores = res.get("rec_scores", [])
    for text, box, score in zip(texts, boxes, scores):
        # box shape: (4, 2) [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        if score >= 0.75:
            x1, y1, x2, y2 = map(int, box)
            output.append((text, [int(x1), int(y1), int(x2), int(y2)], float(score)))
    return output
    # out = []
    # # Dict style.0.
    #     scores = res.get("rec_scores", [1.0] * len(texts))
    #     for t, b, sc in zip(texts, boxes, scores):
    #         # b might be [x1,y1,x2,y2], or 4x2
    #         if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int,float)) for v in b):
    #             x1, y1, x2, y2 = map(int, b)
    #         elif isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(pt, (list,tuple)) and len(pt)==2 for pt in b):
    #             xs = [pt[0] for pt in b]; ys = [pt[1] for pt in b]
    #             x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    #         else:
    #             continue
    #         out.append((t, (x1,y1,x2,y2), float(sc)))
    #     return out

    # # List style (most common)
    # if isinstance(result, list) and len(result) > 0:
    #     for line in result[0]:
    #         bbox = line[0]
    #         text, score = line[1]
    #         if isinstance(bbox, (list,tuple)) and len(bbox)==4 and all(isinstance(v,(int,float)) for v in bbox):
    #             x1,y1,x2,y2 = map(int, bbox)
    #         elif isinstance(bbox, (list,tuple)) and len(bbox)==4 and all(isinstance(pt,(list,tuple)) and len(pt)==2 for pt in bbox):
    #             xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
    #             x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    #         else:
    #             continue
    #         out.append((text, (x1,y1,x2,y2), float(score)))
    # return out






# ============== UTILITIES ==============

def load_frame_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read frame: {path}")
    return img

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def expand_box(box: Tuple[int, int, int, int], w: int, h: int, pad: int = 8) -> Tuple[int, int, int, int]:
    x, y, bw, bh = box
    x2, y2 = x + bw, y + bh
    x = max(0, x - pad)
    y = max(0, y - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return (x, y, x2 - x, y2 - y)

def roi_change_score(prev_gray: np.ndarray, curr_gray: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> float:
    """Average (1 - SSIM) across text ROIs. 0.0 ~ identical, 1.0 ~ very different."""
    if prev_gray is None or curr_gray is None:
        return 1.0
    if not boxes:
        return 1.0

    h, w = prev_gray.shape[:2]
    diffs = []

    for (x, y, bw, bh) in boxes:
        x, y, bw, bh = expand_box((x, y, bw, bh), w, h, pad=8)
        if bw <= 0 or bh <= 0: 
            continue

        crop_prev = prev_gray[y:y+bh, x:x+bw]
        crop_curr = curr_gray[y:y+bh, x:x+bw]
        if crop_prev.size == 0 or crop_curr.size == 0:
            continue

        # Resize small crops to stabilize SSIM
        if min(bw, bh) < 24:
            crop_prev = cv2.resize(crop_prev, (max(24, bw), max(24, bh)))
            crop_curr = cv2.resize(crop_curr, (max(24, bw), max(24, bh)))

        s = ssim(crop_prev, crop_curr, data_range=255)
        diffs.append(1.0 - float(s))

    if not diffs:
        return 1.0
    return float(np.mean(diffs))


# ============== CORE: PROCESS A FRAME WITH CHANGE DETECTION ==============

def process_frame_change_detect(
    idx: int,
    fp: str,
    ocr,
    last_result: Optional[Dict],
    last_gray: Optional[np.ndarray],
    ssim_threshold: float,
    force_refresh_every: int
) -> Tuple[Dict, np.ndarray, Dict]:
    """
    Returns:
      result: Dict saved per-frame
      curr_gray: grayscale of current frame (for next comparison)
      result_for_next: the OCR result that becomes "last_result" for next frame
    """
    img = load_frame_bgr(fp)
    gray = to_gray(img)

    must_refresh = (
        last_result is None or
        last_gray is None or
        (force_refresh_every > 0 and idx % force_refresh_every == 0)
    )

    if not must_refresh:
        # Compare current frame ROIs with last OCR'ed frame ROIs
        boxes = last_result.get("boxes", [])
        change = roi_change_score(last_gray, gray, boxes)

        if change < ssim_threshold:
            # Reuse previous OCR result (copy, update index/filename)
            reused = dict(last_result)
            reused["frame_index"] = idx
            reused["frame_filename"] = os.path.basename(fp)
            reused["_reused"] = True
            return reused, gray, last_result  # last_result unchanged

    # Text changed OR forced refresh -> run OCR
    raw = ocr.ocr(fp)
    tabs = extract_texts_and_boxes(raw)
    fresh = {
        "frame_index": idx,
        "frame_filename": os.path.basename(fp),
        "texts": [t[0] for t in tabs],
        "boxes": [t[1] for t in tabs],
        "scores": [t[2] for t in tabs]
    }
    return fresh, gray, fresh


# ============== PUBLIC API ==============

def run_or_load_ocr_ordered(
    frames: List[str],
    ocr,
    cache_path: str,
    temp_dir_path: str,
    ssim_threshold: float = 0.08,
    force_refresh_every: int = 240  # ~10s at 24 fps; set 0 to disable
) -> List[Dict]:
    """
    Process frames sequentially with text-change detection.
    Writes temp JSONs per frame, then consolidates into cache_path on success.
    """
    if os.path.exists(cache_path):
        print(f"Loading OCR cache: {cache_path}")
        with open(cache_path, "r", encoding="utf8") as f:
            return json.load(f)

    print("Running OCR with change-detection (sequential, temp writes)…")
    os.makedirs(temp_dir_path, exist_ok=True)

    results: List[Dict] = []
    last_result: Optional[Dict] = None
    last_gray: Optional[np.ndarray] = None

    for idx, fp in tqdm(list(enumerate(frames)), total=len(frames)):
        temp_path = os.path.join(temp_dir_path, f"{idx}.json")
        if os.path.exists(temp_path):
            # If a temp exists, load it AND also update last_result/last_gray for continuity.
            with open(temp_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            results.append(saved)
            # Reconstruct last_gray if possible; if not, set None so next frame forces refresh.
            try:
                # Only reload current frame as gray if it was a fresh OCR (optional but safer)
                if not saved.get("_reused", False):
                    img = load_frame_bgr(fp)
                    last_gray = to_gray(img)
                    last_result = saved
            except Exception:
                last_gray = None
            continue

        try:
            result, curr_gray, last_for_next = process_frame_change_detect(
                idx=idx,
                fp=fp,
                ocr=ocr,
                last_result=last_result,
                last_gray=last_gray,
                ssim_threshold=ssim_threshold,
                force_refresh_every=force_refresh_every
            )
        except Exception:
            print(f"Error processing frame {idx} ({fp}): {sys.exc_info()[1]}")
            result = {
                "frame_index": idx,
                "frame_filename": os.path.basename(fp),
                "texts": [],
                "boxes": [],
                "scores": []
            }
            curr_gray = None
            last_for_next = last_result  # keep old state on failure

        # Save per-frame JSON immediately
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        results.append(result)
        last_result = last_for_next
        last_gray = curr_gray

    # Final ordered JSON
    results.sort(key=lambda x: x["frame_index"])
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved final OCR results to {cache_path}")

    # Optional cleanup
    shutil.rmtree(temp_dir_path, ignore_errors=True)
    return results



# =========================
# Tracking and timelines
# =========================
def box_iou(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> float:
    if not b1 or not b2:
        return 0.0
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    a1 = max(0, x2-x1) * max(0, y2-y1)
    a2 = max(0, X2-X1) * max(0, Y2-Y1)
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0

def median_box(boxes: List[Tuple[int,int,int,int]]):
    if not boxes:
        return None
    xs1 = sorted(b[0] for b in boxes)
    ys1 = sorted(b[1] for b in boxes)
    xs2 = sorted(b[2] for b in boxes)
    ys2 = sorted(b[3] for b in boxes)
    m = len(boxes) // 2
    return (xs1[m], ys1[m], xs2[m], ys2[m])

def lcp_len(a: str, b: str) -> int:
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i

def track_lines_across_frames(ocr_results: List[Dict], s: int, e: int, iou_thresh: float = 0.5):
    """
    Greedy box-IOU tracker across frames [s..e].
    Each track: {id, frames=[(frame_idx, text, box)]}
    """
    tracks = []
    next_id = 0
    last_box: Dict[int, Tuple[int,int,int,int]] = {}
    last_seen: Dict[int, int] = {}

    for i in range(s, e + 1):
        texts = ocr_results[i]["texts"]
        boxes = ocr_results[i]["boxes"]
        used = set()
        for t, b in zip(texts, boxes):
            best_id, best_iou = None, 0.0
            for tr in tracks:
                tid = tr["id"]
                if last_seen.get(tid, -999) >= i - 2:
                    iou = box_iou(last_box.get(tid), b)
                    if iou > best_iou:
                        best_id, best_iou = tid, iou
            if best_id is not None and best_iou >= iou_thresh and best_id not in used:
                tr = next(T for T in tracks if T["id"] == best_id)
                tr["frames"].append((i, t, b))
                last_box[best_id] = b
                last_seen[best_id] = i
                used.add(best_id)
            else:
                tid = next_id
                next_id += 1
                tr = {"id": tid, "frames": [(i, t, b)]}
                tracks.append(tr)
                last_box[tid] = b
                last_seen[tid] = i

    # prune all-empty
    pruned = []
    for tr in tracks:
        if any(txt.strip() for (_, txt, _) in tr["frames"]):
            pruned.append(tr)
    return pruned

def summarize_track_timings(tr, min_stable_frames: int = 2):
    """
    For a line-track compute:
      final_text: last non-empty
      start_visible: first frame with any non-empty text
      last_frame: last frame this line is visible (track end)
      start_writing: first frame where any char of final_text appears
      end_writing: first frame where full final_text appears
      final_box: median box where txt == final_text (fallback last box)
      is_static: appears fully-written at start and persists >= min_stable_frames
    """
    frames = tr["frames"]
    non_empty = [(f, t, b) for (f, t, b) in frames if t.strip()]
    if not non_empty:
        return None

    final_text = non_empty[-1][1]
    start_visible = non_empty[0][0]
    last_frame = frames[-1][0]

    # Start/End writing
    start_writing = start_visible
    end_writing = non_empty[-1][0]
    wrote_any = False
    for (f, txt, _) in frames:
        l = lcp_len(txt, final_text)
        if not wrote_any and l > 0:
            start_writing = f
            wrote_any = True
        if txt == final_text:
            end_writing = f
            break

    # final box = median of boxes when final_text observed
    fboxes = [b for (f, txt, b) in frames if txt == final_text and b]
    final_box = median_box(fboxes) if fboxes else non_empty[-1][2]

    # static?
    stable_len = sum(1 for (f, txt, _) in frames if f >= start_visible and txt == final_text)
    is_static = (non_empty[0][1] == final_text) and (stable_len >= min_stable_frames)

    return {
        "final_text": final_text,
        "start_visible": start_visible,
        "last_frame": last_frame,
        "start_writing": start_writing,
        "end_writing": end_writing,
        "final_box": final_box,
        "is_static": is_static,
        "frames": frames,
    }

def split_line_box_to_word_boxes(final_text: str, line_box: Tuple[int,int,int,int]) -> List[Tuple[int,int,int,int]]:
    """
    Split a line bbox into per-word boxes proportional to word lengths.
    """
    words = final_text.split()
    if not words or not line_box:
        return []
    x1, y1, x2, y2 = line_box
    W = max(1, x2 - x1)
    H = max(1, y2 - y1)

    lens = [max(1, len(w)) for w in words]
    total = sum(lens)
    x = x1
    boxes = []
    for i, L in enumerate(lens):
        w = int(round(W * (L / total)))
        # Ensure last box reaches x2
        if i == len(lens) - 1:
            bx2 = x2
        else:
            bx2 = min(x2, x + w)
        boxes.append((x, y1, bx2, y2))
        x = bx2
    return boxes

def word_timelines_in_track(track_summary: Dict) -> List[Dict]:
    """
    Per-word timelines for a (likely handwritten) line. We set:
      start_frame = first time any char of the word appears
      end_frame   = phrase end_writing
      last_frame  = line last_frame
    """
    final_text = track_summary["final_text"]
    start_f = track_summary["start_visible"]
    end_w = track_summary["end_writing"]
    last_f = track_summary["last_frame"]
    if not final_text.strip():
        return []

    words = final_text.split()
    # Sequence of (frame, text) from start_visible .. end_writing
    seq = [(f, t) for (f, t, _b) in track_summary["frames"] if start_f <= f <= end_w]
    seq.sort(key=lambda x: x[0])

    # Map char index -> word index
    full = final_text
    # Compute cumulative character ranges for each word including spaces between words
    indices = []
    pos = 0
    for i, w in enumerate(words):
        start = full.find(w, pos)
        if start < 0:
            # fallback approximate
            start = pos
        end = start + len(w)
        indices.append((i, start, end))
        pos = end

    first_char_frame = [None] * len(words)
    prev_txt = ""
    for (f, txt) in seq:
        # Which new portion of final text became visible since prev?
        l_prev = lcp_len(prev_txt, full)
        l_now = lcp_len(txt, full)
        if l_now > l_prev:
            for ci in range(l_prev, l_now):
                for wi, ws, we in indices:
                    if ws <= ci < we:
                        if first_char_frame[wi] is None:
                            first_char_frame[wi] = f
                        break
        prev_txt = txt

    for i in range(len(words)):
        if first_char_frame[i] is None:
            first_char_frame[i] = start_f

    return [{
        "word": words[i],
        "start_frame": first_char_frame[i],
        "end_frame": end_w,      # IMPORTANT: as requested, per-word end = phrase end_writing
        "last_frame": last_f
    } for i in range(len(words))]

# =========================
# Translation (stub + cache)
# =========================
def load_translation_cache(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_translation_cache(cache: Dict[str, str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def translate_texts(texts: List[str], target_lang: str) -> List[str]:
    """
    Gemini translate. Requires env var GEMINI_API_KEY (or set up however you like).
    If not available, returns identity translations.
    """
    if not texts:
        return []
    if not _GENAI_AVAILABLE:
        return texts[:]  # fallback: identity
    key = GEMINI_API_KEY
    if not key:
        return texts[:]  # fallback: identity
    client = genai.Client(api_key=key)
    outs = []
    for t in texts:
        prompt = f"Translate the following text to {target_lang}. Do not translate digits, if no translation found return empty string, find appropritate translations for short form whenever possible. There may be spelling mistakes in words, over even OCR may detect wrong text, you can assdume the correct work and then transalte in such chases. Reply ONLY with the translation:\n\n{t}"
        try:
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            if not resp.text:
                outs.append("")
            else:
                outs.append(resp.text.strip())
        except Exception as e:
            print("Translate error:", e)
            outs.append(t)
    return outs


# =========================
# Overlay
# =========================
def overlay_words_on_frame(img_path: str, overlays: List[Tuple[str, Tuple[int,int,int,int]]], font_path: Optional[str]) -> Image.Image:
    """
    overlays: List of (text, box)
    """
    base = Image.open(img_path).convert("RGBA")
    lay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(lay)

    for text, box in overlays:
        if not box:
            continue
        draw_text_fit(draw, text or "", box, font_path)

    out = Image.alpha_composite(base, lay).convert("RGB")
    return out

# =========================
# Main
# =========================
def main(input_video: str, out_dir: str, target_lang: str = "gu", ocr_engine: str = "paddle"):
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    output_folder = os.path.join(out_dir, f"{video_name}-{target_lang}")
    ensure_dir(output_folder)

    frames_dir = os.path.join(output_folder, "frames")
    overlay_dir = os.path.join(output_folder, "overlay_frames")
    ocr_cache_dir = os.path.join(output_folder, "ocr_cache")
    ensure_dir(frames_dir)
    ensure_dir(overlay_dir)
    ensure_dir(ocr_cache_dir)

    # 1) Frames
    frames, fps, (W, H) = extract_frames(input_video, frames_dir)
    n_frames = len(frames)
    print(f"[INFO] Extracted {n_frames} frames @ {fps:.3f} fps; size={W}x{H}")

    # 2) OCR (cached)
    ocr = OCREngine(engine=ocr_engine)
    print("[INFO] Running OCR (cached)...")
    ocr_results = run_or_load_ocr_ordered(
        frames=frames,
        ocr=ocr,
        cache_path=os.path.join(output_folder, "ocr_results_ordered.json"),
        temp_dir_path=ocr_cache_dir,
        ssim_threshold=0.08,  # Adjust as needed,
        force_refresh_every=200  # ~10s at 24 fps; set 0 to disable
    )

    # 3) Track lines across full video extent
    print("[INFO] Tracking lines across frames...")
    tracks = track_lines_across_frames(ocr_results, 0, n_frames - 1, iou_thresh=0.5)

    # 4) Summaries & bucketize static vs handwritten
    summaries = []
    for tr in tracks:
        sm = summarize_track_timings(tr, min_stable_frames=2)
        if sm:
            summaries.append(sm)

    static_lines = [sm for sm in summaries if sm["is_static"]]
    hw_lines = [sm for sm in summaries if not sm["is_static"]]

    # 5) Translations with cache
    cache_path = os.path.join(output_folder, "translations_cache.json")
    cache = load_translation_cache(cache_path)

    uniq_texts = []
    for sm in summaries:
        t = sm["final_text"]
        if t not in cache and t not in uniq_texts:
            uniq_texts.append(t)

    if uniq_texts:
        trans_list = translate_texts(uniq_texts, target_lang)
        for t, g in zip(uniq_texts, trans_list):
            cache[t] = g
        save_translation_cache(cache, cache_path)

    # 6) Prepare metadata
    static_meta = []
    hw_meta = []
    font_path = pick_font_path()

    # Handwritten prepared packs for overlay loop
    hw_prepared = []
    for sm in hw_lines:
        words_tl = word_timelines_in_track(sm)
        # Per-word boxes by splitting final box
        word_boxes = split_line_box_to_word_boxes(sm["final_text"], sm["final_box"])
        en_words = [w["word"] for w in words_tl]
        gu_line = cache.get(sm["final_text"], sm["final_text"])
        gu_words = gu_line.split()

        # align pairs by index
        pairs = []
        for i, ew in enumerate(en_words):
            gw = gu_words[i] if i < len(gu_words) else ew
            pairs.append((ew, gw))

        # map word -> box by index (order-based)
        wbox_map = {}
        for i, ew in enumerate(en_words):
            bx = word_boxes[i] if i < len(word_boxes) else sm["final_box"]
            wbox_map[ew] = bx

        hw_prepared.append({
            "summary": sm,
            "words_tl": words_tl,
            "wbox_map": wbox_map,
            "pairs": pairs
        })

        # phrase-level metadata
        hw_meta.append({
            "text": sm["final_text"],
            "translation": gu_line,
            "start_frame": sm["start_writing"],
            "end_frame": sm["end_writing"],
            "last_frame": sm["last_frame"],
            "box": sm["final_box"],
            "words": [{
                "word": w["word"],
                "start_frame": w["start_frame"],
                "end_frame": w["end_frame"],   # equals phrase end_writing (as requested)
                "last_frame": w["last_frame"],
                "box": wbox_map.get(w["word"])
            } for w in words_tl]
        })

    for sm in static_lines:
        static_meta.append({
            "text": sm["final_text"],
            "translation": cache.get(sm["final_text"], sm["final_text"]),
            "start_frame": sm["start_visible"],
            "end_frame": sm["start_visible"],  # fully-formed at first visible
            "last_frame": sm["last_frame"],
            "box": sm["final_box"]
        })

    with open(os.path.join(output_folder, "segments_static.json"), "w", encoding="utf-8") as f:
        json.dump(static_meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_folder, "segments_handwritten.json"), "w", encoding="utf-8") as f:
        json.dump(hw_meta, f, ensure_ascii=False, indent=2)

    # 7) Overlay per-frame
    print("[INFO] Rendering overlays...")
    overlay_paths = []
    for idx, fp in enumerate(tqdm(frames, desc="Overlay")):
        overlays = []

        # Static: full translation across [start_visible .. last_frame]
        for sm in static_lines:
            if sm["start_visible"] <= idx <= sm["last_frame"]:
                gu_text = cache.get(sm["final_text"], sm["final_text"])
                overlays.append((gu_text, sm["final_box"]))

        # Handwritten: per-word progressive until end_writing, then solid
        for pack in hw_prepared:
            sm = pack["summary"]
            if not (sm["start_visible"] <= idx <= sm["last_frame"]):
                continue

            for (en_w, gu_w) in pack["pairs"]:
                # find this word's timeline
                wtl = next((w for w in pack["words_tl"] if w["word"] == en_w), None)
                if not wtl:
                    continue
                if idx < wtl["start_frame"]:
                    frag = ""
                elif idx >= wtl["end_frame"]:
                    frag = gu_w
                else:
                    denom = max(1, wtl["end_frame"] - wtl["start_frame"] + 1)
                    progress = (idx - wtl["start_frame"] + 1) / denom
                    n_chars = max(1, int(math.ceil(len(gu_w) * progress)))
                    frag = gu_w[:n_chars]
                box = pack["wbox_map"].get(en_w, sm["final_box"])
                overlays.append((frag, box))

        img = overlay_words_on_frame(fp, overlays, font_path)
        outp = os.path.join(overlay_dir, os.path.basename(fp))
        img.save(outp)
        overlay_paths.append(outp)

    # 8) Video mux (keep original audio)
    print("[INFO] Writing final video...")
    clip = ImageSequenceClip(overlay_paths, fps=fps)
    # attach original audio if present
    try:
        audio = AudioFileClip(input_video)
        clip = clip.with_audio(audio)
    except Exception:
        pass
    out_video = os.path.join(output_folder, "overlayed.mp4")
    clip.write_videofile(out_video, codec="libx264", audio_codec="aac")
    print("[OK]", out_video)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Blackboard OCR → Indic overlay (static vs handwritten, per-word timelines)")
    ap.add_argument("input_video", help="Path to input .mp4")
    ap.add_argument("output_folder", help="Folder to write outputs")
    ap.add_argument("--lang", default="gu", help="Target language code (default: gu)")
    ap.add_argument("--ocr", default="paddle", choices=["paddle", "mmocr"], help="OCR engine")
    args = ap.parse_args()
    main(args.input_video, args.output_folder, target_lang=args.lang, ocr_engine=args.ocr)
