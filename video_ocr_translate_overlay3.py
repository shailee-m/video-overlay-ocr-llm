#!/usr/bin/env python3
import os
import cv2
import json
import math
import hashlib
import shutil
import requests
from tqdm import tqdm
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip, AudioFileClip

# =========================
# Config
# =========================
OUTPUT_FPS = 20
MIN_STABLE_FRAMES = 4                # how many identical frames to treat a phrase/word as "stable"
SEMITRANSPARENT_RECT = (0,100,0,255) # RGBA for overlay bg
INDIC_FONT = "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf"  # change per target language
FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GEMINI_API_KEY = ""  # Replace with your actual Gemini API Key
# ImageSequenceClip, AudioFileClip = editor.ImageSequenceClip, editor.AudioFileClip

# =========================
# Utils
# =========================
def ensure_font(size=24, prefer=INDIC_FONT):
    try:
        return ImageFont.truetype(prefer, size=size)
    except Exception:
        try:
            return ImageFont.truetype(FALLBACK_FONT, size=size)
        except Exception:
            return ImageFont.load_default()

def text_width(font, s: str) -> int:
    if hasattr(font, "getlength"):
        return max(1, int(font.getlength(s)))
    # fallback
    bbox = font.getbbox(s)
    return max(1, int(bbox[2] - bbox[0]))

def extract_frames(video_path, frames_dir, fps=OUTPUT_FPS):
    if os.path.exists(frames_dir):
        frames = []
        for filename in os.listdir(frames_dir):
            frames.append(os.path.join(frames_dir, filename))
        frames.sort(key=lambda x: x)
        return frames

    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = fps
    frame_interval = max(1, int(round(input_fps / fps)))
    frames = []
    i = save_idx = 0
    print("Extracting frames...")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % frame_interval == 0:
            fp = os.path.join(frames_dir, f"frame_{save_idx:05d}.png")
            cv2.imwrite(fp, img)
            frames.append(fp)
            save_idx += 1
        i += 1
    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames

def extract_texts_and_boxes_from_paddle(result):
    """
        Accepts output from PaddleOCR .ocr() and returns:
        [(text, (x1, y1, x2, y2), score), ...]
        Works for both list and dict (advanced) outputs.
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

def run_or_load_ocr(frames, cache_path, ocr: PaddleOCR):
    if os.path.exists(cache_path):
        print(f"Loading cached OCR from {cache_path} ...")
        with open(cache_path, "r", encoding="utf8") as f:
            ocr_results = json.load(f)
        # sanity sort
        ocr_results.sort(key=lambda x: x["frame_index"])
        return ocr_results

    print("Running OCR on all frames ...")
    ocr_results = []
    for idx, frame_path in enumerate(tqdm(frames)):
        raw = ocr.ocr(frame_path)
        lines = extract_texts_and_boxes_from_paddle(raw)
        ocr_results.append({
            "frame_index": idx,
            "frame_filename": os.path.basename(frame_path),
            "texts": [t for (t, b, s) in lines],
            "boxes": [b for (t, b, s) in lines],
            "scores": [s for (t, b, s) in lines]
        })
    # sanity sort
    ocr_results.sort(key=lambda x: x["frame_index"])
    with open(cache_path, "w", encoding="utf8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    return ocr_results


import re

def _norm(s: str) -> str:
    # normalize for comparisons: lowercase, strip punctuation, collapse spaces
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _lev(a: str, b: str) -> int:
    # simple Levenshtein distance
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1,        # deletion
                        dp[j-1] + 1,      # insertion
                        prev + cost)      # substitution
            prev = cur
    return dp[-1]

def collapse_to_single_final_phrase(phrase_ranges, max_edit_for_end=1, min_prefix_len=3):
    """
    phrase_ranges: list of dicts with keys:
      - start_frame, end_frame, text, boxes
    Returns a single-item list:
      [{start_frame, end_frame, text, boxes}]
    where 'text' is the longest (final) phrase and the span covers the whole run.
    """
    nonempty = [r for r in phrase_ranges if r["text"].strip()]
    if not nonempty:
        return []

    # 1) pick canonical = longest normalized text
    by_len = sorted(
        ((r, _norm(r["text"])) for r in nonempty),
        key=lambda x: len(x[1])
    )
    canonical_range, canonical_norm = by_len[-1]
    canonical_text = canonical_range["text"]

    # 2) find earliest frame where a meaningful prefix of canonical appears
    start_frame = canonical_range["start_frame"]
    candidates_start = []
    for r in phrase_ranges:
        n = _norm(r["text"])
        # Accept as "start of run" if n is a prefix of canonical and has at least min_prefix_len chars
        if n and canonical_norm.startswith(n) and len(n) >= min_prefix_len:
            candidates_start.append(r["start_frame"])
    if candidates_start:
        start_frame = min(candidates_start)

    # 3) find latest end where text is close to canonical (or exact)
    end_frame = canonical_range["end_frame"]
    candidates_end = []
    for r in phrase_ranges:
        n = _norm(r["text"])
        if not n:
            continue
        d = _lev(n, canonical_norm)
        if d <= max_edit_for_end or n == canonical_norm:
            candidates_end.append(r["end_frame"])
    if candidates_end:
        end_frame = max(candidates_end)

    # 4) choose boxes from the earliest frame in the final span that has the most boxes
    span_members = [r for r in phrase_ranges if not (r["end_frame"] < start_frame or r["start_frame"] > end_frame)]
    if span_members:
        # pick the member in span with max number of boxes, preferring earliest
        span_members.sort(key=lambda r: (r["start_frame"], -(len(r.get("boxes", [])))))
        boxes = max(span_members, key=lambda r: len(r.get("boxes", []))).get("boxes", [])
    else:
        boxes = canonical_range.get("boxes", [])

    return [{
        "start_frame": start_frame,
        "end_frame": end_frame,
        "text": canonical_text,
        "boxes": boxes
    }]


def get_phrase_ranges_with_boxes(ocr_results, min_stable_frames=MIN_STABLE_FRAMES, include_empty=True):
    """
    Build stable ranges of full-board text (joined string).
    Each range holds: start_frame, end_frame, text, boxes (from the first frame in range).
    If include_empty=True, empty text ranges are also included.
    """
    ranges = []
    if not ocr_results:
        return ranges
    def flat_text(fr_item):
        return " ".join(fr_item["texts"]).strip()

    current_text = flat_text(ocr_results[0])
    current_boxes = ocr_results[0]["boxes"]
    start_frame = 0
    stable_count = 1

    for i in range(1, len(ocr_results)):
        t = flat_text(ocr_results[i])
        if t == current_text:
            stable_count += 1
        else:
            if (stable_count >= min_stable_frames or ( i > len(ocr_results)-min_stable_frames)) and (include_empty or current_text):
                ranges.append({
                    "start_frame": start_frame,
                    "end_frame": i - 1,
                    "text": current_text,
                    "boxes": current_boxes
                })
            # reset
            start_frame = i
            current_text = t
            current_boxes = ocr_results[i]["boxes"]
            stable_count = 1

    # last
    if (stable_count >= min_stable_frames or ( i > len(ocr_results)-min_stable_frames)) and (include_empty or current_text):
        ranges.append({
            "start_frame": start_frame,
            "end_frame": len(ocr_results) - 1,
            "text": current_text,
            "boxes": current_boxes
        })
    return ranges

# ===== line → per-word boxes (by visual width proportions) =====
def split_line_into_word_boxes(line_text, line_box, font_path=None):
    if not line_text.strip():
        return []
    x1, y1, x2, y2 = line_box
    W = max(1, x2 - x1)
    H = max(1, y2 - y1)
    try:
        font = ImageFont.truetype(font_path or FALLBACK_FONT, size=max(16, H//2))
    except Exception:
        font = ImageFont.load_default()
    words = line_text.split()
    if not words:
        return []
    def w_px(s): return text_width(font, s)
    total_px = w_px(" ".join(words))
    total_px = max(1, total_px)
    out = []
    cursor_px = 0
    for i, w in enumerate(words):
        prefix = " ".join(words[:i])
        prefix_px = w_px(prefix + (" " if prefix else ""))
        word_px = w_px(w)
        start_rel = prefix_px / total_px
        end_rel   = min(1.0, (prefix_px + word_px) / total_px)
        wx1 = int(x1 + start_rel * W)
        wx2 = int(x1 + end_rel   * W)
        out.append((w, [wx1, y1, wx2, y2]))
    return out

def frame_words_from_ocr(ocr_frame, font_path=None):
    items = []
    for text, box in zip(ocr_frame["texts"], ocr_frame["boxes"]):
        for w, wbox in split_line_into_word_boxes(text, box, font_path=font_path):
            items.append((w, wbox))
    # sort by Y then X for deterministic reading order
    items.sort(key=lambda it: (it[1][1], it[1][0]))
    return items

# ===== align a full translation back to the word boxes =====
def align_phrase_to_word_boxes(eng_words, eng_boxes, translated_phrase, min_chars_per_box=1):
    try:
        eng_font = ImageFont.truetype(FALLBACK_FONT, size=24)
    except Exception:
        eng_font = ImageFont.load_default()

    tgt = translated_phrase
    total_chars = len(' '.join(tgt))
    if total_chars == 0 or len(eng_words) == 0:
        return [("", box) for box in eng_boxes]

    widths = [max(1, text_width(eng_font, w)) for w in eng_words]
    sum_w = sum(widths)
    alloc_float = [total_chars * (w / sum_w) for w in widths]
    alloc = [max(min_chars_per_box if eng_words[i].strip() else 0, int(round(a))) for i, a in enumerate(alloc_float)]

    drift = total_chars - sum(alloc)
    fracs = [(i, alloc_float[i] - int(alloc_float[i])) for i in range(len(eng_words))]
    fracs.sort(key=lambda x: x[1], reverse=(drift > 0))
    j = 0
    safety = 0
    while drift != 0 and safety < 10000 and len(fracs) > 0:
        i = fracs[j][0]
        if drift > 0:
            alloc[i] += 1
            drift -= 1
        else:
            if alloc[i] > (min_chars_per_box if eng_words[i].strip() else 0):
                alloc[i] -= 1
                drift += 1
        j = (j + 1) % len(fracs)
        safety += 1

    out = []
    cursor = 0
    for i, box in enumerate(eng_boxes):
        n = alloc[i]
        seg = tgt[cursor:cursor + n] if n > 0 else ""
        cursor += n
        out.append((seg, box))
    if cursor < total_chars and len(out) > 0:
        seg_last, box_last = out[-1]
        out[-1] = (seg_last + tgt[cursor:], box_last)
    return out

# ===== translation (Gemini; falls back to echo) =====
def translate_texts(texts, target_lang):
    api_key = GEMINI_API_KEY
    if not api_key:
        # Fallback: return original (dev mode)
        return texts[:]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    out = []
    for t in texts:
        try:
            prompt = f"Translate the following text to {target_lang}. Respond with {target_lang} text only:\n\n{t}"
            payload = {"contents":[{"parts":[{"text": prompt}]}]}
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            tr = data["candidates"][0]["content"]["parts"][0]["text"]
            out.append(tr.strip())
        except Exception as e:
            print("Translation error, returning source:", e)
            out.append(t)
    return out

from pathlib import Path

def load_trans_cache(cache_path):
    """Load {lang: {phrase: translation}} dict."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_trans_cache(cache, cache_path):
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def translate_phrases_with_cache(phrases, target_lang, cache_path, translate_fn):
    """
    phrases: list[str] (full phrases/sentences)
    target_lang: e.g. 'gu'
    cache_path: where to store translations JSON
    translate_fn: your existing function that accepts list[str] and returns list[str]
    """
    cache = load_trans_cache(cache_path)
    cache.setdefault(target_lang, {})

    to_translate = []
    for ph in phrases:
        key = ph.strip()
        if key and key not in cache[target_lang]:
            to_translate.append(key)

    # call API only for missing phrases
    if to_translate:
        new_trans = translate_fn(to_translate, target_lang)
        for src, tgt in zip(to_translate, new_trans):
            cache[target_lang][src] = tgt.strip()

        save_trans_cache(cache, cache_path)

    # build mapping for requested phrases
    return {ph: cache[target_lang].get(ph.strip(), ph) for ph in phrases}

def split_translation(original_words, translated_phrase):
    # Count characters in original words (excluding punctuation if desired)
    total_chars = sum(len(w) for w in original_words)
    trans_chars = list(translated_phrase)
    result = []
    idx = 0
    for w in original_words:
        proportion = len(w) / total_chars
        take = max(1, round(len(trans_chars) * proportion))
        part = "".join(trans_chars[idx: idx + take])
        idx += take
        result.append(part)
    return result

# ===== overlay helpers =====
# def overlay_text_on_frame_multi(frame_path, overlays, font_path=INDIC_FONT, bg_rgba=SEMITRANSPARENT_RECT):
#     """
#     overlays: list of (text, [x1,y1,x2,y2])
#     Draws semi-transparent boxes and text inside them.
#     """
#     base = Image.open(frame_path).convert("RGBA")
#     overlay = Image.new("RGBA", base.size, (0,100,0,255))
#     draw = ImageDraw.Draw(overlay)
#     for text, (x1,y1,x2,y2) in overlays:
#         # background
#         draw.rectangle([x1, y1, x2, y2], fill=bg_rgba)
#         # font fit: half the box height
#         h = max(16, (y2 - y1) // 2)
#         font = ensure_font(h, prefer=font_path)
#         # simple left-top placement (you can add wrapping if needed)
#         draw.text((x1, y1), text, fill=(255,255,255,255), font=font)
#     # composite
#     out = Image.alpha_composite(base, overlay)
#     return out.convert("RGB")

def overlay_text_on_frame_multi(frame_path, eng_words, eng_boxes, translated_phrase, font_path=INDIC_FONT, bg_rgba=(0, 100, 0, 255)):
    """
    Draws translated text word-by-word aligned to the original English word boxes.
    - eng_words: list of original English words (OCR order)
    - eng_boxes: list of [x1, y1, x2, y2] coords per word
    - translated_phrase: full phrase translated to target language
    """
    # Align translated words with original word boxes
    aligned_translations = align_phrase_to_word_boxes(eng_words, eng_boxes, translated_phrase)

    # Load base image
    base = Image.open(frame_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,100,0,255))  # transparent layer
    draw = ImageDraw.Draw(overlay)

    for trans_word, (x1, y1, x2, y2) in aligned_translations:
        # Draw solid background
        draw.rectangle([x1, y1, x2, y2], fill=bg_rgba)

        # Adaptive font sizing: half box height
        h = max(16, (y2 - y1) // 2)
        font = ensure_font(h, prefer=font_path)

        # Draw translated word
        draw.text((x1, y1), trans_word, fill=(255, 255, 255, 255), font=font)

    # Composite overlay with base frame
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")

# ===== main pipeline =====
def main(input_video, output_folder, target_lang="gu"):
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    out_dir = os.path.join(output_folder, f"{video_name}-{target_lang}")
    frames_dir = os.path.join(out_dir, "frames")
    overlay_frames_dir = os.path.join(out_dir, "overlay_frames")
    os.makedirs(overlay_frames_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1) frames
    frames = extract_frames(input_video, frames_dir, fps=OUTPUT_FPS)

    # 2) OCR or cache
    ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    cache_path = os.path.join(out_dir, "ocr_results_ordered.json")
    ocr_results = run_or_load_ocr(frames, cache_path, ocr)

    # 3) phrase ranges (stable)
    phrase_ranges = get_phrase_ranges_with_boxes(ocr_results, min_stable_frames=MIN_STABLE_FRAMES, include_empty=True)
    # collapse incremental junk into final phrases only
    phrase_ranges = collapse_to_single_final_phrase(phrase_ranges)

    # If you truly want ONE final phrase for the whole video, keep only the last non-empty:
    KEEP_ONLY_LAST_FINAL = False
    if KEEP_ONLY_LAST_FINAL:
        last_nonempty = next((r for r in reversed(phrase_ranges) if r["text"].strip()), None)
        phrase_ranges = [last_nonempty] if last_nonempty else []

    # 4) collect unique non-empty phrases chronologically for translation
    unique_phrases = []
    last = None
    for pr in phrase_ranges:
        if pr["text"] and pr["text"] != last:
            unique_phrases.append(pr["text"])
            last = pr["text"]
    unique_phrases = list(set(unique_phrases))
    trans_cache_path = os.path.join(out_dir, "translations_cache.json")

    phrase_to_trans = translate_phrases_with_cache(
    unique_phrases,
    target_lang,
    trans_cache_path,
    translate_texts  # your existing API wrapper
)
    # phrase_to_trans = dict(zip(unique_phrases, translations))

    # 5) render progressive overlays
    overlayed_paths = []
    print("Rendering progressive overlays...")
    for fidx, frame_path in enumerate(tqdm(frames)):
        # find active phrase range for this frame
        active = None
        for pr in phrase_ranges:
            if pr["start_frame"] <= fidx <= pr["end_frame"]:
                active = pr
                break

        if not active or not active["text"]:
            # empty board or no range -> just copy
            out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
            shutil.copy(frame_path, out_img_path)
            overlayed_paths.append(out_img_path)
            continue

        english_phrase = active["text"]
        translated_phrase = phrase_to_trans.get(english_phrase, "")
        translated_parts = split_translation(pr["text"].split(), translated_phrase)
        if not translated_phrase.strip():
            out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
            shutil.copy(frame_path, out_img_path)
            overlayed_paths.append(out_img_path)
            continue

        # Build English word boxes from the FIRST frame of this range
        first_idx = active["start_frame"]
        eng_items = frame_words_from_ocr(ocr_results[first_idx], font_path=FALLBACK_FONT)
        eng_words = [w for (w, _) in eng_items]
        eng_boxes = [b for (_, b) in eng_items]

        
        # Align full translated phrase to the English word boxes
        aligned_segments = align_phrase_to_word_boxes(eng_words, eng_boxes, translated_parts, min_chars_per_box=1)

        # Progressive reveal within the range
        s, e = active["start_frame"], active["end_frame"]
        total = max(1, e - s + 1)
        progress = (fidx - s + 1) / total
        overlays = []
        for seg, box in aligned_segments:
            n = len(seg)
            show = max(0, int(math.ceil(n * progress)))
            overlays.append((seg[:show], box))

        # Draw
        img = overlay_text_on_frame_multi(frame_path=frame_path,
                                          eng_words=eng_words,
                                          eng_boxes=eng_boxes,
                                          translated_phrase=translated_phrase,
                                          font_path=INDIC_FONT,
                                          bg_rgba=SEMITRANSPARENT_RECT)
        # img = overlay_text_on_frame_multi(frame_path, overlays, font_path=INDIC_FONT, bg_rgba=SEMITRANSPARENT_RECT)
        out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
        img.save(out_img_path)
        overlayed_paths.append(out_img_path)

    # 6) audio & mux
    audio_path = os.path.join(out_dir, "audio.mp3")
    print("Extracting audio...")
    AudioFileClip(input_video).write_audiofile(audio_path)

    print("Muxing video...")
    clip = ImageSequenceClip(overlayed_paths, fps=OUTPUT_FPS)
    audio = AudioFileClip(audio_path)
    clip_with_audio = clip.with_audio(audio)
    out_video = os.path.join(out_dir, f"{video_name}_{target_lang}_overlay.mp4")
    clip_with_audio.write_videofile(out_video, codec="libx264", audio_codec="aac")
    print("Done:", out_video)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blackboard video → Indic overlay (phrase-level translation, word-level placement)")
    parser.add_argument("input_video", help="Path to input .mp4")
    parser.add_argument("output_folder", help="Output directory")
    parser.add_argument("--lang", default="gu", help="Target Indic language code (e.g., gu, hi, bn)")
    args = parser.parse_args()
    main(args.input_video, args.output_folder, target_lang=args.lang)
