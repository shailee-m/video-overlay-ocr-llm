import os
import cv2
from paddleocr import PaddleOCR
from moviepy import ImageSequenceClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import requests
import shutil
from tqdm import tqdm
import hashlib
from google import genai

# ==== CONFIG ====
TARGET_LANG = 'gu' # Set to desired Indic language code (e.g. 'gu' for Gujarati, 'hi' for Hindi, etc.)
OUTPUT_FPS = 20   # Adjust as per input video for best sync
GEMINI_API_KEY = ""  # Replace with your actual Gemini API Key
# ImageSequenceClip, AudioFileClip = editor.ImageSequenceClip, editor.AudioFileClip


# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=GEMINI_API_KEY)  # Replace with your actual Gemini API Key

def hash_texts(texts):
    """Hashes concatenated texts for quick change detection."""
    m = hashlib.md5()
    for t in texts:
        m.update(t.encode("utf-8"))
    return m.hexdigest()


def extract_frames(video_path, frames_dir, fps=OUTPUT_FPS):
    if os.path.exists(frames_dir):
        frames = []
        for filename in os.listdir(frames_dir):
            frames.append(os.path.join(frames_dir, filename))
        frames.sort(key=lambda x: x)
        return frames

    
    # Create frames directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    input_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(input_fps // fps)
    frame_idx = 0
    save_idx = 0
    frames = []

    print("Extracting frames...")
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if frame_idx % frame_interval == 0:
            frame_file = os.path.join(frames_dir, f"frame_{save_idx:05d}.png")
            cv2.imwrite(frame_file, image)
            frames.append(frame_file)
            save_idx += 1
        frame_idx += 1
    vidcap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames

def extract_texts_and_boxes(result):
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
            output.append((text, (int(x1), int(y1), int(x2), int(y2)), float(score)))
    ##########################################################
    # If result is a dict (advanced output, as in your example)
    # if isinstance(result, dict) and "res" in result:
    #     res = result["res"]
    #     texts = res.get("rec_texts", [])
    #     boxes = res.get("rec_boxes", [])
    #     scores = res.get("rec_scores", [])
    #     for text, box, score in zip(texts, boxes, scores):
    #         # box shape: (4, 2) [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    #         x_coords = [pt[0] for pt in box]
    #         y_coords = [pt[1] for pt in box]
    #         x1, y1 = min(x_coords), min(y_coords)
    #         x2, y2 = max(x_coords), max(y_coords)
    #         output.append((text, (int(x1), int(y1), int(x2), int(y2)), float(score)))

    # # If result is a list (default output from ocr.ocr(image))
    # elif isinstance(result, list) and len(result) > 0:
    #     for line in result[0]:
    #         # line = [bbox, (text, score)]
    #         bbox = line[0]
    #         text, score = line[1]
    #         x_coords = [pt[0] for pt in bbox]
    #         y_coords = [pt[1] for pt in bbox]
    #         x1, y1 = min(x_coords), min(y_coords)
    #         x2, y2 = max(x_coords), max(y_coords)
    #         output.append((text, (int(x1), int(y1), int(x2), int(y2)), float(score)))

    return output


def detect_text_in_frame(ocr, frame_path):
    # Returns [(text, (x1, y1, x2, y2)), ...]
    result = ocr.ocr(frame_path)
    detected = []
    # print(f"Detected {len(result[0])} text lines in frame {frame_path}")
    # final_op = result.to_json();
    # for line in result[0]:
    #     txt = line[0][0]
    #     bbox = line[1]
    #     # Defensive: bbox should be a list of 4 points, each [x, y]
    #     try:
    #         if (
    #             isinstance(bbox, list)
    #             and len(bbox) == 4
    #             and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in bbox)
    #             and all(isinstance(coord, (int, float)) for pt in bbox for coord in pt)
    #         ):
    #             x1, y1 = int(bbox[0][0]), int(bbox[0][1])
    #             x2, y2 = int(bbox[2][0]), int(bbox[2][1])
    #             detected.append((txt, (x1, y1, x2, y2)))
    #         else:
    #             print(f"Warning: Skipping invalid bbox in frame {frame_path}: {bbox}")
    #     except Exception as e:
    #         print(f"Warning: Error parsing bbox in frame {frame_path}: {bbox} ({e})")
    # return detected
    texts_and_boxes = extract_texts_and_boxes(result)

    # Example:
    for text, (x1, y1, x2, y2), score in texts_and_boxes:
        print(text, x1, y1, x2, y2, score)
    return texts_and_boxes


def translate_texts(texts, target_lang):
    # Google Gemini translation endpoint (pseudo code, update if using Sarvam or other)
    # You may need to batch or call per line depending on rate limit
    translated = []
    for txt in texts:
        try:
            response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Translate the following text to " + target_lang + ". Give the output in " + target_lang + " only. If there is nothing to translate, return empty string. Only return translated text. Here is the source text to translate: " + txt
        )
            result = response.text
        except Exception as e:
            print(f"Error translating: {txt} - {e}")
            result = txt
        translated.append(result)
    return translated

def overlay_text_on_frame(frame_path, texts_and_boxes, translations, font_path=None):
    img = Image.open(frame_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Use system font for Gujarati/Indic (Noto Sans Gujarati is good; else fallback to default)
    if not font_path:
        font_path = "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # fallback

    for (orig_text, (x1, y1, x2, y2), score), translation in zip(texts_and_boxes, translations):
        # Draw semi-transparent rectangle
        draw.rectangle([x1, y1, x2, y2], fill=(91,0,33,75))
        # Draw translation, font size adaptive
        try:
            font = ImageFont.truetype(font_path, size=max(16, (y2-y1)//2))
        except Exception:
            font = ImageFont.load_default()
        draw.text([x1, y1, x2, y2], translation, fill="white", font=font)
    return img

def extract_audio(input_video, out_audio):
    videoclip = AudioFileClip(input_video)
    videoclip.write_audiofile(out_audio)
    return out_audio

def get_phrase_ranges_with_boxes(ocr_results):
    ranges = []
    if not ocr_results:
        return ranges

    current_text = " ".join(ocr_results[0]["texts"]).strip()
    current_boxes = ocr_results[0]["boxes"]
    start_frame = 0

    for i in range(1, len(ocr_results)):
        t = " ".join(ocr_results[i]["texts"]).strip()
        boxes = ocr_results[i]["boxes"]
        if t != current_text:
            ranges.append({
                "start_frame": start_frame,
                "end_frame": i - 1,
                "text": current_text,
                "orig_text_array": ocr_results[i]["texts"],
                "boxes": current_boxes
            })
            start_frame = i
            current_text = t
            current_boxes = boxes

    # Add the last range
    ranges.append({
        "start_frame": start_frame,
        "end_frame": len(ocr_results) - 1,
        "text": current_text,
        "boxes": current_boxes
    })
    return ranges


def get_phrase_ranges(all_texts_per_frame):
    """
    Given a list of per-frame OCR results (texts as strings, in order), returns an array:
    [{ "start_frame": int, "end_frame": int, "text": str }]
    Each range represents a continuous period where the text on the board is stable.
    All frames are assigned to a range, including empty board periods.
    """
    ranges = []
    if not all_texts_per_frame:
        return ranges

    current_text = all_texts_per_frame[0]
    start_frame = 0

    for i in range(1, len(all_texts_per_frame)):
        text = all_texts_per_frame[i]
        if text != current_text:
            ranges.append({
                "start_frame": start_frame,
                "end_frame": i - 1,
                "text": current_text
            })
            start_frame = i
            current_text = text

    # Add the final range
    ranges.append({
        "start_frame": start_frame,
        "end_frame": len(all_texts_per_frame) - 1,
        "text": current_text
    })
    return ranges

def get_unique_growing_phrases(phrase_ranges):
    """
    From ranges [{start_frame, end_frame, text}], return
    a list of unique, growing phrases (ignoring repeats).
    Returns: [text1, text2, ...] in order of first appearance
    """
    phrases = []
    last = ""
    for rng in phrase_ranges:
        t = rng["text"]
        if t and t != last:
            phrases.append(t)
            last = t
    return phrases


# def main(input_video, output_folder, target_lang=TARGET_LANG):
#     video_name = os.path.splitext(os.path.basename(input_video))[0]
#     out_dir = os.path.join(output_folder, f"{video_name}-{target_lang}")
#     frames_dir = os.path.join(out_dir, "frames")
#     overlay_frames_dir = os.path.join(out_dir, "overlay_frames")
#     os.makedirs(overlay_frames_dir, exist_ok=True)
#     os.makedirs(out_dir, exist_ok=True)

#     # Step 1: Frame Extraction
#     frames_1 = extract_frames(input_video, frames_dir)
#     frames = frames_1 #[:15]  # Limit to first 15 frames for demo; remove this line for full video
#     # Step 2: OCR Init
#     ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)
#     # ocr = PaddleOCR(lang='en', use_angle_cls=True)  # Set use_gpu=True if CUDA GPU present  --, use_gpu=False old doc
#     texts_and_boxes_cache = []
#     frame_to_overlay_frame = []  # For each frame, which overlay frame should be used

#     last_text_hash = None
#     last_overlay_idx = None
#     overlayed_paths = []

#     print("Processing and overlaying translations...")
#     # for frame_path in tqdm(frames):
#     #     # Step 2: Text Detection
#     #     texts_and_boxes = detect_text_in_frame(ocr, frame_path)

#     #     if not texts_and_boxes:
#     #         # No text, just copy frame
#     #         out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
#     #         shutil.copy(frame_path, out_img_path)
#     #         overlayed_paths.append(out_img_path)
#     #         print(f"No text detected in frame {frame_path}, copied without overlay.")
#     #         continue

#     #     texts = [tb[0] for tb in texts_and_boxes]
#     #     # Compute hash for detected text in this frame
#     #     text_hash = hash_texts(texts)
#     #     if text_hash != last_text_hash:
            
#     #         # Step 3: Translate
#     #         translations = translate_texts(texts, target_lang)
#     #         print(f"Translations for frame {frame_path}: {translations}")
#     #         if not translations:
#     #             print(f"No translations for frame {frame_path}, skipping overlay.")
#     #             continue
#     #     # Step 4: Overlay
#     #     img_with_overlay = overlay_text_on_frame(frame_path, texts_and_boxes, translations)
#     #     out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
#     #     img_with_overlay.save(out_img_path)
#     #     overlayed_paths.append(out_img_path)
#     for idx, frame_path in enumerate(frames):
#         texts_and_boxes = detect_text_in_frame(ocr, frame_path)

#         if not texts_and_boxes:
#             # No text, just copy frame
#             out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
#             shutil.copy(frame_path, out_img_path)
#             overlayed_paths.append(out_img_path)
#             print(f"No text detected in frame {frame_path}, copied without overlay.")
#             continue
#         texts = [tb[0] for tb in texts_and_boxes]

#         # Compute hash for detected text in this frame
#         text_hash = hash_texts(texts)

#         if text_hash != last_text_hash:
#             # This is a key frame (text has changed)
#             # (Translate, overlay, and store index)
#             translation = translate_texts(texts, target_lang)
#             print(f"Translations for frame {frame_path}: {translation}")
#             if not translation:
#                 print(f"No translations for frame {frame_path}, skipping overlay.")
#                 continue
#             img_with_overlay = overlay_text_on_frame(frame_path, texts_and_boxes, translation)
#             out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
#             img_with_overlay.save(out_img_path)
#             texts_and_boxes_cache.append((texts_and_boxes, translation, out_img_path))
#             last_overlay_idx = len(texts_and_boxes_cache) - 1
#             last_text_hash = text_hash

#         # Record which key overlay to use for this frame
#         frame_to_overlay_frame.append(last_overlay_idx)
#     # Now, when assembling the video, use:
#     overlayed_paths = [texts_and_boxes_cache[i][2] for i in frame_to_overlay_frame]

#     # Step 5: Audio Extraction
#     audio_path = os.path.join(out_dir, "audio.mp3")
#     extract_audio(input_video, audio_path)

#     # Step 6: Re-assemble Video
#     print("Re-assembling video with overlays and original audio...")
#     clip = ImageSequenceClip(overlayed_paths, fps=OUTPUT_FPS)
#     audio_clip = AudioFileClip(audio_path)
#     final_video = clip.with_audio(audio_clip)
#     out_video_path = os.path.join(out_dir, f"{video_name}_{target_lang}_overlay.mp4")
#     final_video.write_videofile(out_video_path, codec='libx264', audio_codec='aac')

#     print("Done! Output at:", out_video_path)

# def main(input_video, output_folder, target_lang=TARGET_LANG):
#     video_name = os.path.splitext(os.path.basename(input_video))[0]
#     out_dir = os.path.join(output_folder, f"{video_name}-{target_lang}")
#     frames_dir = os.path.join(out_dir, "frames")
#     overlay_frames_dir = os.path.join(out_dir, "overlay_frames")
#     os.makedirs(overlay_frames_dir, exist_ok=True)
#     os.makedirs(out_dir, exist_ok=True)

#     # Step 1: Frame Extraction
#     frames = extract_frames(input_video, frames_dir)

#     # Step 2: OCR Init
#     ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False,
#                     use_doc_unwarping=False, use_textline_orientation=False)

#     # --- Phrase Detection Phase ---
#     previous_text = ""
#     writing = False
#     start_frame = None
#     stable_count = 0
#     PHRASE_STABLE_THRESHOLD = 8

#     phrase_ranges = []  # [(start_frame, end_frame, phrase, boxes)]
#     all_texts_and_boxes = []

#     print("Grouping frames by written phrases...")

#     for idx, frame_path in enumerate(tqdm(frames)):
#         texts_and_boxes = detect_text_in_frame(ocr, frame_path)
#         all_texts_and_boxes.append(texts_and_boxes)
#         texts = [tb[0] for tb in texts_and_boxes]
#         boxes = [tb[1] for tb in texts_and_boxes]
#         current_text = " ".join(texts).strip()

#         if current_text != previous_text:
#             if not writing and current_text != "":
#                 writing = True
#                 start_frame = idx
#             stable_count = 0
#             current_phrase = current_text
#             current_boxes = boxes
#         else:
#             if writing and current_text != "":
#                 stable_count += 1
#                 if stable_count >= PHRASE_STABLE_THRESHOLD:
#                     end_frame = idx - stable_count
#                     if current_phrase:
#                         phrase_ranges.append((start_frame, end_frame, current_phrase, current_boxes))
#                     writing = False
#                     stable_count = 0
#         previous_text = current_text

#     # Handle any final phrase if video ends while writing
#     if writing and current_text != "":
#         phrase_ranges.append((start_frame, len(frames) - 1, current_phrase, current_boxes))

#     # --- Translation Phase ---
#     print("Translating phrases...")
#     translations = []
#     for _, _, phrase, _ in phrase_ranges:
#         translations.append(translate_texts([phrase], target_lang)[0])

#     # --- Overlay Phase (progressive, word-by-word) ---
#     print("Overlaying translated phrases progressively...")
#     overlayed_paths = []
#     for idx, frame_path in enumerate(tqdm(frames)):
#         # Find which phrase applies to this frame
#         overlay_text = ""
#         overlay_boxes = []
#         for (start, end, phrase, boxes), translation in zip(phrase_ranges, translations):
#             if start <= idx <= end:
#                 frames_in_range = end - start + 1
#                 # Progressive word-by-word reveal:
#                 words = translation.split()
#                 words_to_show = max(1, int(len(words) * (idx - start + 1) / frames_in_range))
#                 overlay_text = " ".join(words[:words_to_show])
#                 overlay_boxes = boxes
#                 break  # Only use the first matching phrase
#         # If nothing to overlay, just copy the frame
#         if overlay_text == "":
#             out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
#             shutil.copy(frame_path, out_img_path)
#             overlayed_paths.append(out_img_path)
#             continue
#         # Overlay text on frame (use the boxes from phrase start)
#         img_with_overlay = overlay_text_on_frame(
#             frame_path,
#             [(overlay_text, b, 1.0) for b in overlay_boxes],  # dummy score
#             [overlay_text] * len(overlay_boxes)
#         )
#         out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
#         img_with_overlay.save(out_img_path)
#         overlayed_paths.append(out_img_path)

#     # --- Audio Extraction ---
#     audio_path = os.path.join(out_dir, "audio.mp3")
#     extract_audio(input_video, audio_path)

#     # --- Re-assemble Video ---
#     print("Re-assembling video with overlays and original audio...")
#     from moviepy import ImageSequenceClip, AudioFileClip
#     clip = ImageSequenceClip(overlayed_paths, fps=OUTPUT_FPS)
#     audio_clip = AudioFileClip(audio_path)
#     clip_with_audio = clip.with_audio(audio_clip)
#     out_video_path = os.path.join(out_dir, f"{video_name}_{target_lang}_overlay.mp4")
#     clip_with_audio.write_videofile(out_video_path, codec='libx264', audio_codec='aac')

#     print("Done! Output at:", out_video_path)


from PIL import Image, ImageDraw, ImageFont

def overlay_text_on_frame_multi(frame_path, overlays, font_path=None):
    # overlays: list of (text_to_overlay, (x1, y1, x2, y2))
    img = Image.open(frame_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    if not font_path:
        font_path = "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # fallback
    for text, (x1, y1, x2, y2) in overlays:
        draw.rectangle([x1, y1, x2, y2], fill=(0,100,0,1))
        try:
            font = ImageFont.truetype(font_path, size=max(16, (y2-y1)//2))
        except Exception:
            font = ImageFont.load_default()
        draw.text((x1, y1), text, fill="white", font=font)
    return img


def main(input_video, output_folder, target_lang=TARGET_LANG):
    
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    out_dir = os.path.join(output_folder, f"{video_name}-{target_lang}")
    frames_dir = os.path.join(out_dir, "frames")
    overlay_frames_dir = os.path.join(out_dir, "overlay_frames")
    os.makedirs(overlay_frames_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    ocr_results_file = os.path.join(out_dir,  "ocr_results.json")
    ocr_results = []

     # Step 1: Frame Extraction
    frames_1 = extract_frames(input_video, frames_dir)
    frames = frames_1 #[:15]  # Limit to first 15 frames for demo; remove this line for full video

    import json

    # Step 2: OCR Init
    # Check if OCR results already exist
    if not os.path.exists(ocr_results_file):
        ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)

        for idx, frame_path in  enumerate(tqdm(frames)):
            texts_and_boxes = detect_text_in_frame(ocr, frame_path)
            # texts_and_boxes: [(text, (x1, y1, x2, y2), score), ...]
            ocr_results.append({
                "index": idx,
                "frame": os.path.basename(frame_path),  # Optional: for human checking
                "texts": [t[0] for t in texts_and_boxes],
                "boxes": [t[1] for t in texts_and_boxes],
                "scores": [t[2] for t in texts_and_boxes]
            })
        
        ocr_results.sort(key=lambda x: x["index"])
        with open(ocr_results_file, "w", encoding="utf8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            
    else:
        with open(ocr_results_file, "r", encoding="utf8") as f:
            ocr_results = json.load(f)
            ocr_results.sort(key=lambda x: x["index"])


    
    phrase_ranges = get_phrase_ranges_with_boxes(ocr_results)
    phrases_to_translate = get_unique_growing_phrases(phrase_ranges)
    translated_phrases = translate_texts(phrases_to_translate, target_lang)
    text_to_translation = dict(zip(phrases_to_translate, translated_phrases))

     # 5. Assign translation to each range
    for rng in phrase_ranges:
        rng["translation"] = text_to_translation.get(rng["text"], "")

    # 6. For each frame, overlay char-by-char per line
    overlayed_paths = []
    print("Creating overlays...")
    for idx, frame_path in enumerate(tqdm(frames)):
        # Find range
        rng = next(r for r in phrase_ranges if r["start_frame"] <= idx <= r["end_frame"])
        translation = rng["translation"]
        boxes = rng["boxes"]
        texts = rng["text"].splitlines() if "\n" in rng["text"] else [rng["text"]]
        total_frames = rng["end_frame"] - rng["start_frame"] + 1
        i = idx - rng["start_frame"]
        # Per line, char-by-char reveal
        overlays = []
        if translation and boxes:
            # If you want to split by line, adjust translation splitting to match boxes
            # (assume 1:1 mapping of translation to box for simplicity)
            lines = translation.split("\n") if "\n" in translation else [translation]
            for line_text, box in zip(lines, boxes):
                num_chars = max(1, int(len(line_text) * (i + 1) / total_frames))
                overlays.append((line_text[:num_chars], box))
        if overlays:
            img_with_overlay = overlay_text_on_frame_multi(frame_path, overlays)
            out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
            img_with_overlay.save(out_img_path)
            overlayed_paths.append(out_img_path)
        else:
            out_img_path = os.path.join(overlay_frames_dir, os.path.basename(frame_path))
            shutil.copy(frame_path, out_img_path)
            overlayed_paths.append(out_img_path)
    # ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False,
    # use_doc_unwarping=False,
    # use_textline_orientation=False)
    # # ocr = PaddleOCR(lang='en', use_angle_cls=True)  # Set use_gpu=True if CUDA GPU present  --, use_gpu=False old doc
    # texts_and_boxes_cache = []


    # --- Audio Extraction ---
    audio_path = os.path.join(out_dir, "audio.mp3")
    extract_audio(input_video, audio_path)

    # --- Re-assemble Video ---
    print("Re-assembling video with overlays and original audio...")
    from moviepy import ImageSequenceClip, AudioFileClip
    clip = ImageSequenceClip(overlayed_paths, fps=OUTPUT_FPS)
    audio_clip = AudioFileClip(audio_path)
    clip_with_audio = clip.with_audio(audio_clip)
    out_video_path = os.path.join(out_dir, f"{video_name}_{target_lang}_overlay.mp4")
    clip_with_audio.write_videofile(out_video_path, codec='libx264', audio_codec='aac')

    print("Done! Output at:", out_video_path)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video OCR → Indic Translation → Overlay Pipeline")
    parser.add_argument("input_video", help="Path to input mp4 video")
    parser.add_argument("output_folder", help="Folder to save output frames and video")
    parser.add_argument("--lang", help="Target Indic language code (default: gu for Gujarati)", default="gu")
    args = parser.parse_args()
    main(args.input_video, args.output_folder, target_lang=args.lang)
