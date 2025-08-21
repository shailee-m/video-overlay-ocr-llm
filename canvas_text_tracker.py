"""
Canvas Text Tracker (skeleton)
-----------------------------
Tracks words and sentences drawn/typed on a canvas across frames.
Handles partial/complete deletion, mutating text, and "stable" snapshots for translation + caching.

Quick start:
    from canvas_text_tracker import CanvasTextTracker, OCRToken

    tracker = CanvasTextTracker()
    frame = 0
    ocr_tokens = [
        OCRToken(text="t", bbox=(100,100,112,120), conf=0.95),
        OCRToken(text="l", bbox=(130,100,138,120), conf=0.92),
    ]
    tracker.update(frame, ocr_tokens)

    # ... after more frames ...
    stable_sentences = tracker.get_stable_sentences()
    for s in stable_sentences:
        print(s.sentence_id, s.last_stable_text, s.start_frame, s.end_frame)

Design notes:
- WordTrack: tracks a single logical word through time; maintains last stable text and frame.
- SentenceTrack: ordered list of WordTracks in a line; maintains stable snapshots and translation cache keys.
- Matching: greedy cost-based assignment using IoU, text distance, and center distance.
- States: EMERGING / STABLE / MUTATING / ERASING / DEAD
- Translation: stubbed; integrate your MT provider and alignment if needed.
- Caching: JSON file "translations_cache.json" (path configurable).

Author: (your name)
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json
import math
import os
import time
import uuid

# ---------------------------
# Data structures
# ---------------------------

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

@dataclass
class OCRToken:
    text: str
    bbox: BBox
    conf: float = 1.0

    def normalized_text(self) -> str:
        return " ".join(self.text.strip().lower().split())

@dataclass
class WordTrack:
    word_id: str
    last_text: str
    last_bbox: BBox
    start_frame: int
    last_seen_frame: int
    last_stable_text: str
    last_stable_frame: int
    state: str = "EMERGING"
    miss_count: int = 0
    # History kept small; expand if needed
    bbox_history: List[Tuple[int, BBox]] = field(default_factory=list)
    text_history: List[Tuple[int, str]] = field(default_factory=list)

@dataclass
class StableSnapshot:
    frame: int
    text: str

@dataclass
class SentenceTrack:
    sentence_id: str
    word_ids: List[str] = field(default_factory=list)
    start_frame: int = 0
    last_seen_frame: int = 0
    state: str = "EMERGING"
    last_stable_text: str = ""
    last_stable_frame: int = -1
    stable_snapshots: List[StableSnapshot] = field(default_factory=list)
    translation: Optional[Dict[str, Any]] = None  # {"gu": "...", "alignment": [...]}

# ---------------------------
# Utility functions
# ---------------------------

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

def center_distance(a: BBox, b: BBox) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)

def normalized_levenshtein(a: str, b: str) -> float:
    """
    Returns normalized Levenshtein distance in [0,1], 0 = identical, 1 = completely different.
    Simple DP implementation to avoid external deps. Good enough for short tokens.
    """
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if a == b:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1,      # deletion
                        dp[j - 1] + 1,  # insertion
                        prev + cost)    # substitution
            prev = cur
    dist = dp[n]
    return dist / max(m, n)

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# ---------------------------
# Translation cache (simple JSON file)
# ---------------------------

class TranslationCache:
    def __init__(self, path: str = "translations_cache.json"):
        self.path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def put(self, key: str, value: Dict[str, Any]):
        self._data[key] = value
        self.save()

# ---------------------------
# Core tracker
# ---------------------------

class CanvasTextTracker:
    def __init__(self,
                 i0: float = 0.30, i1: float = 0.50,
                 e0: float = 0.40, e1: float = 0.20,
                 S: int = 4, M: int = 3, T: int = 3,
                 alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1,
                 cache_path: str = "translations_cache.json"):
        self.i0, self.i1, self.e0, self.e1 = i0, i1, e0, e1
        self.S, self.M, self.T = S, M, T
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.word_tracks: Dict[str, WordTrack] = {}
        self.sentence_tracks: Dict[str, SentenceTrack] = {}
        self.cache = TranslationCache(cache_path)

    # ---------- Public API ----------

    def update(self, frame_idx: int, ocr_tokens: List[OCRToken]) -> None:
        tokens = [OCRToken(text=t.normalized_text(), bbox=t.bbox, conf=t.conf) for t in ocr_tokens]

        # 1) Match tokens to word tracks
        assignments, new_tokens, dead_tracks = self._assign_tokens_to_tracks(frame_idx, tokens)

        # 2) Update matched tracks
        for (tok, track) in assignments:
            self._update_word_track_with_token(track, tok, frame_idx)

        # 3) Create new word tracks for unmatched tokens
        for tok in new_tokens:
            wid = new_id("w")
            self.word_tracks[wid] = WordTrack(
                word_id=wid,
                last_text=tok.text,
                last_bbox=tok.bbox,
                start_frame=frame_idx,
                last_seen_frame=frame_idx,
                last_stable_text="",
                last_stable_frame=-1,
                state="EMERGING",
                miss_count=0,
                bbox_history=[(frame_idx, tok.bbox)],
                text_history=[(frame_idx, tok.text)]
            )

        # 4) Process dead tracks (close them)
        for track in dead_tracks:
            self._finalize_dead_word_track(track)

        # 5) Rebuild sentence tracks from active word tracks
        self._rebuild_sentence_tracks(frame_idx)

        # 6) Try translating stable sentences
        self._translate_stable_sentences(frame_idx)

    def get_active_words(self) -> List[WordTrack]:
        return [w for w in self.word_tracks.values() if w.state != "DEAD"]

    def get_all_sentences(self) -> List[SentenceTrack]:
        return list(self.sentence_tracks.values())

    def get_stable_sentences(self) -> List[SentenceTrack]:
        out = []
        for s in self.sentence_tracks.values():
            if s.last_stable_frame >= 0:
                out.append(s)
        return out

    # ---------- Matching & updates ----------

    def _assign_tokens_to_tracks(self, frame_idx: int, tokens: List[OCRToken]):
        active_tracks = [w for w in self.word_tracks.values() if w.state != "DEAD"]
        pairs = []  # (cost, token_idx, track_idx)
        for ti, tok in enumerate(tokens):
            for wi, track in enumerate(active_tracks):
                c = self._match_cost(tok, track)
                pairs.append((c, ti, wi))

        # Greedy: pick lowest cost unique matches under acceptance rules
        pairs.sort(key=lambda x: x[0])
        used_tokens = set()
        used_tracks = set()
        assignments = []
        for cost, ti, wi in pairs:
            if ti in used_tokens or wi in used_tracks:
                continue
            tok = tokens[ti]
            track = active_tracks[wi]
            if self._accept_match(tok, track, cost):
                assignments.append((tok, track))
                used_tokens.add(ti)
                used_tracks.add(wi)

        # Unmatched tokens
        new_tokens = [tokens[i] for i in range(len(tokens)) if i not in used_tokens]

        # Unmatched tracks: increase miss_count; if ≥ M, mark DEAD
        dead_tracks = []
        for wi, track in enumerate(active_tracks):
            if wi not in used_tracks:
                track.miss_count += 1
                if track.miss_count >= self.M:
                    track.state = "DEAD"
                    dead_tracks.append(track)
            else:
                track.miss_count = 0  # reset when matched
        return assignments, new_tokens, dead_tracks

    def _match_cost(self, tok: OCRToken, track: WordTrack) -> float:
        i = iou(tok.bbox, track.last_bbox)
        d_text = normalized_levenshtein(tok.text, track.last_text)
        d_center = center_distance(tok.bbox, track.last_bbox)
        # Normalize center distance by word height to keep scale reasonable
        h = max(1.0, track.last_bbox[3] - track.last_bbox[1])
        d_center_norm = d_center / h
        return self.alpha * (1.0 - i) + self.beta * d_text + self.gamma * d_center_norm

    def _accept_match(self, tok: OCRToken, track: WordTrack, cost: float) -> bool:
        # Accept if either spatial overlap or text similarity is strong
        i = iou(tok.bbox, track.last_bbox)
        d_text = normalized_levenshtein(tok.text, track.last_text)
        if i >= self.i0 or (1.0 - d_text) >= (1.0 - self.e0):
            return True
        return False

    def _update_word_track_with_token(self, track: WordTrack, tok: OCRToken, frame_idx: int) -> None:
        track.last_seen_frame = frame_idx
        track.last_bbox = tok.bbox
        track.last_text = tok.text
        track.bbox_history.append((frame_idx, tok.bbox))
        track.text_history.append((frame_idx, tok.text))

        # Stability check
        d_text = normalized_levenshtein(track.last_text, track.last_stable_text or track.last_text)
        i = iou(track.last_bbox, track.bbox_history[-1][1] if len(track.bbox_history) > 1 else track.last_bbox)

        if track.last_stable_frame < 0:
            # first time we consider stability
            if self._has_been_consistently_similar(track, frames=self.S):
                track.last_stable_text = track.last_text
                track.last_stable_frame = frame_idx
                track.state = "STABLE"
        else:
            # if change is small, keep STABLE else MUTATING
            if d_text <= self.e1 and self._has_been_consistently_similar(track, frames=self.S):
                track.last_stable_text = track.last_text
                track.last_stable_frame = frame_idx
                track.state = "STABLE"
            else:
                track.state = "MUTATING"

    def _has_been_consistently_similar(self, track: WordTrack, frames: int) -> bool:
        """
        Check last N frames for small text change (vs latest) and reasonable bbox overlap.
        """
        if len(track.text_history) < frames:
            return False
        latest_text = track.last_text
        latest_bbox = track.last_bbox
        recent = track.text_history[-frames:]
        for (f, t) in recent:
            if normalized_levenshtein(t, latest_text) > self.e1:
                return False
        # bbox stability
        recent_bboxes = [b for (_, b) in track.bbox_history[-frames:]]
        for b in recent_bboxes:
            if iou(b, latest_bbox) < self.i1:
                return False
        return True

    def _finalize_dead_word_track(self, track: WordTrack) -> None:
        # On death, end at last stable frame (not last seen) if available
        if track.last_stable_frame >= 0:
            track.last_seen_frame = track.last_stable_frame
        # else keep last_seen_frame as is
        track.state = "DEAD"

    # ---------- Sentences ----------

    def _rebuild_sentence_tracks(self, frame_idx: int) -> None:
        # Cluster words into lines by y-center proximity (simple heuristic)
        active_words = [w for w in self.word_tracks.values() if w.state != "DEAD"]
        lines: List[List[WordTrack]] = self._cluster_lines(active_words)

        # Build sentences per line by x-order and punctuation gaps
        new_sentence_tracks: Dict[str, SentenceTrack] = {}
        for line in lines:
            line_sorted = sorted(line, key=lambda w: (w.last_bbox[0], w.start_frame))
            # naive sentence splitting: break on punctuation at end or large x-gap
            current: List[WordTrack] = []
            for w in line_sorted:
                if not current:
                    current.append(w)
                    continue
                gap = w.last_bbox[0] - current[-1].last_bbox[2]
                avg_h = (w.last_bbox[3] - w.last_bbox[1] + current[-1].last_bbox[3] - current[-1].last_bbox[1]) / 2.0
                large_gap = gap > 0.6 * avg_h  # tune
                punct_break = current[-1].last_text.endswith(('.', '!', '?'))
                if large_gap or punct_break:
                    self._commit_sentence_track(new_sentence_tracks, current, frame_idx)
                    current = [w]
                else:
                    current.append(w)
            if current:
                self._commit_sentence_track(new_sentence_tracks, current, frame_idx)

        self.sentence_tracks = new_sentence_tracks

    def _commit_sentence_track(self, store: Dict[str, SentenceTrack], words: List[WordTrack], frame_idx: int):
        sid = new_id("s")
        st = SentenceTrack(sentence_id=sid)
        st.word_ids = [w.word_id for w in words]
        st.start_frame = min(w.start_frame for w in words)
        st.last_seen_frame = frame_idx
        full_text = " ".join((w.last_text or "").strip() for w in words if (w.last_text or "").strip())
        st.last_stable_text = full_text if self._sentence_words_stable(words) else ""
        st.last_stable_frame = frame_idx if st.last_stable_text else -1
        st.state = "STABLE" if st.last_stable_text else "EMERGING"
        if st.last_stable_text:
            st.stable_snapshots.append(StableSnapshot(frame=frame_idx, text=st.last_stable_text))
        store[sid] = st

    def _sentence_words_stable(self, words: List[WordTrack]) -> bool:
        # A sentence is stable if all constituent words are stable now or unchanged recently
        if not words:
            return False
        for w in words:
            if w.last_stable_frame < 0:
                return False
            # if a word is mutating heavily, sentence isn't stable
            if w.state == "MUTATING":
                return False
        return True

    # ---------- Translation ----------

    def _translate_stable_sentences(self, frame_idx: int) -> None:
        for s in self.sentence_tracks.values():
            text = s.last_stable_text.strip() if s.last_stable_text else ""
            if not text:
                continue
            # Use a simple "stable for T frames" check via snapshots
            # If the latest snapshot is newer than T-1 frames, we wait.
            if not s.stable_snapshots:
                continue
            last_snap = s.stable_snapshots[-1]
            if frame_idx - last_snap.frame < self.T:
                continue

            key = text  # normalization already done at word level
            cached = self.cache.get(key)
            if cached:
                s.translation = cached
            else:
                # Stub translation; replace with your MT call
                translation = self._dummy_translate(text)
                s.translation = translation
                self.cache.put(key, translation)

    def _dummy_translate(self, en_sentence: str) -> Dict[str, Any]:
        # Placeholder: identity mapping with trivial word alignment
        words = [w for w in en_sentence.split() if w]
        alignment = [{"en": w, "gu": w, "start_idx": i, "end_idx": i} for i, w in enumerate(words)]
        return {
            "gu": en_sentence,  # stubbed
            "alignment": alignment,
            "last_seen": int(time.time()),
            "version": 1,
        }

    # ---------- Simple line clustering ----------

    def _cluster_lines(self, words: List[WordTrack]) -> List[List[WordTrack]]:
        # Group by y-center within tolerance
        lines: List[List[WordTrack]] = []
        tol = 0.6  # fraction of average height
        for w in sorted(words, key=lambda x: (x.last_bbox[1], x.last_bbox[0])):
            wyc = (w.last_bbox[1] + w.last_bbox[3]) / 2.0
            wh = (w.last_bbox[3] - w.last_bbox[1])
            placed = False
            for line in lines:
                # compare with line's median y
                lyc = sum((x.last_bbox[1] + x.last_bbox[3]) / 2.0 for x in line) / len(line)
                lh = sum((x.last_bbox[3] - x.last_bbox[1]) for x in line) / len(line)
                if abs(wyc - lyc) <= tol * ((wh + lh) / 2.0):
                    line.append(w)
                    placed = True
                    break
            if not placed:
                lines.append([w])
        return lines

# ---------------------------
# Demo (optional)
# ---------------------------

if __name__ == "__main__":
    tracker = CanvasTextTracker()
    # Simulate a few frames of a growing word: "t" -> "th" -> "thi" -> "this"
    frames = [
        [OCRToken("t", (100,100,112,120), 0.95)],
        [OCRToken("th", (100,100,122,120), 0.95)],
        [OCRToken("thi", (100,100,130,120), 0.95)],
        [OCRToken("this", (100,100,140,120), 0.95)],
        [OCRToken("this l.c", (100,100,140,120), 0.95)],
    ]
    for i, toks in enumerate(frames):
        tracker.update(i, toks)
    for s in tracker.get_all_sentences():
        print("[Sentence]", s.sentence_id, s.last_stable_text, s.last_stable_frame, s.translation)
