"""
Palm recognition – kiosk-mode server.

Camera runs on the SERVER, frames are streamed to all browsers as MJPEG.
WebSocket pushes scan/registration state to all connected clients.

Open from any device on the LAN:  http://<server-ip>:8000

Run:  python server.py
"""

import asyncio
import base64
import contextlib
from collections import deque
import io
import json
import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
import socket
import time
from datetime import datetime, timezone, date as date_cls
from pathlib import Path
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2
# from mediapipe.python.solutions import drawing_utils as mp_drawing
# from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
# Замена старого drawing_utils и landmark_pb2
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarksConnections

import torch
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from compnet.model import compnet, preprocess_palm

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── constants ────────────────────────────────────────────────────────────────
EXTRACTION_FILE_PATH = Path(__file__).parent / "import.txt"
DB_PATH = Path(__file__).parent / "palms.json"
ATTENDANCE_PATH = Path(__file__).parent / "attendance.json"
WEIGHTS_PATH = Path(__file__).parent / "compnet" / "weights.pth"
SAMPLES = 150               # кадров на регистрацию (~10 сек при 15 fps)
TEMPLATES_PER_USER = 5      # 5 шаблонов из 150 кадров = 1 шаблон каждые ~2 сек
LOCK_THRESHOLD = 0.80    # порог косинусной близости: ниже — не свой
# MARGIN = 0.05             # старое значение
MARGIN = 0.08               # минимальный отрыв первого кандидата от второго
MIN_SCAN_FRAMES = 5         # (устарело — заменено на AVG_MIN_FRAMES)
# ── усреднение эмбеддингов при распознавании ──
AVG_MIN_FRAMES = 6          # сколько КАЧЕСТВЕННЫХ кадров усреднить перед локом (обычно 5–10)
AVG_WINDOW = 12             # скользящее окно: усредняем ПОСЛЕДНИЕ N кадров (старые забываем)
# ── контроль качества входного кадра (отбраковка до эмбеддинга) ──
MIN_PALM_SIZE = 80          # мин. длина ладони wrist→middle MCP в px; меньше — далеко/мелко
MAX_TILT = 1.2              # макс. наклон ладони (z-разброс / размер); больше — слишком косо
FRAMES_PER_ATTEMPT = 15     # как часто инкрементится номер «попытки» в UI (~1 сек при 15 fps)
HAND_LOST_FRAMES = 6        # сколько кадров без руки до сброса сканирования
ROI_PAD = 1.3               # v1: размер кропа относительно длины ладони (wrist→middle MCP)
# ── ROI v2: конвенция межпальцевых впадин (как в Tongji/IITD, на которых учили CompNet) ──
ROI2_SCALE = 1.0            # сторона квадрата в долях расстояния A→B
ROI2_OFFSET = 0.10          # отступ верхней грани квадрата вниз от линии A→B (в долях стороны)
ROI2_MIN_DIST = 20          # мин. расстояние A→B в px, ниже — ладонь слишком мелкая
HULL_EXPAND = 1.18          # на сколько раздуть выпуклую оболочку ладони наружу от центра
MASK_DILATE_FRAC = 0.04     # дилатация маски в долях от размера кропа
EMBED_DIM = 512             # размерность embedding-вектора CompNet
CAMERA_INDEX = 0           # индекс веб-камеры (0 = первая)
# Разрешение захвата. Чем крупнее ладонь в пикселях, тем меньше ROI
# приходится растягивать до 128x128. Камера может отдать меньше —
# фактическое разрешение печатается при открытии.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
JPEG_QUALITY = 80           # качество JPEG для MJPEG-стрима (1–100)
TARGET_FPS = 15             # целевой FPS обработки кадров
TARGET_RECT_RATIO = 0.75    # сторона квадрата-прицела = ratio × высоты кадра

# ── MediaPipe ────────────────────────────────────────────────────────────────
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# mp_styles = mp.solutions.drawing_styles
# _hands = mp_hands.Hands(
#     model_complexity=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5,
#     max_num_hands=1,
#     static_image_mode=False,
# )

# 1. Задаем путь к скачанному файлу модели
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

# 2. Настраиваем конфигурацию (полный аналог ваших старых параметров)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # False в static_image_mode означает режим видео/камеры
    min_hand_detection_confidence=0.7,     # min_detection_confidence
    min_hand_presence_confidence=0.5,      # min_tracking_confidence (для детекции)
    min_tracking_confidence=0.5,           # min_tracking_confidence (для трекинга)
    num_hands=1                            # max_num_hands
)

# 3. Создаем объект детектора
_hands = vision.HandLandmarker.create_from_options(options)

# ── CNN: CompNet (palm-print specialist) ─────────────────────────────────────
print(f"Loading CompNet weights from {WEIGHTS_PATH.name}...")
_cnn = compnet(num_classes=600)   # 600 = Tongji training classes; not used at inference
_state = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
_cnn.load_state_dict(_state, strict=True)
_cnn.eval()
print("CompNet ready.")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


# ── ROI + embedding ──────────────────────────────────────────────────────────
def extract_palm_roi(frame: np.ndarray, landmarks):
    """Returns (crop_bgr, palm_mask_uint8) or (None, None).

    palm_mask is 255 inside the convex hull of palm landmarks (slightly
    expanded), 0 outside — so downstream code can drop the background.
    """
    h, w = frame.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    wrist, middle_mcp = pts[0], pts[9]
    palm_size = float(np.linalg.norm(middle_mcp - wrist))
    if palm_size < 20:
        return None, None
    center = (wrist + middle_mcp) * 0.5
    dx, dy = middle_mcp - wrist
    angle = float(np.degrees(np.arctan2(dy, dx))) + 90
    side = int(palm_size * ROI_PAD)
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # Apply the same rotation to all 21 landmarks so we can build a mask in
    # crop-space coordinates.
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])  # (21, 3)
    rotated_pts = (M @ pts_h.T).T                                       # (21, 2)

    half = side // 2
    x1, y1 = int(center[0]) - half, int(center[1]) - half
    x2, y2 = x1 + side, y1 + side
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    crop = rotated[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, None
    if any((pad_l, pad_r, pad_t, pad_b)):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_REPLICATE)

    # Palm hull: wrist + thumb base + finger MCPs. Excludes fingers/background.
    # 0=wrist, 1=thumb CMC, 2=thumb MCP, 5/9/13/17=index/middle/ring/pinky MCP.
    palm_idx = [0, 1, 2, 5, 9, 13, 17]
    crop_pts = rotated_pts[palm_idx] - np.array([x1, y1], dtype=np.float32)
    hull_center = crop_pts.mean(axis=0)
    expanded = hull_center + (crop_pts - hull_center) * HULL_EXPAND
    hull = cv2.convexHull(expanded.astype(np.int32))

    ch, cw = crop.shape[:2]
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # Soft margin: dilation covers palm flesh that extends past landmark joints.
    k = max(3, int(round(min(ch, cw) * MASK_DILATE_FRAC)))
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return crop, mask


# ── ROI v2: по межпальцевым впадинам (конвенция Tongji/IITD) ─────────────────
def _roi_v2_geometry(frame_shape, landmarks):
    """Геометрия ROI v2, или None если ладонь слишком мелкая.

    A = середина между MCP указательного (5) и среднего (9)  — впадина 1
    B = середина между MCP безымянного (13) и мизинца (17)   — впадина 2

    center — середина A→B; angle — поворот кадра, после которого линия A→B
    горизонтальна, а запястье оказывается СНИЗУ; dist = |AB|; side — сторона
    квадрата ROI. Запястье (0) используется только для выбора «низа», на
    масштаб и положение ROI оно не влияет.
    """
    h, w = frame_shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    A = (pts[5] + pts[9]) * 0.5
    B = (pts[13] + pts[17]) * 0.5
    dist = float(np.linalg.norm(B - A))
    if dist < ROI2_MIN_DIST:
        return None

    center = (A + B) * 0.5
    dx, dy = (B - A)
    angle = float(np.degrees(np.arctan2(dy, dx)))   # после поворота A→B горизонтальна

    # Запястье должно оказаться ниже линии — иначе доворачиваем на 180°,
    # чтобы квадрат лёг на ладонь, а не на пальцы.
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    wrist_y = float((M @ np.array([pts[0][0], pts[0][1], 1.0], dtype=np.float32))[1])
    if wrist_y < float(center[1]):
        angle += 180.0

    return {"center": center, "angle": angle, "dist": dist,
            "side": int(round(dist * ROI2_SCALE))}


def _crop_square(img: np.ndarray, x1: int, y1: int, side: int):
    """Квадратный кроп; если вылез за кадр — добираем края репликацией."""
    h, w = img.shape[:2]
    x2, y2 = x1 + side, y1 + side
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return None
    if any((pad_l, pad_r, pad_t, pad_b)):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return crop


def extract_palm_roi_v2(frame: np.ndarray, landmarks):
    """ROI по межпальцевым впадинам — как в базах, на которых учили CompNet.

        5      9      13     17
        *------*      *------*
            A            B
        A -------------- B
                 |
            +---------+
            | ЛАДОНЬ  |   <- это уходит в сеть
            +---------+

    Ориентация, масштаб и положение задаются ТОЛЬКО точками 5/9/13/17
    (v1 брал запястье 0 и средний MCP 9). Возвращает (crop_bgr, None):
    маска не нужна — в квадрат попадает практически только кожа ладони.
    """
    g = _roi_v2_geometry(frame.shape, landmarks)
    if g is None:
        return None, None

    h, w = frame.shape[:2]
    center, side = g["center"], g["side"]
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), g["angle"], 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    x1 = int(round(float(center[0]) - side / 2.0))
    y1 = int(round(float(center[1]) + ROI2_OFFSET * side))
    crop = _crop_square(rotated, x1, y1, side)
    if crop is None:
        return None, None
    return crop, None


def palm_quality(frame_shape, landmarks) -> tuple[bool, str]:
    """Отбраковка кадра ДО эмбеддинга. Возвращает (ok, короткая_причина).

    Плохие кадры (мелкая/за кадром/сильно наклонённая ладонь) не должны
    попадать в усреднение — иначе они портят итоговый вектор и дают
    ложные совпадения. Пороги — вверху файла (MIN_PALM_SIZE / MAX_TILT).
    """
    h, w = frame_shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    palm_size = float(np.linalg.norm(pts[9] - pts[0]))   # wrist(0) → middle MCP(9)

    # 1) слишком маленькая / далеко от камеры
    if palm_size < MIN_PALM_SIZE:
        return False, "ближе"

    # 2) частично за кадром: опорные точки ладони должны быть внутри кадра
    palm_idx = [0, 1, 2, 5, 9, 13, 17]   # запястье + основания пальцев
    m = 2.0
    for i in palm_idx:
        x, y = pts[i]
        if x < m or x > w - m or y < m or y > h - m:
            return False, "в кадр"

    # 3) наклон: разброс глубины z опорных точек относительно размера ладони.
    #    z у MediaPipe примерно в тех же единицах, что x (доля ширины кадра).
    zs = np.array([landmarks[i].z for i in palm_idx], dtype=np.float32)
    palm_norm = palm_size / w
    tilt = float(zs.max() - zs.min()) / (palm_norm + 1e-6)
    if tilt > MAX_TILT:
        return False, "ровнее"

    return True, "ok"


def compute_embedding(roi: np.ndarray, mask: np.ndarray | None = None,
                      debug_dir: str | None = None) -> np.ndarray | None:
    if roi is None or roi.size == 0:
        return None
    x = preprocess_palm(roi, mask=mask, size=128, debug_dir=debug_dir)  # (1, 1, 128, 128)
    with torch.no_grad():
        feat = _cnn.getFeatureCode(x).squeeze(0).numpy()
    # getFeatureCode already L2-normalizes
    return feat.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _entry_templates(entry: dict) -> list[np.ndarray]:
    raw = entry.get("feats") or [entry["feat"]]
    return [np.array(f, dtype=np.float32) for f in raw]


# ── vectorized template cache ────────────────────────────────────────────────
# Stack ALL templates (5 per user × N users) into one big (T, 512) matrix.
# Matching becomes a single BLAS call:  scores = matrix @ feat
# Per-user max is computed via np.maximum.at on indices.
def _rebuild_template_cache() -> None:
    db = state["db"]
    cache = state["tpl_cache"]
    if not db:
        cache["matrix"] = None
        cache["user_idx"] = None
        cache["names"] = []
        cache["display_names"] = []
        return

    names: list[str] = []
    display_names: list[str] = []
    rows: list[list[float]] = []
    indices: list[int] = []
    for u_idx, (key, entry) in enumerate(db.items()):
        names.append(key)
        # new format: {"name": "...", "feats": [...]}; old format: {"feats": [...]}
        display_names.append(entry.get("name", key) if isinstance(entry, dict) else key)
        raw = entry.get("feats") or [entry["feat"]]
        for vec in raw:
            rows.append(vec)
            indices.append(u_idx)

    cache["matrix"] = np.asarray(rows, dtype=np.float32)        # (T, 512)
    cache["user_idx"] = np.asarray(indices, dtype=np.int32)     # (T,)
    cache["names"] = names                                       # length n_users (codes)
    cache["display_names"] = display_names                       # human-readable names
    print(f"[cache] rebuilt: {len(names)} users, {cache['matrix'].shape[0]} templates")


def _vectorized_per_user_max(feat: np.ndarray) -> tuple[list[str], np.ndarray] | None:
    """Returns (names, max_score_per_user). None if DB empty."""
    cache = state["tpl_cache"]
    if cache["matrix"] is None:
        return None
    scores = cache["matrix"] @ feat                             # (T,) BLAS
    n_users = len(cache["names"])
    per_user_max = np.full(n_users, -1.0, dtype=np.float32)
    np.maximum.at(per_user_max, cache["user_idx"], scores)
    return cache["names"], per_user_max


def scan_update(feat: np.ndarray | None, db: dict, hand_present: bool) -> dict | None:
    """Накапливает КАЧЕСТВЕННЫЕ эмбеддинги, усредняет их (mean-pool) и решает
    по одному «чистому» вектору, а не по лучшему случайному кадру.

    feat is None при живой руке = кадр отбракован контролем качества —
    такой кадр в усреднение не идёт, но потерей руки НЕ считается.
    """
    sc = state["scan"]

    # рука пропала из кадра — считаем потерю, при HAND_LOST_FRAMES сбрасываем
    if not hand_present:
        sc["missing"] += 1
        if sc["missing"] >= HAND_LOST_FRAMES:
            _reset_scan()
        return sc["locked"]
    sc["missing"] = 0
    sc["seen_frames"] += 1

    # уже залочено — держим результат, копить дальше не нужно
    if sc["locked"] is not None:
        return sc["locked"]
    if not db:
        return None

    # копим только качественные кадры в СКОЛЬЗЯЩЕЕ окно (deque сам вытесняет
    # старые за maxlen=AVG_WINDOW) — ранние неудачные кадры со временем забываются
    win = sc["emb_window"]
    if feat is not None:
        win.append(feat)

    # ещё нет ни одного качественного кадра — просто сканируем
    if len(win) == 0:
        return {
            "name": None, "code": None, "score": None, "second": None,
            "ok": False, "scanning": True,
            "attempt": sc["seen_frames"] // FRAMES_PER_ATTEMPT + 1,
        }

    # усреднённый эмбеддинг (mean-pool по окну) + ре-нормализация → один вектор
    avg = np.mean(np.stack(win, axis=0), axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-9)

    res = _vectorized_per_user_max(avg)           # сравниваем ОДИН усреднённый вектор
    if res is None:
        return None
    names, scores = res

    # top-2 via argpartition (O(n) — no full sort even at 50K users)
    if len(scores) == 1:
        best_idx, second_peak = 0, -1.0
    else:
        idx2 = np.argpartition(-scores, 1)[:2]
        if scores[idx2[0]] >= scores[idx2[1]]:
            best_idx, second_idx = int(idx2[0]), int(idx2[1])
        else:
            best_idx, second_idx = int(idx2[1]), int(idx2[0])
        second_peak = float(scores[second_idx])
    display = state["tpl_cache"].get("display_names", names)
    best_code = names[best_idx]
    best_display = display[best_idx] if display else best_code
    best_peak = float(scores[best_idx])
    margin_ok = second_peak < 0 or (best_peak - second_peak) >= MARGIN

    # лок только когда в окне достаточно качественных кадров И порог/отрыв ок
    if len(win) >= AVG_MIN_FRAMES and best_peak >= LOCK_THRESHOLD and margin_ok:
        # запас до порога и отрыв от второго кандидата: по ним видно,
        # держится распознавание уверенно или проходит впритык
        print(f"[match] {best_display}: score={best_peak:.4f} "
              f"(порог {LOCK_THRESHOLD}), второй={second_peak:.4f}, "
              f"отрыв={best_peak - second_peak:.4f} (нужен {MARGIN})")
        sc["locked"] = {
            "name": best_display, "code": best_code, "score": round(best_peak, 4),
            "second": round(second_peak, 4) if second_peak > -1 else None,
            "ok": True, "scanning": False,
        }
        return sc["locked"]

    return {
        "name": best_display, "code": best_code, "score": round(best_peak, 4),
        "second": round(second_peak, 4) if second_peak > -1 else None,
        "ok": False, "scanning": True,
        "attempt": sc["seen_frames"] // FRAMES_PER_ATTEMPT + 1,
    }


# ── db ───────────────────────────────────────────────────────────────────────
def load_db() -> dict:
    if not DB_PATH.exists():
        return {}
    try:
        data = json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"DB load error: {e}; starting empty")
        return {}
    if not data:
        return {}
    first = next(iter(data.values()))
    # legacy: list values were 210-d distance vectors
    if isinstance(first, list):
        DB_PATH.write_text("{}", encoding="utf-8")
        print("[!] Old DB format (distance vectors) -> cleared.")
        return {}
    # legacy: MobileNet-era 960-d / 1280-d embeddings — incompatible with CompNet
    feats = first.get("feats") or [first.get("feat")]
    if feats and len(feats[0]) != EMBED_DIM:
        DB_PATH.write_text("{}", encoding="utf-8")
        print(f"[!] Old DB ({len(feats[0])}-d embeddings) -> cleared. "
              f"Re-register palms with CompNet ({EMBED_DIM}-d).")
        return {}
    return data


def save_db(db: dict) -> None:
    DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")


def _users_list(db: dict) -> list[dict]:
    return [
        {"code": k, "name": v.get("name", k) if isinstance(v, dict) else k}
        for k, v in db.items()
    ]


# ── attendance ───────────────────────────────────────────────────────────────
def load_attendance() -> list:
    if not ATTENDANCE_PATH.exists():
        return []
    try:
        data = json.loads(ATTENDANCE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_attendance(att: list) -> None:
    ATTENDANCE_PATH.write_text(
        json.dumps(att, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def record_event(code: str, name: str, kind: str) -> dict:
    """Appends a flat event {date, time, type, code, name} to the log."""
    assert kind in ("in", "out")
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    t = now.strftime("%H:%M:%S")
    state["attendance"].append({"date": day, "time": t, "type": kind, "code": code, "name": name})
    save_attendance(state["attendance"])
    return {"day": day, "code": code, "name": name, "kind": kind, "time": t, "changed": True}


# ── shared state ─────────────────────────────────────────────────────────────
state = {
    "db": load_db(),
    "attendance": load_attendance(),
    "tpl_cache": {"matrix": None, "user_idx": None, "names": []},
    "reg": {"active": False, "name": None, "code": None, "buffer": []},
    "scan": {"emb_window": deque(maxlen=AVG_WINDOW),
             "seen_frames": 0, "missing": 0, "locked": None, "acted": False},
    # attendance intent: None | "in" | "out". When set, the next confident lock
    # records that event for the recognized user and clears the intent.
    "intent": None,
    # last-event toast (broadcast to clients, shown briefly in the UI)
    "last_event": None,         # {name, kind, time, day, ts (monotonic)}
    "latest_jpeg": None,
    "clients": set(),
    "camera_ok": False,
    "camera_info": None,        # параметры камеры, снятые при открытии
    "save_debug": False,
    "last_debug_dir": None,
    "save_testframe": False,     # запрос сырого кадра из /testframe
    "testframe_info": None,      # результат: статистика последнего сырого кадра
    "testframe_stats": None,     # промежуточное: статистика сырого кадра
    "testframe_roi": None,       # промежуточное: стадии ROI (v1/v2) этого кадра
    "testframe_dir": None,       # папка текущего снимка в screenshots/
}


def _reset_scan():
    state["scan"].update(emb_window=deque(maxlen=AVG_WINDOW),
                         seen_frames=0, missing=0, locked=None, acted=False)


# Build cache once on startup
_rebuild_template_cache()


SHOTS_DIR = Path(__file__).parent / "screenshots"


def _new_shot_dir() -> Path:
    """Отдельная папка под каждый снимок: screenshots/ГГГГММДД_ЧЧММСС[_n].

    Раньше стадии перезаписывались в static/ и от снимка оставался только
    последний. Теперь снимки копятся и их можно сравнивать между собой.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = SHOTS_DIR / stamp
    n = 2
    while d.exists():                       # несколько снимков в одну секунду
        d = SHOTS_DIR / f"{stamp}_{n}"
        n += 1
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dump_testframe(frame: np.ndarray) -> None:
    """Сохраняет СЫРОЙ кадр (raw/gray/clahe) в папку снимка и кладёт
    статистику в state['testframe_stats']. Нужно, чтобы понять: камера отдаёт
    монохром (похоже на ИК) или обычную цветную RGB-картинку, и видны ли вены."""
    static = _new_shot_dir()
    state["testframe_dir"] = static
    raw = frame.copy()
    cv2.imwrite(str(static / "01_raw.png"), raw)

    if raw.ndim == 3 and raw.shape[2] == 3:
        channels = 3
        r = raw[..., 2].astype(np.int32)
        g = raw[..., 1].astype(np.int32)
        b = raw[..., 0].astype(np.int32)
        # насколько каналы различаются: ~0 => монохром (сенсор без цвета, ИК-подобно)
        chan_diff = float((np.abs(r - g).mean() + np.abs(g - b).mean()) / 2.0)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        channels = 1
        chan_diff = 0.0
        gray = raw if raw.ndim == 2 else raw[..., 0]

    cv2.imwrite(str(static / "02_gray.png"), gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    cv2.imwrite(str(static / "03_clahe.png"), clahe)

    state["testframe_stats"] = {
        "shape": list(raw.shape),
        "channels": channels,
        "dtype": str(raw.dtype),
        "channel_diff": round(chan_diff, 2),      # ≈0 → монохром/ИК-подобно
        "looks_grayscale": bool(chan_diff < 2.0),
        "brightness_mean": round(float(gray.mean()), 1),
        "brightness_min": int(gray.min()),
        "brightness_max": int(gray.max()),
    }


def _roi_quality_stats(small: np.ndarray | None) -> dict:
    """Замеры по 128x128, т.е. ровно по тому, что получает CompNet.

    Считаем на входе сети, а не на исходном кропе, чтобы числа были
    сравнимы между снимками при разном размере ладони в кадре.
    """
    if small is None or small.size == 0:
        return {}
    g = small.astype(np.float32)
    n = float(g.size)
    return {
        # средняя яркость: ~90..170 нормально, край диапазона — темно/пересвет
        "mean": round(float(g.mean()), 1),
        # разброс яркости: линии ладони видны примерно от 20 и выше
        "contrast": round(float(g.std()), 1),
        # выбитые в ноль / в белое пиксели: там информации уже нет
        "clip_dark_pct": round(float((small == 0).sum()) / n * 100, 1),
        "clip_bright_pct": round(float((small == 255).sum()) / n * 100, 1),
        # резкость: дисперсия лапласиана, чем больше — тем чётче линии
        "focus": round(float(cv2.Laplacian(small, cv2.CV_64F).var()), 1),
        # «мыло или нет» — сравнение картинки с её же размытой копией.
        # Абсолютные меры резкости зависят и от контраста, и от того, сколько
        # в объекте мелкой фактуры: у гладкой ладони её на порядки меньше, чем
        # у шумного поля, поэтому единого эталона не существует. А вот падение
        # лапласиана после дополнительного размытия от фактуры почти не зависит:
        # у резкой картинки есть что терять (отношение заметно больше 1),
        # у уже размытой терять нечего (отношение около 1).
        "sharpness": _blur_ratio(small),
    }


def _blur_ratio(img: np.ndarray) -> float:
    lap = float(cv2.Laplacian(img, cv2.CV_64F).var())
    soft = cv2.GaussianBlur(img, (0, 0), 1.5)
    lap_soft = float(cv2.Laplacian(soft, cv2.CV_64F).var())
    return round(lap / (lap_soft + 1e-6), 2)


def _dump_testframe_roi(frame: np.ndarray, hand) -> None:
    """Стадии ROI того же кадра: v1 (сейчас в работе) и v2 (кандидат).

    Для каждого варианта — сам кроп, картинка 128×128 в том виде, в каком её
    получает CompNet, и итоговый нормализованный тензор. Кадр приходит сюда
    ещё без скелета и оверлея, так что стадии честные.
    """
    static = state["testframe_dir"]      # папка этого снимка, уже создана

    def _save(name: str, img) -> None:
        if img is not None and getattr(img, "size", 0):
            cv2.imwrite(str(static / name), img)

    def _net_input(roi, mask):
        """(128×128 как видит сеть, нормализованный тензор для показа)."""
        if roi is None or roi.size == 0:
            return None, None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        if mask is not None:
            m = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
            small = cv2.bitwise_and(small, small, mask=m)
        # тензор ~N(0,1) — растягиваем в 0..255, иначе глазами не увидеть
        t = preprocess_palm(roi, mask=mask, size=128).squeeze().numpy()
        lo, hi = float(t.min()), float(t.max())
        norm = ((t - lo) / (hi - lo + 1e-6) * 255).clip(0, 255).astype(np.uint8)
        return small, norm

    roi1, mask1 = extract_palm_roi(frame, hand)
    roi2, _ = extract_palm_roi_v2(frame, hand)
    small1, norm1 = _net_input(roi1, mask1)
    small2, norm2 = _net_input(roi2, None)

    _save("04_v1_crop.png", roi1)
    _save("05_v1_mask.png", mask1)
    _save("06_v1_128.png", small1)
    _save("07_v1_norm.png", norm1)
    _save("08_v2_crop.png", roi2)
    _save("09_v2_128.png", small2)
    _save("10_v2_norm.png", norm2)

    h, w = frame.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float32)
    g = _roi_v2_geometry(frame.shape, hand)
    q_ok, q_reason = palm_quality(frame.shape, hand)

    state["testframe_roi"] = {
        "quality": q_reason,
        "quality_ok": bool(q_ok),
        # v1: масштаб от запястье→средний MCP
        "v1_side_px": int(round(float(np.linalg.norm(pts[9] - pts[0])) * ROI_PAD)),
        # v2: масштаб от впадины к впадине
        "v2_side_px": (g["side"] if g else 0),
        "v2_angle_deg": (round(g["angle"], 1) if g else None),
        "v2_valley_dist_px": (round(g["dist"], 1) if g else None),
        # качество именно того участка, который уходит в сеть
        "v2_quality": _roi_quality_stats(small2),
        "v1_quality": _roi_quality_stats(small1),
    }


def _finish_testframe() -> None:
    """Собирает единый результат /testframe (с рукой или без) и гасит флаг."""
    stats = state.get("testframe_stats")
    shot_dir = state.get("testframe_dir")
    if stats is None:
        state["testframe_info"] = {"error": "не удалось снять кадр"}
    else:
        info = {**stats,
                "shot": (shot_dir.name if shot_dir else None),
                "roi": state.get("testframe_roi"),
                "camera": state.get("camera_info")}
        state["testframe_info"] = info
        # те же цифры рядом с картинками — чтобы потом разбирать снимки
        # пачкой, не переснимая и не листая страницу
        if shot_dir is not None:
            try:
                (shot_dir / "info.json").write_text(
                    json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[testframe] не удалось записать info.json: {e}")
    state["save_testframe"] = False


# ── per-frame processing (runs in executor thread) ───────────────────────────
def handle_frame(frame: np.ndarray) -> tuple[bytes | None, dict]:
    frame = cv2.flip(frame, 1)  # mirror for selfie view: hand moves the same way on screen

    # запрос из /testframe — сырые стадии снимаем ДО любой обработки/отрисовки.
    # Флаг здесь НЕ гасим: стадии ROI снимаются ниже, когда найдены landmarks,
    # а итог собирается в конце кадра (_finish_testframe).
    if state.get("save_testframe"):
        state["testframe_stats"] = None
        state["testframe_roi"] = None
        try:
            _dump_testframe(frame)
        except Exception as e:
            print(f"[testframe] raw dump failed: {e}")
            state["save_testframe"] = False
            state["testframe_info"] = {"error": str(e)}

    out: dict = {
        "hand": False, "match": None, "register": None,
        "users": _users_list(state["db"]),
    }

    # 1. Конвертируем кадр OpenCV (BGR) в формат MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # 2. Получаем временную метку кадра в миллисекундах (критично для режима VIDEO)
    # Если у вас есть объект cap (cv2.VideoCapture), лучше использовать: int(cap.get(cv2.CAP_PROP_POS_MSEC))
    # В качестве универсального решения используем системное время:
    # frame_timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
    frame_timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # 3. Запускаем распознавание через новый метод
    result = _hands.detect_for_video(mp_image, frame_timestamp_ms)

    # debug dump request (consumed once)
    debug_dir = None
    if state["save_debug"]:
        debug_dir = str(Path(__file__).parent / "debug" /
                        datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(debug_dir, exist_ok=True)
        # save the un-annotated frame as stage 0 (before any landmarks drawn)
        cv2.imwrite(os.path.join(debug_dir, "0_full_frame.png"), frame)
        state["save_debug"] = False
        state["last_debug_dir"] = debug_dir

    feat = None
    
    # В новом API поле называется 'hand_landmarks' вместо 'multi_hand_landmarks'
    # if result.hand_landmarks:
    #     # ИСПРАВЛЕНО: Берем именно первую найденную руку (индекс [0])
    #     # Теперь hand — это список объектов NormalizedLandmark
    #     hand = result.hand_landmarks[0]  
    #     h_, w_ = frame.shape[:2]
        
    #     # Теперь lm.x и lm.y отработают корректно без ошибок
    #     pts = np.array([[lm.x * w_, lm.y * h_] for lm in hand],
    #                    dtype=np.float32)
    #     out["palm_bbox"] = [float(pts[:, 0].min()), float(pts[:, 1].min()),
    #                         float(pts[:, 0].max()), float(pts[:, 1].max())]
        
    #     # Передаем точки в ваши кастомные функции обработки ладони
    #     roi, palm_mask = extract_palm_roi(frame, hand)
    #     feat = compute_embedding(roi, mask=palm_mask, debug_dir=debug_dir)
        
    #     # 4. Отрисовка точек через новые утилиты Tasks API
    #     mp_drawing.draw_landmarks(
    #         image=frame,                         
    #         landmark_list=hand,                  
    #         connections=HandLandmarksConnections.HAND_CONNECTIONS, 
    #         landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    #         connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
    #                     # Передаем размеры кадра (ширину и высоту) для не-квадратных видеопотоков:
    #         image_dimensions=ImageDimensions(width=w_, height=h_)
    #     )
    #     out["hand"] = True



    # if result.hand_landmarks:
    #         # 1. Объявите индексы связей скелета ладони (за пределами или внутри условия)
    #     HAND_CONNECTIONS = [
    #         (0, 1), (1, 2), (2, 3), (3, 4),       # Большой палец
    #         (0, 5), (5, 6), (6, 7), (7, 8),       # Указательный
    #         (5, 9), (9, 10), (10, 11), (11, 12),  # Средний
    #         (9, 13), (13, 14), (14, 15), (15, 16),# Безымянный
    #         (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Мизинец и ладонь
    #     ]
    #     # result.hand_landmarks[0] — список из 21 точки первой руки
    #     hand = result.hand_landmarks[0]  
    #     h_, w_ = frame.shape[:2]
        
    #     # Расчет bbox и эмбеддингов (ваш оригинальный код)
    #     pts = np.array([[lm.x * w_, lm.y * h_] for lm in hand], dtype=np.float32)
    #     out["palm_bbox"] = [float(pts[:, 0].min()), float(pts[:, 1].min()),
    #                         float(pts[:, 0].max()), float(pts[:, 1].max())]
        
    #     roi, palm_mask = extract_palm_roi(frame, hand)
    #     feat = compute_embedding(roi, mask=palm_mask, debug_dir=debug_dir)
        
    #     # 2. ИСПРАВЛЕНИЕ: Отрисовка на чистом OpenCV вместо mp_drawing.draw_landmarks
    #     # Переводим нормализованные координаты всех 21 точек в физические пиксели кадра
    #     pixel_pts = [(int(lm.x * w_), int(lm.y * h_)) for lm in hand]
        
    #     # Рисуем соединительные линии (зеленый цвет, толщина 2)
    #     for connection in HAND_CONNECTIONS:
    #         start_idx, end_idx = connection
    #         cv2.line(frame, pixel_pts[start_idx], pixel_pts[end_idx], (0, 255, 0), 2)
            
    #     # Рисуем суставы/точки поверх линий (красный цвет, радиус 4)
    #     for pt in pixel_pts:
    #         cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    #     out["hand"] = True
    if result.hand_landmarks:
        # Словарь, где для каждого пальца задан список его связей и цвет в формате BGR
        FINGERS_DATA = {
            "thumb": {  # Большой палец
                "connections": [(0, 1), (1, 2), (2, 3), (3, 4)],
                "color": (255, 0, 255)  # Пурпурный
            },
            "index": {  # Указательный
                "connections": [(0, 5), (5, 6), (6, 7), (7, 8)],
                "color": (255, 0, 0)    # Синий
            },
            "middle": { # Средний
                "connections": [(5, 9), (9, 10), (10, 11), (11, 12)],
                "color": (0, 255, 0)    # Зеленый
            },
            "ring": {   # Безымянный
                "connections": [(9, 13), (13, 14), (14, 15), (15, 16)],
                "color": (0, 255, 255)  # Желтый
            },
            "pinky": {  # Мизинец и основание ладони
                "connections": [(13, 17), (0, 17), (17, 18), (18, 19), (19, 20)],
                "color": (0, 165, 255)  # Оранжевый
            }
        }

        # В новом API Tasks берем список точек для первой руки
        hand = result.hand_landmarks[0] 
        h_, w_ = frame.shape[:2]
        
        # Расчет bbox и эмбеддингов
        pts = np.array([[lm.x * w_, lm.y * h_] for lm in hand], dtype=np.float32)
        out["palm_bbox"] = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                            float(pts[:, 0].max()), float(pts[:, 1].max())]

        # контроль качества: мелкую / за кадром / наклонённую ладонь в
        # усреднение не пускаем (feat=None), но руку считаем присутствующей
        q_ok, q_reason = palm_quality(frame.shape, hand)
        out["hand_quality"] = q_reason
        if q_ok:
            # ROI по межпальцевым впадинам — та же конвенция, что в Tongji/IITD,
            # на которых учили CompNet. Маска не передаётся: в квадрат попадает
            # практически только кожа ладони, а жёсткий контур маски сеть
            # принимала бы за линию ладони.
            roi, _ = extract_palm_roi_v2(frame, hand)
            feat = compute_embedding(roi, mask=None, debug_dir=debug_dir)
        else:
            feat = None

        # запрос из /testframe — кадр здесь ещё без скелета и оверлея,
        # поэтому кропы честно совпадают с тем, что ушло бы в CompNet
        if state.get("save_testframe"):
            try:
                _dump_testframe_roi(frame, hand)
            except Exception as e:
                print(f"[testframe] roi dump failed: {e}")
                state["testframe_roi"] = None

        # Переводим нормализованные координаты всех 21 точек в физические пиксели кадра
        pixel_pts = [(int(lm.x * w_), int(lm.y * h_)) for lm in hand]
        
        # 1. Сначала рисуем линии для каждого пальца своим цветом
        for finger_name, data in FINGERS_DATA.items():
            finger_color = data["color"]
            for connection in data["connections"]:
                start_idx, end_idx = connection
                cv2.line(frame, pixel_pts[start_idx], pixel_pts[end_idx], finger_color, 1)
                
        # 2. Затем рисуем точки (суставы). Чтобы они соответствовали цвету пальца,
        # мы красим их в зависимости от их индекса (ID от 0 до 20)
        for idx, pt in enumerate(pixel_pts):
            # По умолчанию для запястья (ID 0) используем белый цвет
            pt_color = (255, 255, 255) 
            
            # # Определяем, к какому пальцу относится текущая точка
            # if idx in [1, 2, 3, 4]:
            #     pt_color = FINGERS_DATA["thumb"]["color"]
            # elif idx in [5, 6, 7, 8]:
            #     pt_color = FINGERS_DATA["index"]["color"]
            # elif idx in [9, 10, 11, 12]:
            #     pt_color = FINGERS_DATA["middle"]["color"]
            # elif idx in [13, 14, 15, 16]:
            #     pt_color = FINGERS_DATA["ring"]["color"]
            # elif idx in [17, 18, 19, 20]:
            #     pt_color = FINGERS_DATA["pinky"]["color"]
                
            # Рисуем саму точку
            cv2.circle(frame, pt, 4, pt_color, -1)

        out["hand"] = True





    reg = state["reg"]
    db = state["db"]

    if reg["active"]:
        if feat is not None:
            reg["buffer"].append(feat)
        progress = len(reg["buffer"])
        done = progress >= SAMPLES
        out["register"] = {
            "name": reg["name"], "progress": progress,
            "total": SAMPLES, "done": done,
        }
        if done:
            buf = np.stack(reg["buffer"], axis=0)
            step = max(1, len(buf) // TEMPLATES_PER_USER)
            picks = buf[::step][:TEMPLATES_PER_USER]
            picks = picks / (np.linalg.norm(picks, axis=1, keepdims=True) + 1e-9)
            reg_key = str(reg["code"]) if reg.get("code") else reg["name"]
            db[reg_key] = {"name": reg["name"], "feats": picks.astype(np.float32).tolist()}
            save_db(db)
            _rebuild_template_cache()
            out["users"] = _users_list(db)
            # record registration event in attendance log
            _now = datetime.now()
            state["attendance"].append({
                "date": _now.strftime("%Y-%m-%d"),
                "time": _now.strftime("%H:%M:%S"),
                "type": "reg",
                "code": reg_key,
                "name": reg["name"],
            })
            save_attendance(state["attendance"])
            reg.update(active=False, name=None, code=None, buffer=[])
            _reset_scan()
    elif db:
        out["match"] = scan_update(feat, db, out["hand"])

        # ─── intent commit: record check-in/out the FIRST time a lock fires ──
        m = out["match"]
        sc = state["scan"]
        if (m and m.get("ok") and state["intent"] is not None
                and not sc.get("acted")):
            _code = m.get("code", m["name"])
            _name = m["name"]
            _intent = state["intent"]
            _today = datetime.now().strftime("%Y-%m-%d")
            # find last in/out event for this user today
            _last_type = None
            for _e in reversed(state["attendance"]):
                if _e.get("date") == _today and _e.get("code") == _code and _e.get("type") in ("in", "out"):
                    _last_type = _e["type"]
                    break
            # check if action is allowed
            if _intent == "in" and _last_type == "in":
                _block_msg = f"Вы уже пришли — сначала уйдите"
                _blocked = True
            elif _intent == "out" and _last_type != "in":
                _block_msg = f"{('Уже отмечен уход' if _last_type == 'out' else 'Сначала прийдите, потом уходите')}"
                _blocked = True
            else:
                _blocked = False
                _block_msg = None
            sc["acted"] = True
            state["intent"] = None
            if _blocked:
                state["last_event"] = {
                    "code": _code, "name": _name, "kind": _intent,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "day": _today, "changed": False,
                    "blocked": True, "message": _block_msg,
                    "ts": time.monotonic(),
                }
            else:
                ev = record_event(_code, _name, _intent)
                state["last_event"] = {**ev, "ts": time.monotonic()}

    out["intent"] = state["intent"]
    # show event toast for 4 seconds after recording
    le = state["last_event"]
    if le and (time.monotonic() - le["ts"]) < 4.0:
        out["last_event"] = {k: v for k, v in le.items() if k != "ts"}
    _today = datetime.now().strftime("%Y-%m-%d")
    out["attendance_today"] = [e for e in state["attendance"] if e.get("date") == _today]

    # /testframe: кадр обработан целиком — публикуем единый результат
    if state.get("save_testframe"):
        _finish_testframe()

    _draw_overlay(frame, out)

    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return (jpeg.tobytes() if ok else None), out


# ── PIL-based text rendering (supports Cyrillic) ────────────────────────────
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _find_font_path() -> str | None:
    # matplotlib ships DejaVu Sans which has full Cyrillic support
    try:
        import matplotlib
        p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    candidates = [
        # macOS
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

_FONT_PATH: str | None = _find_font_path()


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    if _FONT_PATH:
        try:
            f = ImageFont.truetype(_FONT_PATH, size)
            _FONT_CACHE[size] = f
            return f
        except OSError:
            pass
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _draw_overlay(frame: np.ndarray, out: dict) -> None:
    """All overlays go through PIL so Cyrillic renders correctly."""
    h, w = frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")

    def text_size(s: str, fnt) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), s, font=fnt)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    # ─── target rectangle (corner brackets) ──
    side = int(h * TARGET_RECT_RATIO)
    rx0 = (w - side) // 2
    ry0 = (h - side) // 2
    rx1, ry1 = rx0 + side, ry0 + side

    palm_bbox = out.get("palm_bbox")
    rect_color = (120, 200, 255)        # default — empty
    rect_hint = "Поднесите ладонь в рамку"
    if palm_bbox:
        cx = (palm_bbox[0] + palm_bbox[2]) / 2
        cy = (palm_bbox[1] + palm_bbox[3]) / 2
        palm_w = palm_bbox[2] - palm_bbox[0]
        palm_h = palm_bbox[3] - palm_bbox[1]
        in_rect = rx0 <= cx <= rx1 and ry0 <= cy <= ry1
        # rough size check: palm should fill at least ~40% of rect
        size_ok = max(palm_w, palm_h) >= side * 0.4
        if in_rect and size_ok:
            rect_color = (80, 255, 130)
            rect_hint = None
        elif in_rect and not size_ok:
            rect_color = (255, 200, 60)
            rect_hint = "Ближе к камере"
        else:
            rect_color = (255, 200, 60)
            rect_hint = "Сдвиньте ладонь в центр"

    corner = 36
    thickness = 4
    for (x, y, dx1, dy1, dx2, dy2) in [
        (rx0, ry0,  corner,  0,       0,       corner),     # top-left
        (rx1, ry0, -corner,  0,       0,       corner),     # top-right
        (rx0, ry1,  corner,  0,       0,      -corner),     # bot-left
        (rx1, ry1, -corner,  0,       0,      -corner),     # bot-right
    ]:
        draw.line([(x, y), (x + dx1, y + dy1)], fill=rect_color, width=thickness)
        draw.line([(x, y), (x + dx2, y + dy2)], fill=rect_color, width=thickness)

    if rect_hint:
        fnt = _font(18)
        tw, th = text_size(rect_hint, fnt)
        tx, ty = (w - tw) // 2, ry1 + 8
        draw.rectangle([tx - 8, ty - 4, tx + tw + 8, ty + th + 4], fill=(0, 0, 0))
        draw.text((tx, ty), rect_hint, fill=rect_color, font=fnt)

    def banner(text: str, color_rgb: tuple[int, int, int]):
        fnt = _font(22)
        tw, th = text_size(text, fnt)
        draw.rectangle([0, 0, w, th + 20], fill=(0, 0, 0))
        draw.text(((w - tw) // 2, 8), text, fill=color_rgb, font=fnt)

    def toast(text: str, color_rgb: tuple[int, int, int]):
        fnt = _font(26)
        tw, th = text_size(text, fnt)
        x0, y0 = (w - tw) // 2 - 14, h // 2 - th // 2 - 10
        x1, y1 = (w + tw) // 2 + 14, h // 2 + th // 2 + 10
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 220))
        draw.text(((w - tw) // 2, h // 2 - th // 2), text, fill=color_rgb, font=fnt)

    def bottom(text: str, color_rgb: tuple[int, int, int], sub: str | None = None):
        fnt = _font(20)
        sub_fnt = _font(14)
        tw, th = text_size(text, fnt)
        sw, sh = (0, 0) if not sub else text_size(sub, sub_fnt)
        pad = 8
        box_w = max(tw, sw) + 2 * pad
        y_bottom = h - 10
        y_top = y_bottom - th - (sh + 4 if sub else 0) - 2 * pad
        draw.rectangle([4, y_top, 4 + box_w, y_bottom], fill=(0, 0, 0))
        if sub:
            draw.text((4 + pad, y_top + pad), sub, fill=(200, 200, 200), font=sub_fnt)
            draw.text((4 + pad, y_top + pad + sh + 4), text, fill=color_rgb, font=fnt)
        else:
            draw.text((4 + pad, y_top + pad), text, fill=color_rgb, font=fnt)

    # ─── intent banner ──
    intent = out.get("intent")
    # if intent == "in":
    #     banner("ПРИХОД — поднесите ладонь", (100, 255, 0))
    # elif intent == "out":
    #     banner("УХОД — поднесите ладонь", (255, 90, 90))

    # ─── recording toast ──
    le = out.get("last_event")
    if le:
        if le.get("blocked"):
            toast(le.get("message", "Действие заблокировано"), (255, 150, 60))
        # Успешное распознавание больше не подтверждаем плашкой на кадре: об
        # этом говорит полноэкранное приветствие /greeting/. Плашка рисуется
        # мгновенно, а приветствие проявляется ~250 мс — из-за этого зазора
        # зелёный прямоугольник с ФИО успевал мелькнуть перед приветствием.
        # elif le.get("changed", True):
        #     kind_label = "приход" if le["kind"] == "in" else "уход"
        #     toast(f"{le['name']} — {kind_label}  {le['time']}", (120, 255, 120))
        # else:
        #     kind_label = "приход" if le["kind"] == "in" else "уход"
        #     toast(f"{le['name']} — {kind_label} уже был в {le['time']}", (255, 200, 60))

    # ─── bottom status ──
    r = out["register"]
    m = out["match"]
    # if r is not None:
    #     if r["done"]:
    #         bottom(f"Сохранён: {r['name']}", (100, 255, 100))
    #     else:
    #         bottom(f"Запись {r['name']}: {r['progress']}/{r['total']}",
    #                (255, 200, 60),
    #                sub="Медленно подвигайте рукой ближе/дальше, чуть наклоняйте")
    # elif m is not None:
    #     score_pct = m["score"] * 100
    #     if m.get("scanning"):
    #         attempt = m.get("attempt", 1)
    #         bottom(f"Сканирую #{attempt}…  {score_pct:.0f}%", (255, 200, 60))
    #     elif m["ok"]:
    #         bottom(f"MATCH: {m['name']}  {score_pct:.0f}%", (100, 255, 100))
    #     else:
    #         bottom(f"Неизвестно  {score_pct:.0f}%", (255, 100, 100))
    # elif not out["hand"]:
    #     bottom("Нет руки", (140, 140, 140))
    # else:
    #     bottom("Рука найдена — БД пуста", (180, 180, 180))

    # write back into the frame buffer
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── camera loop ──────────────────────────────────────────────────────────────
def _probe_camera(cap: cv2.VideoCapture, w: int, h: int) -> None:
    """Снимает параметры камеры один раз при открытии, кладёт в state.

    Драйверы отдают -1 или 0 для того, чего не умеют, поэтому значения
    показываем как есть — важен сам факт, поддерживает камера фокус или нет.
    """
    props = {
        "fps": cv2.CAP_PROP_FPS,
        "autofocus": cv2.CAP_PROP_AUTOFOCUS,
        "focus": cv2.CAP_PROP_FOCUS,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "gain": cv2.CAP_PROP_GAIN,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "sharpness": cv2.CAP_PROP_SHARPNESS,
    }
    info = {"width": w, "height": h}
    for name, prop in props.items():
        try:
            info[name] = round(float(cap.get(prop)), 3)
        except Exception:
            info[name] = None
    state["camera_info"] = info
    print("[camera] параметры: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    if info.get("autofocus", -1) in (-1, 0):
        print("[camera] автофокус недоступен или выключен — если ладонь мылит, "
              "камера скорее всего не умеет фокусироваться так близко")


def _try_open_camera(index: int) -> cv2.VideoCapture | None:
    """Try DSHOW first (Windows-friendly), then default backend."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            # MJPG до выставления размера: без него DSHOW отдаёт HD на ~10 fps
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"[camera] запрошено {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}, "
                      f"камера отдаёт {w}x{h}")
                if w < CAPTURE_WIDTH:
                    print("[camera] камера не поддерживает запрошенное разрешение — "
                          "ладонь в кадре будет мельче, ROI придётся растягивать")
                _probe_camera(cap, w, h)
                return cap
            cap.release()
    return None


def _open_any_camera() -> tuple[cv2.VideoCapture | None, int | None]:
    """Try CAMERA_INDEX first, then 0..3 as fallback. Returns (cap, index)."""
    candidates = [CAMERA_INDEX] + [i for i in range(4) if i != CAMERA_INDEX]
    for idx in candidates:
        cap = _try_open_camera(idx)
        if cap is not None:
            return cap, idx
    return None, None


async def camera_loop():
    loop = asyncio.get_event_loop()

    cap, idx = await loop.run_in_executor(None, _open_any_camera)
    if cap is not None:
        state["camera_ok"] = True
        print(f"Camera #{idx} opened.")
    else:
        print("[!] Cannot open any camera. Possible causes:")
        print("    • another app is using it (Zoom, Teams, browser tab with getUserMedia)")
        print("    • Windows camera privacy disabled for desktop apps")
        print("    • no camera connected")
        print("    (will keep retrying — plug the camera in and it will recover)")

    period = 1.0 / TARGET_FPS
    frame_count = 0
    fail_streak = 0
    try:
        while True:
            t0 = loop.time()

            # (re)connect if we currently have no working camera
            if cap is None:
                cap, idx = await loop.run_in_executor(None, _open_any_camera)
                if cap is None:
                    state["camera_ok"] = False
                    await asyncio.sleep(1.0)          # retry once per second
                    continue
                state["camera_ok"] = True
                fail_streak = 0
                print(f"Camera #{idx} reconnected.")

            try:
                ok, frame = await loop.run_in_executor(None, cap.read)
                if not ok or frame is None:
                    fail_streak += 1
                    # ~1s of dead frames => camera likely unplugged, drop & reopen
                    if fail_streak >= max(TARGET_FPS, 5):
                        print("[camera] frames stopped — releasing to reconnect...")
                        state["camera_ok"] = False
                        await loop.run_in_executor(None, cap.release)
                        cap = None
                        fail_streak = 0
                    else:
                        await asyncio.sleep(0.05)
                    continue

                fail_streak = 0
                jpeg, status = await loop.run_in_executor(None, handle_frame, frame)
                if jpeg is not None:
                    state["latest_jpeg"] = jpeg
                    frame_count += 1
                    if frame_count in (1, 10, 100):
                        print(f"[camera] produced frame #{frame_count}, "
                              f"jpeg={len(jpeg)} bytes, clients={len(state['clients'])}")

                for ws in list(state["clients"]):
                    try:
                        await ws.send_text(json.dumps(status))
                    except Exception:
                        state["clients"].discard(ws)
            except Exception as e:
                import traceback
                print(f"[camera] frame error: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)

            dt = loop.time() - t0
            if dt < period:
                await asyncio.sleep(period - dt)
    except asyncio.CancelledError:
        pass
    finally:
        if cap is not None:
            cap.release()
        print("[camera] released")



# Схема JSON-ответа (пользователь)
class UserResponse(BaseModel):
    code: int = Field(serialization_alias="code")
    name: str = Field(serialization_alias="name")
    profile_code: int = Field(serialization_alias="profile-code")
    position_fio: Optional[str] = Field(default=None, serialization_alias="position-fio")
    password: Optional[str] = Field(default=None, serialization_alias="password")
    user_card: Optional[str] = Field(default=None, serialization_alias="user-card")
    inn: Optional[str] = Field(default=None, serialization_alias="inn")


# Класс парсинга (читает файл с диска в кодировке Windows-1251)
class Extract:
    def __init__(self):
        self.users: List[UserResponse] = []

    def process_file(self, file_path: str):
        logger.info(f"extraction.ProcessFile: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="cp1251") as f:
            content = f.read()

        ss = [line.rstrip("\r") for line in content.split("\n")]
        if not ss or ss == [""]:
            raise ValueError("empty extraction")

        for i, uu in enumerate(ss):
            if uu == "$$$DELETEALLUSERS":
                self.users = []
            elif uu == "$$$ADDUSERS":
                self.add_users(ss[i + 1 :])
            elif uu == "$$$DELETEUSERSBYCODE":
                self.del_user_by_code(ss[i + 1 :])

    def del_user_by_code(self, ss: List[str]):
        for code in ss:
            if code.startswith("$$$"):
                return
            if not code.strip():
                continue
            try:
                cc = int(code)
            except ValueError as e:
                raise ValueError(f"wrong field Code({code}) in DelUser: {e}")
            self.users = [u for u in self.users if u.code != cc]

    def add_users(self, ss: List[str]):
        for i, uu in enumerate(ss):
            if not uu.strip():
                continue
            
            data = uu.split(";")
            if len(data) > 1:
                if data[0].startswith("$$$"):
                    return
                try:
                    code = int(data[0])
                except ValueError as e:
                    raise ValueError(f"wrong field Code({data[0]}) in User[{i+1}]: {e}")
            else:
                if data[0].startswith("$$$") or data[0] == "":
                    return
                continue

            try:
                p_code = int(data[3]) if len(data) > 3 and data[3] else 0
            except ValueError as e:
                raise ValueError(f"wrong field Profile Code({data[3]}) in User[{i+1}]: {e}")

            user = UserResponse(
                code=code,
                name=data[1] if len(data) > 1 else "",
                profile_code=p_code,
                position_fio=data[2] if len(data) > 2 and data[2] else None,
                password=data[4] if len(data) > 4 and data[4] else None,
                user_card=data[5] if len(data) > 5 and data[5] else None,
                inn=data[6] if len(data) > 6 and data[6] else None
            )
            self.users.append(user)


# ── FastAPI ──────────────────────────────────────────────────────────────────
@contextlib.asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(camera_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

# serve /assets/* built by Vite
_assets = STATIC / "assets"
if _assets.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

# Приветственный экран (приход и уход — одна страница, режим задаётся ?kind=).
# Отдельная папка, а не /assets: пересборка фронта чистит assets и снесла бы
# картинки со шрифтами.
_greeting = STATIC / "greeting"
if _greeting.exists():
    app.mount("/greeting", StaticFiles(directory=str(_greeting), html=True), name="greeting")


@app.get("/")
async def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/theme-light.css")
async def theme_light():
    """Светлая палитра дашборда — переопределяет тёмную тему из assets/index-*.css.
    Отдельным файлом, а не внутри assets: пересборка фронта чистит assets."""
    return Response((STATIC / "theme-light.css").read_text(encoding="utf-8"),
                    media_type="text/css")


@app.get("/api/testframe/grab")
async def testframe_grab():
    """Снимает один кадр и отдаёт ВСЕ его стадии: сырой кадр и обработки."""
    state["testframe_info"] = None
    state["save_testframe"] = True
    for _ in range(80):                       # ждём обработки кадра до ~4 сек
        if state["testframe_info"] is not None:
            break
        await asyncio.sleep(0.05)
    info = state["testframe_info"]
    if info is None:
        state["save_testframe"] = False
        return {"ok": False, "error": "камера не отдаёт кадры (не открыта?)"}
    if info.get("error"):
        return {"ok": False, "error": info["error"]}

    shot_dir = state.get("testframe_dir")

    def b64(name: str) -> str:
        if shot_dir is None:
            return ""
        p = shot_dir / name
        return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

    return {
        "ok": True,
        "info": info,
        # сырой кадр и его общие представления
        "raw": b64("01_raw.png"),
        "gray": b64("02_gray.png"),
        "clahe": b64("03_clahe.png"),
        # ROI v1 — то, что сейчас реально уходит в CompNet
        "v1_crop": b64("04_v1_crop.png"),
        "v1_mask": b64("05_v1_mask.png"),
        "v1_128": b64("06_v1_128.png"),
        "v1_norm": b64("07_v1_norm.png"),
        # ROI v2 — кандидат по межпальцевым впадинам
        "v2_crop": b64("08_v2_crop.png"),
        "v2_128": b64("09_v2_128.png"),
        "v2_norm": b64("10_v2_norm.png"),
    }


@app.get("/testframe")
async def testframe():
    """Одна страница: онлайн-видео + все стадии обработки снятого кадра."""
    html = """<!doctype html><meta charset="utf-8">
<body style="font-family:sans-serif;background:#111;color:#eee;padding:16px;line-height:1.5">
  <h2>Тестовый кадр — все стадии</h2>
  <div>Наведи ладонь и жми «Снимок». Чтобы проверить стабильность кропа,
       сними одну и ту же руку ~10 раз и сравни колонку
       <b>128×128</b> у v2 — она должна повторяться.</div>
  <img src="/stream" style="max-width:420px;border:1px solid #444;border-radius:8px;margin:8px 0">
  <div>
    <button onclick="grab()" style="font-size:18px;padding:10px 22px;border-radius:999px;
      border:0;background:#2d7;cursor:pointer;font-weight:700">Снимок</button>
    <button onclick="clearAll()" style="font-size:14px;padding:10px 18px;border-radius:999px;
      border:0;background:#555;color:#eee;cursor:pointer;margin-left:8px">Очистить</button>
    <span id="status" style="margin-left:12px;color:#9cf"></span>
    <span id="count" style="margin-left:12px;color:#888"></span>
  </div>
  <div id="shots" style="margin-top:16px"></div>
  <script>
    let n = 0;
    async function grab() {
      const s = document.getElementById('status');
      s.textContent = 'снимаю...';
      try {
        const r = await fetch('/api/testframe/grab');
        const d = await r.json();
        if (!d.ok) { s.textContent = 'ошибка: ' + (d.error || ''); return; }
        s.textContent = '';
        n += 1;
        document.getElementById('count').textContent = 'снимков: ' + n;
        document.getElementById('shots').prepend(render(d, n));
      } catch (e) { s.textContent = 'ошибка: ' + e; }
    }
    function clearAll() {
      n = 0;
      document.getElementById('shots').innerHTML = '';
      document.getElementById('count').textContent = '';
    }
    function card(title, b64, wide) {
      if (!b64) return '';
      const w = wide ? 300 : 150;
      return '<div><div style="font-size:12px;color:#aaa">' + title + '</div>' +
        '<img src="data:image/png;base64,' + b64 + '" style="width:' + w +
        'px;image-rendering:pixelated;border:1px solid #444"></div>';
    }
    function quality(q) {
      if (!q || q.mean === undefined) return '';
      const bad = [];
      if (q.contrast < 20) bad.push('низкий контраст — линии ладони почти не видны');
      if (q.sharpness < 2.0) bad.push('размыто — не в фокусе или движение');
      if (q.clip_bright_pct > 1) bad.push('пересвет ' + q.clip_bright_pct + '% — лампа бьёт в ладонь');
      if (q.clip_dark_pct > 1) bad.push('провал в чёрное ' + q.clip_dark_pct + '%');
      if (q.mean < 70) bad.push('темно');
      if (q.mean > 190) bad.push('слишком светло');
      const verdict = bad.length
        ? '<span style="color:#e77">' + bad.join('; ') + '</span>'
        : '<span style="color:#7d7">картинка для сети нормальная</span>';
      return '<div style="color:#888;font-size:13px">качество ROI: ' +
        'яркость ' + q.mean + ' &middot; контраст ' + q.contrast +
        ' &middot; резкость x' + q.sharpness + ' (лапласиан ' + q.focus + ')' +
        ' &middot; пересвет ' + q.clip_bright_pct + '%' +
        ' &middot; чёрное ' + q.clip_dark_pct + '% &rarr; ' + verdict + '</div>';
    }
    function row(label, cards) {
      const body = cards.filter(Boolean).join('');
      if (!body) return '';
      return '<div style="margin:10px 0">' +
        '<div style="color:#9cf;font-weight:700;margin-bottom:4px">' + label + '</div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start">' +
        body + '</div></div>';
    }
    function render(d, idx) {
      const i = d.info, roi = i.roi;
      const box = document.createElement('div');
      box.style.cssText = 'border-bottom:1px solid #333;padding-bottom:16px;margin-bottom:16px';
      const verdict = i.looks_grayscale
        ? 'похоже на МОНОХРОМ / ИК' : 'похоже на обычную ЦВЕТНУЮ RGB-картинку';
      let geom;
      if (!roi) {
        geom = '<div style="color:#e77">рука в кадре не найдена - стадии ROI недоступны</div>';
      } else {
        const warn = (roi.v2_side_px && roi.v2_side_px < 128)
          ? ' <span style="color:#e77">(меньше 128: кроп растягивается, держи руку ближе)</span>'
          : '';
        geom = '<div style="color:#888;font-size:13px">' +
          'качество кадра: <b>' + roi.quality + '</b> &middot; ' +
          'v1 сторона ' + roi.v1_side_px + 'px &middot; ' +
          'v2 сторона ' + roi.v2_side_px + 'px' + warn + ' &middot; ' +
          'v2 угол ' + roi.v2_angle_deg + '&deg; &middot; ' +
          'впадины ' + roi.v2_valley_dist_px + 'px</div>' + quality(roi.v2_quality);
      }
      box.innerHTML =
        '<h3 style="margin:0">Снимок #' + idx + ' - ' + verdict + '</h3>' +
        '<div style="color:#777;font-size:12px">папка: screenshots/' +
        (i.shot || '?') + '</div>' + geom +
        row('1. Кадр с камеры', [
          card('RAW (как пришёл)', d.raw, true),
          card('GRAY', d.gray, true),
          card('CLAHE - только диагностика ИК, в сеть НЕ идёт', d.clahe, true)]) +
        row('2. ROI v1 - сейчас в работе (запястье, средний MCP)', [
          card('кроп', d.v1_crop),
          card('маска ладони', d.v1_mask),
          card('128x128 - вход сети', d.v1_128),
          card('после нормализации', d.v1_norm)]) +
        row('3. ROI v2 - кандидат (по межпальцевым впадинам)', [
          card('кроп', d.v2_crop),
          card('128x128 - вход сети', d.v2_128),
          card('после нормализации', d.v2_norm)]) +
        '<details style="margin-top:8px"><summary style="cursor:pointer;color:#888">' +
        'статистика кадра (JSON)</summary>' +
        '<pre style="background:#000;padding:10px;border-radius:8px;overflow:auto">' +
        JSON.stringify(i, null, 2) + '</pre></details>';
      return box;
    }
  </script>
</body>"""
    return HTMLResponse(html)


@app.get("/high-five.html")
async def high_five():
    return HTMLResponse((STATIC / "high-five.html").read_text(encoding="utf-8"))


@app.get("/heart.html")
async def heart():
    return HTMLResponse((STATIC / "heart.html").read_text(encoding="utf-8"))

# Эндпоинт FastAPI (принимает пустой POST-запрос и парсит файл из константы)
@app.get("/parse-extraction", response_model=List[UserResponse], response_model_exclude_none=True)
async def parse_extraction():
    try:
        extractor = Extract()
        # Вызов метода парсинга с использованием константы
        extractor.process_file(EXTRACTION_FILE_PATH)
        
        # FastAPI автоматически превратит список моделей в JSON,
        # применит алиасы (profile-code) и скроет пустые поля (exclude_none=True)
        return extractor.users

    except FileNotFoundError as fnf:
        logger.error(f"File error: {fnf}")
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.get("/api/users")
async def get_users():
    return {"users": _users_list(state["db"])}


@app.delete("/api/users/{name}")
async def delete_user(name: str):
    state["db"].pop(name, None)
    save_db(state["db"])
    _rebuild_template_cache()
    _reset_scan()
    return {"users": _users_list(state["db"])}


@app.post("/api/register/{code}")
async def start_register(code: str, name: str):
    state["reg"].update(active=True, name=name, code=code, buffer=[])
    return {"status": "started", "code": code, "name": name}


@app.post("/api/register/cancel")
async def cancel_register():
    state["reg"].update(active=False, name=None, buffer=[])
    return {"status": "cancelled"}


@app.post("/api/debug/save")
async def debug_save():
    """Dump the next processed frame's pipeline stages to debug/<timestamp>/."""
    state["save_debug"] = True
    return {"status": "queued"}


# ─── attendance ──────────────────────────────────────────────────────────────
@app.post("/api/attendance/mode/{kind}")
async def set_intent(kind: str):
    if kind not in ("in", "out"):
        return {"error": "kind must be 'in' or 'out'"}
    state["intent"] = kind
    _reset_scan()           # restart scan so the next palm registers fresh
    return {"intent": kind}


@app.post("/api/attendance/cancel")
async def cancel_intent():
    state["intent"] = None
    return {"intent": None}


@app.get("/api/attendance/today")
async def attendance_today():
    day = datetime.now().strftime("%Y-%m-%d")
    return [e for e in state["attendance"] if e.get("date") == day]


@app.get("/api/attendance/all")
async def attendance_all():
    return state["attendance"]



def _build_xlsx() -> bytes:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance"

    headers = ["Дата", "Время", "Тип", "Пользователь", "Код"]
    ws.append(headers)
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1E1E2E")
    for col_idx in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=col_idx)
        c.font = head_font
        c.fill = head_fill
        c.alignment = Alignment(horizontal="center")

    type_labels = {"reg": "Регистрация", "in": "Приход", "out": "Уход"}
    for ev in sorted(state["attendance"], key=lambda e: (e.get("date", ""), e.get("time", ""))):
        ws.append([
            ev.get("date", ""),
            ev.get("time", ""),
            type_labels.get(ev.get("type", ""), ev.get("type", "")),
            ev.get("name", ""),
            ev.get("code", ""),
        ])

    # column widths
    widths = [12, 20, 12, 12, 14]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i)].width = w

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@app.get("/api/attendance/export.xlsx")
async def export_xlsx():
    data = _build_xlsx()
    fname = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/stream")
async def stream():
    async def gen():
        boundary = b"--frame\r\n"
        while True:
            jpeg = state.get("latest_jpeg")
            if jpeg:
                yield (boundary
                       + b"Content-Type: image/jpeg\r\n"
                       + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            await asyncio.sleep(1.0 / TARGET_FPS)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    state["clients"].add(websocket)
    try:
        # initial snapshot
        await websocket.send_text(json.dumps({
            "hand": False, "match": None, "register": None,
            "users": _users_list(state["db"]),
        }))
        while True:
            await websocket.receive_text()  # ignore pings/messages
    except WebSocketDisconnect:
        pass
    finally:
        state["clients"].discard(websocket)


def _list_ipv4() -> list[str]:
    """List all IPv4 addresses on this machine, skipping loopback.
    Falls back to `ipconfig` on Windows if getaddrinfo returns too few."""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import re, subprocess
            out = subprocess.check_output(["ipconfig"], text=True,
                                          encoding="utf-8", errors="ignore", timeout=3)
            ips.update(re.findall(r"IPv4.*?:\s*(\d+\.\d+\.\d+\.\d+)", out))
        except Exception:
            pass
    ips.discard("127.0.0.1")
    return sorted(ips)


if __name__ == "__main__":
    print("\n  Local:  http://localhost:8000")
    addrs = _list_ipv4()
    if addrs:
        print("  LAN candidates (pick the one matching your Wi-Fi/Ethernet subnet):")
        for ip in addrs:
            print(f"          http://{ip}:8000")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning",
                timeout_graceful_shutdown=3)
