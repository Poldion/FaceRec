"""
Gesichter finden (SCRFD, det_10g) und Gesichtsabdruck berechnen (ArcFace, w600k_r50), beides
InsightFace-Modelle "buffalo_l" ueber ONNX Runtime. Ohne das insightface-Paket: die paar Schritte
(Vorverarbeitung, SCRFD-Dekodierung, Ausrichtung auf 112x112) stehen hier direkt.

Rechenwerk ueber FACEREC_PROVIDERS (Default CPU). Die NPU liefe spaeter ueber den
VitisAIExecutionProvider von AMDs Ryzen-AI-Paket - der Rest der App bleibt dabei gleich.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

# Zielpunkte der 5 Merkmale (Augen, Nase, Mundwinkel) im 112x112-Ausschnitt, wie beim Training von ArcFace
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass
class Face:
    bbox: tuple[float, float, float, float]     # x1, y1, x2, y2 im Originalbild
    score: float                                # Sicherheit der Gesichtserkennung
    embedding: np.ndarray                       # 512 Werte, auf Laenge 1 normiert
    crop: bytes                                 # JPEG-Ausschnitt fuer die Oberflaeche


def decode_image(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def face_crop(img: np.ndarray, bbox, size: int = 192) -> bytes:
    """Quadratischer Ausschnitt mit etwas Rand um das Gesicht."""
    x1, y1, x2, y2 = bbox
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.8
    h, w = img.shape[:2]
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    crop = img[b:d, a:c]
    if crop.size == 0:
        crop = img
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter) <= thresh]
    return keep


class Engine:
    def __init__(self, model_dir: str, providers: list[str], det_size: int = 640,
                 det_thresh: float = 0.5, min_face: int = 40, threads: int = 0,
                 provider_options: list[dict] | None = None):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.det, self.rec = self._sessions(model_dir, opts, providers, provider_options)
        self.det_size, self.det_thresh, self.min_face = det_size, det_thresh, min_face
        self._anchors: dict[tuple, np.ndarray] = {}
        try:
            self._warmup()
        except Exception as e:
            # manche GPU-Fehler zeigen sich erst beim ersten Lauf, nicht beim Laden
            if providers == ["CPUExecutionProvider"]:
                raise
            print(f"Rechenwerk {providers} scheitert beim ersten Lauf ({str(e)[-200:]}) - weiter mit CPU", flush=True)
            self.det, self.rec = self._sessions(model_dir, opts, ["CPUExecutionProvider"], None)
            self._warmup()
        self.providers = self.det.get_providers()

    @staticmethod
    def _sessions(model_dir, opts, providers, provider_options):
        """Sitzungen mit dem gewuenschten Rechenwerk (GPU/NPU); klappt das nicht, CPU statt Absturz."""
        try:
            return (ort.InferenceSession(f"{model_dir}/det_10g.onnx", opts, providers=providers,
                                         provider_options=provider_options),
                    ort.InferenceSession(f"{model_dir}/w600k_r50.onnx", opts, providers=providers,
                                         provider_options=provider_options))
        except Exception as e:
            if providers == ["CPUExecutionProvider"]:
                raise
            print(f"Rechenwerk {providers} nicht nutzbar ({e}) - weiter mit CPU", flush=True)
            cpu = ["CPUExecutionProvider"]
            return (ort.InferenceSession(f"{model_dir}/det_10g.onnx", opts, providers=cpu),
                    ort.InferenceSession(f"{model_dir}/w600k_r50.onnx", opts, providers=cpu))

    def _warmup(self):
        """Erster Aufruf ist deutlich langsamer (Speicher, ggf. GPU-Kompilierung) - nicht erst beim ersten Gast."""
        img = np.zeros((480, 640, 3), np.uint8)
        self.detect(img)
        self.embed(img, ARCFACE_DST * 2)

    # ------------------------------------------------------------------ SCRFD
    def _centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._anchors:
            ys, xs = np.mgrid[:h, :w]
            c = (np.stack([xs, ys], axis=-1) * stride).reshape(-1, 2).astype(np.float32)
            self._anchors[key] = np.repeat(c, 2, axis=0)          # 2 Anker pro Rasterpunkt
        return self._anchors[key]

    def detect(self, img: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """-> [(bbox[4], score, kps[5,2])] im Originalbild, groesste Gesichter zuerst."""
        size = self.det_size
        h, w = img.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.det.run(None, {self.det.get_inputs()[0].name: blob})
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            sc = outs[i].reshape(-1)
            bb = outs[i + 3].reshape(-1, 4) * stride
            kp = outs[i + 6].reshape(-1, 10) * stride
            centers = self._centers(size // stride, size // stride, stride)
            idx = np.where(sc >= self.det_thresh)[0]
            if not idx.size:
                continue
            c = centers[idx]
            boxes.append(np.hstack([c - bb[idx, :2], c + bb[idx, 2:]]))
            kpss.append((np.tile(c, 5) + kp[idx]).reshape(-1, 5, 2))
            scores.append(sc[idx])
        if not boxes:
            return []
        boxes = np.vstack(boxes) / scale
        kpss = np.vstack(kpss) / scale
        scores = np.concatenate(scores)
        keep = _nms(boxes, scores, 0.4)
        res = [(boxes[k], float(scores[k]), kpss[k]) for k in keep
               if min(boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]) >= self.min_face]
        return sorted(res, key=lambda f: -(f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))

    # ------------------------------------------------------------------ ArcFace
    def embed(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)
        aligned = cv2.warpAffine(img, m, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        v = self.rec.run(None, {self.rec.get_inputs()[0].name: blob})[0][0]
        return (v / np.linalg.norm(v)).astype(np.float32)

    def analyze(self, data: bytes) -> list[Face]:
        img = decode_image(data)
        if img is None:
            raise ValueError("Kein lesbares Bild")
        return [Face(tuple(float(x) for x in bbox), score, self.embed(img, kps), face_crop(img, bbox))
                for bbox, score, kps in self.detect(img)]
