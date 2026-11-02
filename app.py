"""
Gesichtserkennung fuer die Tuerklingel (Home Assistant).

Ablauf: Home Assistant schickt beim Klingeln das Standbild (oder nennt die Bild-Entitaet, dann holt
der Dienst es selbst) an POST /api/recognize. Jedes Gesicht wird mit den bekannten Personen
verglichen. Bekannte Gesichter werden gemeldet und nur FACEREC_RECENT_HOURS (48) lang aufgehoben,
damit sich Verwechslungen korrigieren lassen ("falsch erkannt"); danach weg. Unbekannte landen in einer
Liste und lassen sich einer Person zuweisen (per Push-Knopf aus Home Assistant oder in der
Oberflaeche). Zugewiesene Gesichter verbessern die Erkennung dieser Person. Aehnliche unbekannte
Gesichter werden gruppiert, damit "dieselbe Person nochmal" in einem Rutsch zugewiesen wird.

Daten (FACEREC_DATA): facerec.db (SQLite), faces/ (Gesichtsausschnitte), snapshots/ (Standbilder
nur von Klingel-Ereignissen mit unbekannten Gesichtern, weg sobald alle zugewiesen/verworfen).
"""
import asyncio
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager, closing

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

import i18n

import tracker

__version__ = "0.3.0"          # SemVer; einzige Stelle (Oberflaeche, /api/settings, /proxy-module)

DATA = os.environ.get("FACEREC_DATA", "./data")
MODELS = os.environ.get("FACEREC_MODELS", "./models")
PROVIDERS = [p.strip() for p in os.environ.get("FACEREC_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]
# Optionen je Rechenwerk als JSON-Liste (gleiche Reihenfolge wie FACEREC_PROVIDERS), z.B. [{"device_id": "0"}, {}]
PROVIDER_OPTIONS = json.loads(os.environ["FACEREC_PROVIDER_OPTIONS"]) if os.environ.get("FACEREC_PROVIDER_OPTIONS") else None
DET_SIZE = int(os.environ.get("FACEREC_DET_SIZE", "640"))       # Eingangsgroesse der Gesichtssuche (480 = schneller)
THREADS = int(os.environ.get("FACEREC_THREADS", "0"))           # CPU-Threads je Modell, 0 = automatisch
MATCH = float(os.environ.get("FACEREC_MATCH", "0.35"))        # ab hier gilt ein Gesicht als erkannt
SUGGEST = float(os.environ.get("FACEREC_SUGGEST", "0.28"))    # ab hier wird eine Person vorgeschlagen
TOKEN = os.environ.get("FACEREC_TOKEN", "")                   # leer = ohne Anmeldung (nur im LAN!)
HA_URL = os.environ.get("HA_URL", "").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
KEEP_EVENTS = int(os.environ.get("FACEREC_KEEP_EVENTS", "300"))
RECENT_HOURS = float(os.environ.get("FACEREC_RECENT_HOURS", "48"))   # so lange sind Erkennungen korrigierbar
MAX_UPLOAD = 20 * 1024 * 1024
PORT = int(os.environ.get("FACEREC_PORT", "8191"))
LANG = "en" if os.environ.get("FACEREC_LANG", "de").lower().startswith("en") else "de"  # Sprache der Ansage
LANG_HEADER = "x-facerec-lang"                                # Sprache der Oberfläche fuer Server-Meldungen
MAX_FPS = float(os.environ.get("FACEREC_STREAM_FPS", "12"))   # Stream-Scan: hoechstens so viele Bilder/s auswerten
# Stream-Scan: Kamera-Entitaet -> eigene MJPEG-Adresse (z.B. ESP32 direkt, ohne Umweg ueber Home Assistant),
# Komma-getrennt: "camera.flur=http://192.168.1.50:8080/,camera.garten=http://..."
STREAM_URLS = {k.strip(): v.strip() for k, v in (
    item.split("=", 1) for item in os.environ.get("FACEREC_STREAM_URLS", "").split(",") if "=" in item)}
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FACES = os.path.join(DATA, "faces")
SNAPS = os.path.join(DATA, "snapshots")
ENTITY_RE = re.compile(r"^(camera|image)\.[a-z0-9_]+$")
FILE_RE = re.compile(r"^[a-z0-9_]+\.jpg$")

ENGINE = None               # engine.Engine; Tests setzen hier ein nachgebautes ein
LOCK = asyncio.Lock()       # Schreibzugriffe nacheinander (Gruppierung, Zuweisen)


# ------------------------------------------------------------------ Datenbank
def db() -> sqlite3.Connection:
    con = sqlite3.connect(os.path.join(DATA, "facerec.db"), timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db():
    os.makedirs(FACES, exist_ok=True)
    os.makedirs(SNAPS, exist_ok=True)
    with closing(db()) as con, con:
        con.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
            embedding BLOB NOT NULL, crop TEXT, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
            source TEXT, snapshot TEXT, text TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS unknowns (id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER REFERENCES events(id) ON DELETE SET NULL,
            embedding BLOB NOT NULL, crop TEXT, det_score REAL, created REAL NOT NULL,
            group_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS recent (id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER REFERENCES events(id) ON DELETE SET NULL,
            embedding BLOB NOT NULL, crop TEXT, person_id INTEGER, score REAL, sample_id INTEGER,
            created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS samples_person ON samples(person_id);
        CREATE INDEX IF NOT EXISTS unknowns_group ON unknowns(group_id);
        """)


def vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def best_person(con, emb: np.ndarray) -> tuple[int | None, str | None, float, int | None]:
    """Aehnlichste Person (Kosinus, bestes Einzelbild der Person) -> (id, Name, Wert, Lernbild-id)."""
    rows = con.execute("SELECT s.id, s.person_id, p.name, s.embedding FROM samples s "
                       "JOIN persons p ON p.id = s.person_id").fetchall()
    if not rows:
        return None, None, 0.0, None
    sims = np.stack([vec(r["embedding"]) for r in rows]) @ emb
    i = int(sims.argmax())
    return rows[i]["person_id"], rows[i]["name"], float(sims[i]), rows[i]["id"]


def save_jpeg(folder: str, prefix: str, data: bytes) -> str:
    name = f"{prefix}_{int(time.time())}_{secrets.token_hex(6)}.jpg"     # unratbar: Links ohne Anmeldung
    with open(os.path.join(folder, name), "wb") as f:
        f.write(data)
    return name


def remove_file(folder: str, name: str | None):
    if name:
        try:
            os.remove(os.path.join(folder, name))
        except FileNotFoundError:
            pass


def drop_unused_snapshots(con):
    """Standbilder nur behalten, solange zu ihrem Ereignis noch unbekannte Gesichter offen sind."""
    for r in con.execute("SELECT id, snapshot FROM events WHERE snapshot IS NOT NULL AND id NOT IN "
                         "(SELECT event_id FROM unknowns WHERE event_id IS NOT NULL)").fetchall():
        remove_file(SNAPS, r["snapshot"])
        con.execute("UPDATE events SET snapshot = NULL WHERE id = ?", (r["id"],))


def say(names: list[str], unknown: int, lang: str | None = None) -> str:
    """Satzteil fuer Ansage/Push: 'Anna und eine unbekannte Person' (Sprache: FACEREC_LANG)."""
    en = (lang or LANG) == "en"
    words = ({1: "an", 2: "two", 3: "three", 4: "four"} if en else {1: "eine", 2: "zwei", 3: "drei", 4: "vier"})
    parts = list(names)
    if unknown:
        if en:
            parts.append(f"{words.get(unknown, unknown)} unknown " + ("people" if unknown > 1 else "person"))
        else:
            parts.append(f"{words.get(unknown, unknown)} unbekannte Person" + ("en" if unknown > 1 else ""))
    if not parts:
        return "no one recognisable" if en else "niemand erkennbar"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + (" and " if en else " und ") + parts[-1]


def sentence(names: list[str], unknown: int, lang: str | None = None) -> str:
    """Ganzer Satz fuer Push/Ansage: 'Anna und Bob stehen vor der Tuer.' / 'Anna and Bob are at the door.'"""
    en = (lang or LANG) == "en"
    if not names and not unknown:
        return "Someone is at the door, no face visible." if en else "Es klingelt, kein Gesicht erkennbar."
    text = say(names, unknown, lang)
    many = len(names) + unknown > 1
    if en:
        return text[0].upper() + text[1:] + (" are" if many else " is") + " at the door."
    return text[0].upper() + text[1:] + (" stehen" if many else " steht") + " vor der Tür."


def purge_recent(con, now: float):
    """Erkannte Gesichter nach RECENT_HOURS endgueltig loeschen (Ausschnitt + Abdruck)."""
    for r in con.execute("SELECT id, crop FROM recent WHERE created < ?", (now - RECENT_HOURS * 3600,)).fetchall():
        remove_file(FACES, r["crop"])
        con.execute("DELETE FROM recent WHERE id = ?", (r["id"],))


def rewrite_event(con, event_id: int | None, rid: int, face: dict) -> dict | None:
    """Ergebnis eines Klingel-Ereignisses nach einer Korrektur neu zusammensetzen (Namen, Satz)."""
    row = con.execute("SELECT result FROM events WHERE id = ?", (event_id,)).fetchone() if event_id else None
    if not row or not row["result"]:
        return None
    res = json.loads(row["result"])
    for f in res["faces"]:
        if f.get("recent_id") == rid:
            f.pop("recent_id", None)
            f.update(face, corrected=True)
    names = []
    for f in res["faces"]:
        if f.get("known") and f["name"] not in names:
            names.append(f["name"])
    unknown = sum(1 for f in res["faces"] if not f.get("known"))
    res.update(names=names, unknown_count=unknown, known=bool(names) and not unknown,
               text=say(names, unknown), sentence=sentence(names, unknown))
    con.execute("UPDATE events SET text = ?, result = ? WHERE id = ?", (res["text"], json.dumps(res), event_id))
    return res


def media(kind: str, name: str | None) -> str | None:
    return f"/media/{kind}/{name}" if name else None


# ------------------------------------------------------------------ App
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    init_db()
    if ENGINE is None:
        from engine import Engine
        if os.environ.get("ORT_MIGRAPHX_MODEL_CACHE_PATH"):
            # MIGraphX legt den Cache-Ordner nicht selbst an; ohne ihn scheitert schon der erste GPU-Lauf
            os.makedirs(os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"], exist_ok=True)
        ENGINE = await run_in_threadpool(lambda: Engine(MODELS, PROVIDERS, det_size=DET_SIZE, threads=THREADS,
                                                        provider_options=PROVIDER_OPTIONS))
        print(f"Modelle geladen, Rechenwerk: {ENGINE.providers}", flush=True)
        print("Hinweis: InsightFace buffalo_l ist nur fuer nicht-kommerzielle Nutzung freigegeben "
              "(https://github.com/deepinsight/insightface).", flush=True)
    task = asyncio.create_task(_purge_loop())
    yield
    task.cancel()


async def _purge_loop():
    while True:
        try:
            async with LOCK:
                with closing(db()) as con, con:
                    purge_recent(con, time.time())
        except Exception as e:      # Aufraeumen darf den Dienst nie stoppen
            print(f"Aufraeumen fehlgeschlagen: {e}", flush=True)
        await asyncio.sleep(3600)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    # Meldungen entstehen auf Deutsch; Englisch, wenn die Oberflaeche (Kopfzeile) oder der Client es wuenscht
    detail = i18n.translate(exc.detail, i18n.request_lang(request.headers, LANG_HEADER))
    return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    # Fremde Webseiten duerfen nichts ausloesen (Home Assistant schickt keinen Origin-Kopf)
    if request.method not in ("GET", "HEAD") and (origin := request.headers.get("origin")):
        if origin.split("://", 1)[-1] != request.headers.get("host", ""):
            return JSONResponse({"detail": i18n.translate("Fremder Ursprung", i18n.request_lang(request.headers, LANG_HEADER))},
                                status_code=403)
    if TOKEN and path.startswith("/api/"):
        auth = request.headers.get("authorization", "")
        given = auth[7:] if auth.lower().startswith("bearer ") else request.cookies.get("facerec_token", "")
        if not secrets.compare_digest(given.encode(), TOKEN.encode()):
            return JSONResponse({"detail": i18n.translate("Anmeldung noetig", i18n.request_lang(request.headers, LANG_HEADER))},
                                status_code=401)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if not path.startswith("/media/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/health")
async def health():
    return {"ok": True, "providers": getattr(ENGINE, "providers", None)}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/i18n.js")
async def i18n_js():
    return Response(i18n.script("facerec.lang", "X-FaceRec-Lang"), media_type="text/javascript; charset=utf-8")


@app.get("/media/{kind}/{name}")
async def media_file(kind: str, name: str):
    folder = {"face": FACES, "snap": SNAPS}.get(kind)
    if not folder or not FILE_RE.match(name) or not os.path.isfile(os.path.join(folder, name)):
        raise HTTPException(404)
    return FileResponse(os.path.join(folder, name), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


async def fetch_from_ha(entity: str) -> bytes:
    if not ENTITY_RE.match(entity):
        raise HTTPException(400, "entity_id muss camera.xxx oder image.xxx sein")
    if not HA_URL or not HA_TOKEN:
        raise HTTPException(400, "HA_URL/HA_TOKEN nicht gesetzt - Bild bitte direkt mitschicken")
    kind = "image_proxy" if entity.startswith("image.") else "camera_proxy"
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{HA_URL}/api/{kind}/{entity}", headers={"Authorization": f"Bearer {HA_TOKEN}"})
    if r.status_code != 200 or not r.content:
        raise HTTPException(502, f"Home Assistant lieferte kein Bild ({r.status_code})")
    return r.content


@app.post("/api/recognize")
async def recognize(request: Request):
    """Bild als Formularfeld 'image' oder Rohdaten (image/jpeg), oder JSON {"entity_id": "image.xxx"}."""
    ctype = request.headers.get("content-type", "")
    source = "upload"
    if ctype.startswith("multipart/"):
        form = await request.form()
        up = form.get("image")
        if up is None or isinstance(up, str):
            raise HTTPException(400, "Feld 'image' fehlt")
        data = await up.read(MAX_UPLOAD + 1)
    elif ctype.startswith("application/json"):
        body = await request.json()
        source = str(body.get("entity_id", ""))
        data = await fetch_from_ha(source)
    else:
        data = await request.body()
    if not data:
        raise HTTPException(400, "Kein Bild")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Bild zu gross")
    try:
        faces = await run_in_threadpool(ENGINE.analyze, data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    async with LOCK:
        with closing(db()) as con, con:
            now = time.time()
            purge_recent(con, now)
            ev = con.execute("INSERT INTO events (created, source) VALUES (?, ?)", (now, source)).lastrowid
            out, names, unknown = [], [], 0
            for f in faces:
                pid, name, score, sid = best_person(con, f.embedding)
                item = {"bbox": [round(x) for x in f.bbox], "det_score": round(f.score, 3), "score": round(score, 3)}
                if pid is not None and score >= MATCH:
                    # kurz aufheben, falls es eine Verwechslung war (siehe /api/recent/{id}/correct)
                    crop = save_jpeg(FACES, "r", f.crop)
                    rid = con.execute("INSERT INTO recent (event_id, embedding, crop, person_id, score, sample_id, created) "
                                      "VALUES (?, ?, ?, ?, ?, ?, ?)", (ev, f.embedding.tobytes(), crop, pid, score,
                                                                       sid, now)).lastrowid
                    item.update(known=True, person_id=pid, name=name, recent_id=rid, crop=media("face", crop))
                    if name not in names:
                        names.append(name)
                else:
                    unknown += 1
                    crop = save_jpeg(FACES, "u", f.crop)
                    group = _group_for(con, f.embedding)
                    uid = con.execute("INSERT INTO unknowns (event_id, embedding, crop, det_score, created, group_id) "
                                      "VALUES (?, ?, ?, ?, ?, ?)", (ev, f.embedding.tobytes(), crop, f.score, now,
                                                                    group or 0)).lastrowid
                    if not group:
                        con.execute("UPDATE unknowns SET group_id = id WHERE id = ?", (uid,))
                    item.update(known=False, unknown_id=uid, crop=media("face", crop),
                                suggestion=name if pid is not None and score >= SUGGEST else None)
                out.append(item)
            snapshot = save_jpeg(SNAPS, "s", data_as_jpeg(data)) if unknown else None
            text = say(names, unknown)
            result = {"event_id": ev, "faces": out, "names": names, "known": bool(names) and not unknown,
                      "unknown_count": unknown, "face_count": len(out), "text": text,
                      "sentence": sentence(names, unknown),
                      "snapshot": media("snap", snapshot)}
            con.execute("UPDATE events SET snapshot = ?, text = ?, result = ? WHERE id = ?",
                        (snapshot, text, json.dumps(result), ev))
            _trim_events(con)
    return result


def data_as_jpeg(data: bytes) -> bytes:
    """Standbild fuer die Oberflaeche als JPEG ablegen (Eingang kann PNG/WebP sein)."""
    if data[:3] == b"\xff\xd8\xff":
        return data
    import cv2
    from engine import decode_image
    return cv2.imencode(".jpg", decode_image(data), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def _group_for(con, emb: np.ndarray) -> int | None:
    """Gruppe des aehnlichsten offenen unbekannten Gesichts (falls aehnlich genug)."""
    rows = con.execute("SELECT group_id, embedding FROM unknowns").fetchall()
    if not rows:
        return None
    sims = np.stack([vec(r["embedding"]) for r in rows]) @ emb
    i = int(sims.argmax())
    return rows[i]["group_id"] if sims[i] >= MATCH else None


def _trim_events(con):
    """Grenze je Quelle: Eine oft ausloesende Flurkamera verdraengt nicht den Verlauf der Klingel."""
    old = [r for (s,) in con.execute("SELECT DISTINCT source FROM events").fetchall() for r in con.execute(
        "SELECT id FROM events WHERE source IS ? ORDER BY id DESC LIMIT -1 OFFSET ?", (s, KEEP_EVENTS)).fetchall()]
    for r in old:
        if not con.execute("SELECT 1 FROM unknowns WHERE event_id = ? UNION SELECT 1 FROM recent WHERE event_id = ?",
                           (r["id"], r["id"])).fetchone():
            con.execute("DELETE FROM events WHERE id = ?", (r["id"],))


# ------------------------------------------------------------------ Unbekannte


# ------------------------------------------------------------------ Stream Scanner (On-Demand Scharfschaltung)
STREAM_SCANS: dict[str, asyncio.Task] = {}   # laufender Stream-Scan je Kamera


async def fire_ha_event(event_type: str, data: dict):
    """Event an Home Assistant senden."""
    if not HA_URL or not HA_TOKEN:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(
                f"{HA_URL}/api/events/{event_type}",
                headers={"Authorization": f"Bearer {HA_TOKEN}"},
                json=data,
            )
    except Exception as e:
        print(f"HA-Event {event_type} fehlgeschlagen: {e}", flush=True)


async def fire_stream_event(event_type: str, legacy_type: str, data: dict):
    """Ereignis des Stream-Scans; der alte Name (flur_*) wird fuer bestehende Automationen weiter mitgeschickt."""
    await fire_ha_event(event_type, data)
    await fire_ha_event(legacy_type, data)


@app.post("/api/stream_scan")
async def start_stream_scan(request: Request):
    """Startet On-Demand-Streaming-Erkennung fuer <duration> Sekunden."""
    body = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            pass
    duration = float(body.get("duration", 15.0))
    entity_id = str(body.get("entity_id") or next(iter(STREAM_URLS), ""))
    stream_url = body.get("stream_url", None)

    # Ein neuer Scan ersetzt nur den laufenden derselben Kamera
    running = STREAM_SCANS.get(entity_id)
    if running and not running.done():
        running.cancel()
    STREAM_SCANS[entity_id] = asyncio.create_task(_run_stream_scan(entity_id, stream_url, duration))
    return {"status": "started", "duration": duration, "entity_id": entity_id}


# Abdruecke aller Lernbilder im Speicher (statt pro Gesicht und Bild aus SQLite); neu geladen, wenn sich
# Lernbilder oder Personen aendern
_SAMPLES = {"key": None, "mat": None, "pids": [], "names": [], "sids": []}


def person_matches(con, emb: np.ndarray) -> dict:
    """Bester Treffer je Person: {Name: tracker.Match} (bestes Lernbild der Person)."""
    key = tuple(con.execute(
        "SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(SUM(person_id * id), 0), "
        "(SELECT group_concat(id || ':' || name) FROM persons) FROM samples").fetchone())
    if key != _SAMPLES["key"]:
        rows = con.execute("SELECT s.id, s.person_id, p.name, s.embedding FROM samples s "
                           "JOIN persons p ON p.id = s.person_id").fetchall()
        _SAMPLES.update(key=key, mat=np.stack([vec(r["embedding"]) for r in rows]) if rows else None,
                        pids=[r["person_id"] for r in rows], names=[r["name"] for r in rows],
                        sids=[r["id"] for r in rows])
    if _SAMPLES["mat"] is None:
        return {}
    sims = _SAMPLES["mat"] @ emb
    out = {}
    for i in np.argsort(-sims):
        name = _SAMPLES["names"][i]
        if name not in out:
            out[name] = tracker.Match(_SAMPLES["pids"][i], float(sims[i]), _SAMPLES["sids"][i])
    return out


def _analyze_frame(jpeg: bytes, scene: "tracker.SceneTracker", now: float, stats: dict) -> list:
    """Ein Bild auswerten (laeuft im Thread): Gesichter finden, Spuren zuordnen, nur fuer offene Spuren
    Abdruecke berechnen. Gibt die Spuren zurueck, die jetzt angesagt werden sollen."""
    from engine import decode_image, face_crop
    img = decode_image(jpeg)
    if img is None:
        return []
    t0 = time.perf_counter()
    dets = ENGINE.detect(img)
    stats["det_ms"] += (time.perf_counter() - t0) * 1000
    stats["frames"] += 1
    if not dets:
        return []
    bboxes = [tuple(float(x) for x in d[0]) for d in dets]
    tracks = scene.associate(bboxes, now)
    items = []
    with closing(db()) as con:
        for (bbox, det_score, kps), bb, track in zip(dets, bboxes, tracks):
            if not track.needs_evidence():
                continue
            t1 = time.perf_counter()
            emb = ENGINE.embed(img, kps)
            stats["emb_ms"] += (time.perf_counter() - t1) * 1000
            stats["embeds"] += 1
            sample = tracker.Sample(tracker.quality(det_score, bb), emb, face_crop(img, bb), float(det_score), bb, jpeg)
            matches = person_matches(con, emb)
            items.append((track, sample, matches))
            best = max(matches.items(), key=lambda kv: kv[1].score) if matches else (None, None)
            print(f"Stream-Scan Spur {track.id}: {best[0]} {best[1].score if best[1] else 0:.3f} "
                  f"det={det_score:.2f} {int(bb[2] - bb[0])}x{int(bb[3] - bb[1])}", flush=True)
    decided = scene.add_evidence(items, now) if items else []
    for t in decided:
        print(f"Stream-Scan Spur {t.id} entschieden: {t.decision.kind} {t.decision.name or ''} "
              f"({t.decision.score:.3f}, {t.quality_frames} gute Bilder)", flush=True)
    return scene.to_announce(decided, now)


async def _report_tracks(tracks: list, entity_id: str, batch: dict):
    """Neu entschiedene Personen sofort melden (HA-Ereignis 'facerec_stream_person' je Person), damit
    Home Assistant gleich mit Namen gruessen kann. Speichert Erkennung/Unbekannte und lernt."""
    if not tracks:
        return
    items = []
    async with LOCK:
        with closing(db()) as con, con:
            now = time.time()
            if batch.get("event_id") is None:
                purge_recent(con, now)
                batch.update(event_id=con.execute("INSERT INTO events (created, source) VALUES (?, ?)",
                                                  (now, entity_id)).lastrowid,
                             faces=[], names=[], unknown=0, snap=None)
            ev = batch["event_id"]
            for t in tracks:
                d, s = t.decision, t.decision.sample
                item = {"bbox": [round(x) for x in s.bbox], "det_score": round(s.det_score, 3),
                        "score": round(d.score, 3), "track": t.id}
                if d.kind == "known":
                    crop = save_jpeg(FACES, "r", s.crop)
                    rid = con.execute("INSERT INTO recent (event_id, embedding, crop, person_id, score, sample_id, created) "
                                      "VALUES (?, ?, ?, ?, ?, ?, ?)", (ev, s.embedding.tobytes(), crop, d.person_id,
                                                                       d.score, d.sample_id, now)).lastrowid
                    item.update(known=True, person_id=d.person_id, name=d.name, recent_id=rid, crop=media("face", crop))
                    batch["names"].append(d.name)
                    if learn_from_stream(con, t, now):
                        item["learned"] = True
                else:
                    crop = save_jpeg(FACES, "u", s.crop)
                    group = _group_for(con, s.embedding)
                    uid = con.execute("INSERT INTO unknowns (event_id, embedding, crop, det_score, created, group_id) "
                                      "VALUES (?, ?, ?, ?, ?, ?)", (ev, s.embedding.tobytes(), crop, s.det_score, now,
                                                                    group or 0)).lastrowid
                    if not group:
                        con.execute("UPDATE unknowns SET group_id = id WHERE id = ?", (uid,))
                    item.update(known=False, unknown_id=uid, crop=media("face", crop),
                                suggestion=d.name if d.kind == "suggest" else None)
                    if d.kind == "suggest":
                        batch["names"].append(d.name)       # wie bisher: Vorschlag wird mit Namen begruesst
                    else:
                        batch["unknown"] += 1
                    batch["snap"] = batch["snap"] or s.jpeg
                batch["faces"].append(item)
                items.append(item)
    for item in items:
        name = item.get("name") or item.get("suggestion")
        print(f"Stream-Scan sofort gemeldet: {name or 'unbekannt'}", flush=True)
        await fire_stream_event("facerec_stream_person", "flur_person_erkannt", {
            "names": [name] if name else [], "unknown": not name, "unknown_count": 0 if name else 1,
            "faces": [item], "source": entity_id, "event_id": batch["event_id"], "single": True})


async def _finish_batch(entity_id: str, batch: dict):
    """Sammelfenster abschliessen: Ereignis fertig speichern und 'facerec_stream_complete' mit allen senden
    (Home Assistant spricht dann einen gemeinsamen Kommentar)."""
    if batch.get("event_id") is None:
        return
    ev, faces, names, unknown = batch["event_id"], batch["faces"], batch["names"], batch["unknown"]
    async with LOCK:
        with closing(db()) as con, con:
            snapshot = save_jpeg(SNAPS, "s", data_as_jpeg(batch["snap"])) if batch["snap"] else None
            text = say(names, unknown)
            result = {"event_id": ev, "faces": faces, "names": names, "known": bool(names) and not unknown,
                      "unknown_count": unknown, "face_count": len(faces), "text": text,
                      "sentence": sentence(names, unknown), "snapshot": media("snap", snapshot)}
            con.execute("UPDATE events SET snapshot = ?, text = ?, result = ? WHERE id = ?",
                        (snapshot, text, json.dumps(result), ev))
            _trim_events(con)
    batch.clear()
    print(f"Stream-Scan Sammelfenster abgeschlossen: {names} unbekannt={unknown}", flush=True)
    await fire_stream_event("facerec_stream_complete", "flur_personen_komplett", {
        "names": names, "unknown": unknown > 0, "unknown_count": unknown,
        "faces": faces, "source": entity_id, "event_id": ev})


# Lernen aus der Flurkamera: Die Lernbilder stammen meist von der Klingel (andere Kamera, anderer Winkel).
# Sehr sicher erkannte Flur-Gesichter werden deshalb als weiteres Lernbild gespeichert - aber nur, wenn
# eindeutig (Abstand zur naechsten Person) und neu genug (nicht fast gleich wie ein vorhandenes Lernbild).
LEARN_MIN_SCORE = float(os.environ.get("FACEREC_LEARN_MIN", "0.45"))
LEARN_MARGIN = 0.15
LEARN_MAX_SIMILAR = 0.75
LEARN_MAX_SAMPLES = 40


def learn_from_stream(con, track, now: float) -> bool:
    d = track.decision
    scores = track.evidence.get(d.name, [])
    if d.kind != "known" or LEARN_MIN_SCORE <= 0 or d.score < LEARN_MIN_SCORE or sum(x >= MATCH for x in scores) < 2:
        return False
    others = [max(v) for n, v in track.evidence.items() if n != d.name]
    if others and d.score - max(others) < LEARN_MARGIN:
        return False
    rows = con.execute("SELECT embedding FROM samples WHERE person_id = ?", (d.person_id,)).fetchall()
    if len(rows) >= LEARN_MAX_SAMPLES:
        return False
    emb = d.sample.embedding
    if rows and float(np.max(np.stack([vec(r["embedding"]) for r in rows]) @ emb)) >= LEARN_MAX_SIMILAR:
        return False
    crop = save_jpeg(FACES, "s", d.sample.crop)
    con.execute("INSERT INTO samples (person_id, embedding, crop, created) VALUES (?, ?, ?, ?)",
                (d.person_id, emb.tobytes(), crop, now))
    print(f"Stream-Scan: Flurbild als Lernbild fuer {d.name} gespeichert ({d.score:.3f})", flush=True)
    return True


def _split_frames(buffer: bytes, boundary: bytes) -> tuple[list[bytes], bytes]:
    """Vollstaendige JPEGs aus einem MJPEG-Puffer (multipart, getrennt durch boundary) -> (Bilder, Rest)."""
    frames = []
    while True:
        b1 = buffer.find(boundary)
        if b1 == -1:
            return frames, buffer[-len(boundary):]
        b2 = buffer.find(boundary, b1 + len(boundary))
        if b2 == -1:
            return frames, buffer[b1:]
        part, buffer = buffer[b1:b2], buffer[b2:]
        s, e = part.find(b"\xff\xd8"), part.rfind(b"\xff\xd9")
        if s != -1 and e > s:
            frames.append(part[s:e + 2])


async def _read_frames(url: str, headers: dict, duration: float, latest: dict, new_frame: asyncio.Event):
    """Liest den MJPEG-Stream und behaelt nur das neueste Bild (aeltere werden verworfen, damit die
    Auswertung nie hinter der Wirklichkeit herlaeuft)."""
    end = time.monotonic() + duration
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=duration + 10)) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    print(f"Stream-Verbindung fehlgeschlagen ({response.status_code}) zu {url}", flush=True)
                    return
                ctype = response.headers.get("content-type", "")
                boundary = b"--123456789000000000000987654321"
                if "boundary=" in ctype:
                    boundary = b"--" + ctype.split("boundary=")[1].split(";")[0].strip().strip('"').encode("ascii")
                buffer = b""
                async for chunk in response.aiter_bytes():
                    frames, buffer = _split_frames(buffer + chunk, boundary)
                    if frames:
                        latest["received"] += len(frames)
                        latest["jpeg"], latest["seq"] = frames[-1], latest["seq"] + len(frames)
                        new_frame.set()
                    if time.monotonic() > end:
                        break
    except Exception as e:
        print(f"Stream-Scan Lesefehler: {e}", flush=True)
    finally:
        latest["done"] = True
        new_frame.set()


async def _run_stream_scan(entity_id: str, stream_url: str | None, duration: float):
    headers = {}
    if not stream_url:
        if entity_id in STREAM_URLS:
            stream_url = STREAM_URLS[entity_id]
        elif entity_id.startswith("camera.") and HA_URL and HA_TOKEN:
            stream_url = f"{HA_URL}/api/camera_proxy_stream/{entity_id}"
            headers["Authorization"] = f"Bearer {HA_TOKEN}"
        else:
            print(f"Stream-Scan: keine Adresse fuer {entity_id} (FACEREC_STREAM_URLS oder HA_URL/HA_TOKEN setzen)",
                  flush=True)
            return
    print(f"Stream-Scan gestartet fuer {entity_id} ({duration}s) an {stream_url}", flush=True)

    scene = tracker.SceneTracker(MATCH, SUGGEST)
    latest = {"jpeg": None, "seq": 0, "received": 0, "done": False}
    new_frame = asyncio.Event()
    stats = {"frames": 0, "det_ms": 0.0, "embeds": 0, "emb_ms": 0.0}
    reader = asyncio.create_task(_read_frames(stream_url, headers, duration, latest, new_frame))
    batch = {}                  # laufendes Sammelfenster: event_id, faces, names, unknown, snap
    last_seq, last_run = 0, 0.0
    try:
        while True:
            try:
                await asyncio.wait_for(new_frame.wait(), timeout=0.3)
            except asyncio.TimeoutError:
                pass
            new_frame.clear()
            now = time.monotonic()
            if scene.batch_ready(now):
                scene.take_batch()
                await _finish_batch(entity_id, batch)
            if latest["seq"] == last_seq:
                if latest["done"]:
                    break
                continue
            if now - last_run < 1 / MAX_FPS:
                await asyncio.sleep(1 / MAX_FPS - (now - last_run))
            last_seq, last_run = latest["seq"], time.monotonic()
            try:
                new = await run_in_threadpool(_analyze_frame, latest["jpeg"], scene, time.monotonic(), stats)
                await _report_tracks(new, entity_id, batch)
            except Exception as e:
                print(f"Stream-Scan Auswertungsfehler: {e}", flush=True)
    except asyncio.CancelledError:
        pass
    finally:
        reader.cancel()
        if scene.batch or batch:
            try:
                scene.take_batch()
                await _finish_batch(entity_id, batch)
            except Exception as e:
                print(f"Fehler beim Abschliessen des Sammelfensters: {e}", flush=True)
        n = max(stats["frames"], 1)
        print(f"Stream-Scan beendet: {latest['received']} Bilder empfangen, {stats['frames']} ausgewertet, "
              f"Gesichtssuche {stats['det_ms'] / n:.0f} ms/Bild, Abdruck "
              f"{stats['emb_ms'] / max(stats['embeds'], 1):.0f} ms/Gesicht ({stats['embeds']}x), "
              f"{len(scene.announced_names)} Personen begruesst", flush=True)


@app.get("/api/unknowns")
async def unknowns():
    """Offene unbekannte Gesichter, nach Gruppe (vermutlich gleiche Person) zusammengefasst."""
    with closing(db()) as con:
        rows = con.execute("SELECT u.*, e.snapshot FROM unknowns u LEFT JOIN events e ON e.id = u.event_id "
                           "ORDER BY u.created DESC").fetchall()
        groups: dict[int, dict] = {}
        for r in rows:
            g = groups.setdefault(r["group_id"], {"group_id": r["group_id"], "faces": []})
            g["faces"].append({"id": r["id"], "crop": media("face", r["crop"]), "snapshot": media("snap", r["snapshot"]),
                               "created": r["created"], "event_id": r["event_id"]})
        for g in groups.values():
            emb = np.mean([vec(r["embedding"]) for r in rows if r["group_id"] == g["group_id"]], axis=0)
            pid, name, score, _ = best_person(con, emb / np.linalg.norm(emb))
            g["suggestion"] = name if pid is not None and score >= SUGGEST else None
            g["suggestion_score"] = round(score, 3) if g["suggestion"] else None
    return list(groups.values())


def _targets(con, uid: int, group: bool) -> list[sqlite3.Row]:
    row = con.execute("SELECT * FROM unknowns WHERE id = ?", (uid,)).fetchone()
    if not row:
        raise HTTPException(404, "Unbekanntes Gesicht nicht (mehr) vorhanden")
    if not group:
        return [row]
    return con.execute("SELECT * FROM unknowns WHERE group_id = ?", (row["group_id"],)).fetchall()


def _person_id(con, name: str) -> tuple[int, bool]:
    name = " ".join(name.split())
    if not name or len(name) > 60:
        raise HTTPException(400, "Name fehlt oder ist zu lang")
    row = con.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()   # NOCASE: 'anna' = 'Anna'
    if row:
        return row["id"], False
    return con.execute("INSERT INTO persons (name, created) VALUES (?, ?)", (name, time.time())).lastrowid, True


@app.post("/api/unknowns/{uid}/assign")
async def assign(uid: int, request: Request):
    """{"name": "Anna"} (vorhandene Person, sonst neu) oder {"person_id": 3}; "group": false nur dieses Gesicht."""
    body = await request.json()
    async with LOCK:
        with closing(db()) as con, con:
            rows = _targets(con, uid, body.get("group", True))
            if body.get("person_id") is not None:
                p = con.execute("SELECT id FROM persons WHERE id = ?", (int(body["person_id"]),)).fetchone()
                if not p:
                    raise HTTPException(404, "Person nicht gefunden")
                pid, created = p["id"], False
            else:
                pid, created = _person_id(con, str(body.get("name", "")))
            for r in rows:
                con.execute("INSERT INTO samples (person_id, embedding, crop, created) VALUES (?, ?, ?, ?)",
                            (pid, r["embedding"], r["crop"], time.time()))
                con.execute("DELETE FROM unknowns WHERE id = ?", (r["id"],))
            drop_unused_snapshots(con)
            name = con.execute("SELECT name FROM persons WHERE id = ?", (pid,)).fetchone()["name"]
    return {"person_id": pid, "name": name, "created": created, "assigned": len(rows)}


@app.post("/api/unknowns/{uid}/ignore")
async def ignore(uid: int, request: Request):
    """Verwerfen (Fremde, Paketbote ...). Standard: die ganze Gruppe."""
    body = await request.json() if await request.body() else {}
    async with LOCK:
        with closing(db()) as con, con:
            rows = _targets(con, uid, body.get("group", True))
            for r in rows:
                con.execute("DELETE FROM unknowns WHERE id = ?", (r["id"],))
                remove_file(FACES, r["crop"])
            drop_unused_snapshots(con)
    return {"deleted": len(rows)}


# ------------------------------------------------------------------ Falsch erkannt
NOT_A_NAME = {"", "?", "-", "unbekannt", "fremd", "niemand"}


@app.post("/api/recent/{rid}/correct")
async def correct(rid: int, request: Request):
    """Verwechslung korrigieren. {"name": "Anna"}: das Gesicht wird Lernbild von Anna (neu, falls noetig).
    {"name": ""} (oder "?", "unbekannt"): Gesicht wandert nach "Offen", gelernt wird nichts."""
    body = await request.json() if await request.body() else {}
    name = " ".join(str(body.get("name") or "").split())
    async with LOCK:
        with closing(db()) as con, con:
            r = con.execute("SELECT * FROM recent WHERE id = ?", (rid,)).fetchone()
            if not r:
                raise HTTPException(404, f"Erkennung nicht mehr vorhanden (wird nach {RECENT_HOURS:g} h geloescht)")
            old = con.execute("SELECT name FROM persons WHERE id = ?", (r["person_id"],)).fetchone()
            old_name = old["name"] if old else None
            matched = con.execute("SELECT id, crop FROM samples WHERE id = ? AND person_id = ?",
                                  (r["sample_id"], r["person_id"])).fetchone()
            created = False
            if name.lower() in NOT_A_NAME:
                group = _group_for(con, vec(r["embedding"]))
                uid = con.execute("INSERT INTO unknowns (event_id, embedding, crop, det_score, created, group_id) "
                                  "VALUES (?, ?, ?, NULL, ?, ?)", (r["event_id"], r["embedding"], r["crop"],
                                                                   r["created"], group or 0)).lastrowid
                if not group:
                    con.execute("UPDATE unknowns SET group_id = id WHERE id = ?", (uid,))
                face, new_name = {"known": False, "unknown_id": uid, "name": None, "person_id": None}, None
            else:
                pid, created = _person_id(con, name)
                if pid == r["person_id"]:
                    raise HTTPException(400, f"Das ist bereits {old_name}")
                con.execute("INSERT INTO samples (person_id, embedding, crop, created) VALUES (?, ?, ?, ?)",
                            (pid, r["embedding"], r["crop"], time.time()))
                new_name = con.execute("SELECT name FROM persons WHERE id = ?", (pid,)).fetchone()["name"]
                face = {"known": True, "person_id": pid, "name": new_name}
            con.execute("DELETE FROM recent WHERE id = ?", (rid,))
            event = rewrite_event(con, r["event_id"], rid, face)
    return {"name": new_name, "created": created, "was": old_name, "event": event,
            # das Lernbild, mit dem verwechselt wurde - ist es selbst falsch zugeordnet: entfernen
            "matched_sample": {"id": matched["id"], "crop": media("face", matched["crop"]), "person": old_name}
            if matched else None}


# ------------------------------------------------------------------ Personen
@app.get("/api/persons")
async def persons():
    with closing(db()) as con:
        out = []
        for p in con.execute("SELECT * FROM persons ORDER BY name").fetchall():
            samples = con.execute("SELECT id, crop, created FROM samples WHERE person_id = ? ORDER BY created DESC",
                                  (p["id"],)).fetchall()
            out.append({"id": p["id"], "name": p["name"], "count": len(samples),
                        "samples": [{"id": s["id"], "crop": media("face", s["crop"]), "created": s["created"]}
                                    for s in samples]})
    return out


@app.patch("/api/persons/{pid}")
async def rename(pid: int, request: Request):
    """Umbenennen; gibt es den Namen schon, werden beide Personen zusammengelegt."""
    body = await request.json()
    async with LOCK:
        with closing(db()) as con, con:
            if not con.execute("SELECT 1 FROM persons WHERE id = ?", (pid,)).fetchone():
                raise HTTPException(404, "Person nicht gefunden")
            name = " ".join(str(body.get("name", "")).split())
            if not name or len(name) > 60:
                raise HTTPException(400, "Name fehlt oder ist zu lang")
            other = con.execute("SELECT id FROM persons WHERE name = ? AND id != ?", (name, pid)).fetchone()
            if other:
                con.execute("UPDATE samples SET person_id = ? WHERE person_id = ?", (other["id"], pid))
                con.execute("DELETE FROM persons WHERE id = ?", (pid,))
                return {"id": other["id"], "merged": True}
            con.execute("UPDATE persons SET name = ? WHERE id = ?", (name, pid))
    return {"id": pid, "merged": False}


@app.delete("/api/persons/{pid}")
async def delete_person(pid: int):
    async with LOCK:
        with closing(db()) as con, con:
            crops = [r["crop"] for r in con.execute("SELECT crop FROM samples WHERE person_id = ?", (pid,))]
            if not con.execute("DELETE FROM persons WHERE id = ?", (pid,)).rowcount:
                raise HTTPException(404, "Person nicht gefunden")
    for c in crops:
        remove_file(FACES, c)
    return {"deleted": True, "samples": len(crops)}


@app.delete("/api/samples/{sid}")
async def delete_sample(sid: int):
    """Falsch zugewiesenes Gesicht entfernen."""
    async with LOCK:
        with closing(db()) as con, con:
            row = con.execute("SELECT crop FROM samples WHERE id = ?", (sid,)).fetchone()
            if not row:
                raise HTTPException(404, "Gesicht nicht gefunden")
            con.execute("DELETE FROM samples WHERE id = ?", (sid,))
    remove_file(FACES, row["crop"])
    return {"deleted": True}


@app.post("/api/samples/{sid}/move")
async def move_sample(sid: int, request: Request):
    """Gesicht einer anderen (ggf. neuen) Person zuordnen: {"name": "..."}."""
    body = await request.json()
    async with LOCK:
        with closing(db()) as con, con:
            row = con.execute("SELECT person_id FROM samples WHERE id = ?", (sid,)).fetchone()
            if not row:
                raise HTTPException(404, "Gesicht nicht gefunden")
            pid, _ = _person_id(con, str(body.get("name", "")))
            con.execute("UPDATE samples SET person_id = ? WHERE id = ?", (pid, sid))
            # die alte Person faellt weg, wenn das ihr letztes Gesicht war
            con.execute("DELETE FROM persons WHERE id = ? AND NOT EXISTS (SELECT 1 FROM samples WHERE person_id = ?)",
                        (row["person_id"], row["person_id"]))
    return {"person_id": pid}


@app.post("/api/persons/{pid}/photo")
async def add_photo(pid: int, image: UploadFile = File(...)):
    """Person per Foto anlernen (z.B. Handyfoto), groesstes Gesicht im Bild zaehlt."""
    data = await image.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Bild zu gross")
    try:
        faces = await run_in_threadpool(ENGINE.analyze, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not faces:
        raise HTTPException(400, "Kein Gesicht gefunden")
    async with LOCK:
        with closing(db()) as con, con:
            if not con.execute("SELECT 1 FROM persons WHERE id = ?", (pid,)).fetchone():
                raise HTTPException(404, "Person nicht gefunden")
            crop = save_jpeg(FACES, "p", faces[0].crop)
            con.execute("INSERT INTO samples (person_id, embedding, crop, created) VALUES (?, ?, ?, ?)",
                        (pid, faces[0].embedding.tobytes(), crop, time.time()))
    return {"added": 1, "faces_in_image": len(faces)}


@app.post("/api/persons")
async def create_person(request: Request):
    body = await request.json()
    async with LOCK:
        with closing(db()) as con, con:
            pid, created = _person_id(con, str(body.get("name", "")))
    return {"id": pid, "created": created}


# ------------------------------------------------------------------ Verlauf
@app.get("/api/events")
async def events(limit: int = 50, source: str | None = None):
    with closing(db()) as con:
        where, args = ("WHERE source = ? ", (source,)) if source else ("", ())
        rows = con.execute(f"SELECT id, created, source, text, result FROM events {where}ORDER BY id DESC LIMIT ?",
                           (*args, max(1, min(limit, 500)))).fetchall()
        open_ids = {r["id"] for r in con.execute("SELECT id FROM recent")}
    out = []
    for r in rows:
        res = json.loads(r["result"]) if r["result"] else None
        for f in (res or {}).get("faces", []):
            f["correctable"] = f.get("recent_id") in open_ids
        out.append({"id": r["id"], "created": r["created"], "source": r["source"], "text": r["text"], "result": res})
    return out


@app.get("/api/events/sources")
async def event_sources():
    with closing(db()) as con:
        return [r["source"] for r in con.execute(
            "SELECT DISTINCT source FROM events WHERE source IS NOT NULL ORDER BY source").fetchall()]


@app.get("/api/settings")
async def settings():
    return {"match": MATCH, "suggest": SUGGEST, "recent_hours": RECENT_HOURS, "providers": getattr(ENGINE, "providers", None),
            "ha": bool(HA_URL and HA_TOKEN), "token": bool(TOKEN), "version": __version__}



# ---------------------------------------------------------------- Modul fuer den OllamaProxy
# Der OllamaProxy zeigt Module mit Link und Speicher in seiner Oberflaeche (Schnittstelle /proxy-module).


def _rss() -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


@app.get("/proxy-module")
async def proxy_module(request: Request):
    lang = i18n.request_lang(request.headers, LANG_HEADER)
    return {"name": "FaceRec", "version": __version__,
            "links": [{"label": i18n.tr("Gesichter öffnen", lang), "path": "/"}], "capabilities": ["memory"]}


@app.get("/proxy-module/memory")
async def proxy_module_memory(request: Request):
    rss = _rss()
    return {"rss": rss, "gpu": 0, "held": rss or 0, "reclaimable": False, "busy": False,
            "detail": i18n.tr("Erkennungsmodell geladen", i18n.request_lang(request.headers, LANG_HEADER))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning", access_log=False)
