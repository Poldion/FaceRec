"""
Tests der Gesichtserkennung ohne echte Modelle: ein nachgebauter Erkenner liest die "Gesichter"
aus dem Testbild. Jedes 32x32-Feld ist ein Gesicht; Rot = Person, Gruen = Aufnahme (andere
Variante derselben Person), Blau = wie stark die Aufnahme abweicht (0 = gleich, 250 = sehr).
"""
import importlib
import os
import sys

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine  # noqa: E402


def base(person: int) -> np.ndarray:
    return np.random.default_rng(person).normal(size=512)


class FakeEngine:
    providers = ["Fake"]

    def analyze(self, data: bytes):
        img = engine.decode_image(data)
        if img is None:
            raise ValueError("Kein lesbares Bild")
        faces = []
        for x in range(0, img.shape[1], 32):
            b, g, r = (int(v) for v in img[16, x + 16])
            if r == 0:                      # schwarzes Feld = kein Gesicht
                continue
            v = base(r) + (b / 100) * np.random.default_rng(10_000 + r * 256 + g).normal(size=512)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            faces.append(engine.Face((x, 0, x + 32, 32), 0.9, v, engine.face_crop(img, (x, 0, x + 32, 32), 48)))
        return faces


def picture(*faces: tuple[int, int, int]) -> bytes:
    """faces: (Person, Aufnahme, Abweichung) je Gesicht; (0, 0, 0) = kein Gesicht."""
    img = np.zeros((32, 32 * max(1, len(faces)), 3), np.uint8)
    for i, (p, g, b) in enumerate(faces):
        img[:, i * 32:(i + 1) * 32] = (b, g, p)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACEREC_DATA", str(tmp_path))
    import app as appmod
    appmod = importlib.reload(appmod)
    appmod.ENGINE = FakeEngine()
    with TestClient(appmod.app) as c:
        yield c, appmod, tmp_path


def rec(c, *faces):
    r = c.post("/api/recognize", files={"image": ("k.png", picture(*faces), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_proxy_module(env):
    c, _, _ = env
    info = c.get("/proxy-module").json()
    assert info["name"] == "FaceRec" and info["capabilities"] == ["memory"]
    m = c.get("/proxy-module/memory").json()
    assert m["reclaimable"] is False and m["held"] == (m["rss"] or 0)


def files(path, sub):
    return sorted(os.listdir(os.path.join(path, sub)))


# ------------------------------------------------------------------ Kernablauf

def test_unknown_assign_then_recognized_kept_only_briefly(env):
    c, appmod, tmp = env
    r = rec(c, (5, 1, 30))
    assert r["text"] == "eine unbekannte Person" and r["unknown_count"] == 1 and not r["known"]
    assert r["snapshot"] and len(files(tmp, "snapshots")) == 1
    # dieselbe Person, andere Aufnahme -> gleiche Gruppe
    rec(c, (5, 2, 30))
    groups = c.get("/api/unknowns").json()
    assert len(groups) == 1 and len(groups[0]["faces"]) == 2
    uid = groups[0]["faces"][0]["id"]
    a = c.post(f"/api/unknowns/{uid}/assign", json={"name": "  Anna  "}).json()
    assert a == {"person_id": a["person_id"], "name": "Anna", "created": True, "assigned": 2}
    assert c.get("/api/unknowns").json() == []
    assert files(tmp, "snapshots") == []                       # Standbilder weg, sobald alles zugewiesen
    faces_before = files(tmp, "faces")
    assert len(faces_before) == 2                              # die zwei zugewiesenen Ausschnitte bleiben
    # wiedererkannt: kein Standbild, nur der Ausschnitt fuer eine moegliche Korrektur ...
    r = rec(c, (5, 3, 30))
    assert r["known"] and r["names"] == ["Anna"] and r["text"] == "Anna" and r["snapshot"] is None
    assert r["faces"][0]["recent_id"] and files(tmp, "snapshots") == []
    assert len(files(tmp, "faces")) == 3
    # ... und der ist nach FACEREC_RECENT_HOURS weg (wird beim naechsten Klingeln aufgeraeumt)
    appmod.RECENT_HOURS = 0
    rec(c, (0, 0, 0))
    assert files(tmp, "faces") == faces_before
    assert c.post(f"/api/recent/{r['faces'][0]['recent_id']}/correct", json={"name": "Bob"}).status_code == 404
    p = c.get("/api/persons").json()
    assert [(x["name"], x["count"]) for x in p] == [("Anna", 2)]


def test_several_faces_and_texts(env):
    c, _, _ = env
    uid = rec(c, (7, 1, 20))["faces"][0]["unknown_id"]
    c.post(f"/api/unknowns/{uid}/assign", json={"name": "Peter"})
    r = rec(c, (7, 2, 20), (9, 1, 20), (11, 1, 20))
    assert r["text"] == "Peter und zwei unbekannte Personen" and r["face_count"] == 3
    assert len(c.get("/api/unknowns").json()) == 2             # zwei verschiedene Fremde = zwei Gruppen
    assert rec(c, (0, 0, 0))["text"] == "niemand erkennbar"


def test_assign_existing_name_case_insensitive_and_single(env):
    c, _, _ = env
    u1 = rec(c, (3, 1, 20))["faces"][0]["unknown_id"]
    c.post(f"/api/unknowns/{u1}/assign", json={"name": "Anna"})
    u2 = rec(c, (4, 1, 20))["faces"][0]["unknown_id"]
    rec(c, (4, 2, 20))
    r = c.post(f"/api/unknowns/{u2}/assign", json={"name": "anna", "group": False}).json()
    assert r["name"] == "Anna" and not r["created"] and r["assigned"] == 1
    assert len(c.get("/api/unknowns").json()[0]["faces"]) == 1  # der Rest der Gruppe bleibt offen
    assert c.post(f"/api/unknowns/{u2}/assign", json={"name": "x"}).status_code == 404
    assert c.post(f"/api/unknowns/999/assign", json={"name": "x"}).status_code == 404
    left = c.get("/api/unknowns").json()[0]["faces"][0]["id"]
    assert c.post(f"/api/unknowns/{left}/assign", json={"name": "   "}).status_code == 400


def test_suggestion_for_similar_face(env, monkeypatch):
    c, appmod, _ = env
    monkeypatch.setattr(appmod, "MATCH", 0.45)                 # Test rechnet mit den Schwellen aus dem README
    monkeypatch.setattr(appmod, "SUGGEST", 0.35)
    uid = rec(c, (6, 1, 20))["faces"][0]["unknown_id"]
    c.post(f"/api/unknowns/{uid}/assign", json={"name": "Lena"})
    r = rec(c, (6, 9, 220))                                    # stark abweichende Aufnahme: nur aehnlich
    f = r["faces"][0]
    assert not f["known"] and f["suggestion"] == "Lena" and appmod.SUGGEST <= f["score"] < appmod.MATCH
    assert c.get("/api/unknowns").json()[0]["suggestion"] == "Lena"
    assert rec(c, (8, 1, 0))["faces"][0]["suggestion"] is None   # Fremder: kein Vorschlag


def test_ignore_removes_group_and_files(env):
    c, _, tmp = env
    uid = rec(c, (12, 1, 20))["faces"][0]["unknown_id"]
    rec(c, (12, 2, 20))
    assert c.post(f"/api/unknowns/{uid}/ignore").json() == {"deleted": 2}
    assert c.get("/api/unknowns").json() == [] and files(tmp, "faces") == [] and files(tmp, "snapshots") == []


def test_snapshot_kept_while_other_face_of_event_open(env):
    c, _, tmp = env
    r = rec(c, (13, 1, 20), (14, 1, 20))
    c.post(f"/api/unknowns/{r['faces'][0]['unknown_id']}/assign", json={"name": "A"})
    assert len(files(tmp, "snapshots")) == 1
    c.post(f"/api/unknowns/{r['faces'][1]['unknown_id']}/ignore")
    assert files(tmp, "snapshots") == []


# ------------------------------------------------------------------ Personen pflegen

def test_rename_merge_delete_and_move(env):
    c, _, tmp = env
    for person, name in ((20, "Anna"), (21, "Ana"), (22, "Bob")):
        uid = rec(c, (person, 1, 20))["faces"][0]["unknown_id"]
        c.post(f"/api/unknowns/{uid}/assign", json={"name": name})
    ps = {p["name"]: p for p in c.get("/api/persons").json()}
    assert c.patch(f"/api/persons/{ps['Ana']['id']}", json={"name": "anna"}).json()["merged"] is True
    ps = {p["name"]: p for p in c.get("/api/persons").json()}
    assert set(ps) == {"Anna", "Bob"} and ps["Anna"]["count"] == 2
    # falsch zugeordnetes Gesicht zu Bob verschieben, dann zurueck in eine neue Person
    sid = ps["Anna"]["samples"][0]["id"]
    c.post(f"/api/samples/{sid}/move", json={"name": "Bob"})
    c.post("/api/persons", json={"name": "Leer"})              # Person ohne Gesichter bleibt erhalten
    bob = next(p for p in c.get("/api/persons").json() if p["name"] == "Bob")
    for s in bob["samples"]:
        c.post(f"/api/samples/{s['id']}/move", json={"name": "Carl"})
    names = {p["name"]: p["count"] for p in c.get("/api/persons").json()}
    assert names == {"Anna": 1, "Carl": 2, "Leer": 0}          # Bob war leer -> weg, "Leer" bleibt
    carl = next(p for p in c.get("/api/persons").json() if p["name"] == "Carl")
    assert c.delete(f"/api/persons/{carl['id']}").json() == {"deleted": True, "samples": 2}
    assert len(files(tmp, "faces")) == 1
    anna = next(p for p in c.get("/api/persons").json() if p["name"] == "Anna")
    assert c.delete(f"/api/samples/{anna['samples'][0]['id']}").json() == {"deleted": True}
    assert files(tmp, "faces") == []
    assert c.delete("/api/persons/999").status_code == 404


def test_add_photo_to_person(env):
    c, _, _ = env
    pid = c.post("/api/persons", json={"name": "Mia"}).json()["id"]
    r = c.post(f"/api/persons/{pid}/photo", files={"image": ("a.png", picture((30, 1, 20)), "image/png")})
    assert r.json() == {"added": 1, "faces_in_image": 1}
    assert c.post(f"/api/persons/{pid}/photo", files={"image": ("a.png", picture((0, 0, 0)), "image/png")}).status_code == 400
    assert rec(c, (30, 2, 20))["names"] == ["Mia"]


# ------------------------------------------------------------------ Eingaenge

def test_raw_body_and_bad_input(env):
    c, _, _ = env
    r = c.post("/api/recognize", content=picture((40, 1, 20)), headers={"content-type": "image/png"})
    assert r.status_code == 200 and r.json()["unknown_count"] == 1
    assert c.post("/api/recognize", content=b"kaputt", headers={"content-type": "image/jpeg"}).status_code == 400
    assert c.post("/api/recognize", content=b"", headers={"content-type": "image/jpeg"}).status_code == 400


def test_entity_from_home_assistant(env, monkeypatch):
    c, appmod, _ = env
    assert c.post("/api/recognize", json={"entity_id": "image.tuer"}).status_code == 400     # HA nicht eingerichtet
    monkeypatch.setattr(appmod, "HA_URL", "http://ha:8123")
    monkeypatch.setattr(appmod, "HA_TOKEN", "t")
    seen = []

    def handler(req):
        seen.append((req.url.path, req.headers["authorization"]))
        return httpx.Response(200, content=picture((41, 1, 20)))
    real = httpx.AsyncClient
    monkeypatch.setattr(appmod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r = c.post("/api/recognize", json={"entity_id": "image.haustuer_event_image"})
    assert r.status_code == 200 and r.json()["unknown_count"] == 1
    c.post("/api/recognize", json={"entity_id": "camera.haustuer"})
    assert seen == [("/api/image_proxy/image.haustuer_event_image", "Bearer t"),
                    ("/api/camera_proxy/camera.haustuer", "Bearer t")]
    assert c.post("/api/recognize", json={"entity_id": "../../etc"}).status_code == 400
    assert c.get("/api/events").json()[0]["source"] == "camera.haustuer"


# ------------------------------------------------------------------ Schutz

def test_origin_and_token(env, monkeypatch):
    c, appmod, _ = env
    r = c.post("/api/persons", json={"name": "X"}, headers={"origin": "http://boese.example"})
    assert r.status_code == 403
    assert c.post("/api/persons", json={"name": "X"}, headers={"origin": "http://testserver"}).status_code == 200
    monkeypatch.setattr(appmod, "TOKEN", "geheim")
    assert c.get("/api/persons").status_code == 401
    assert c.get("/api/persons", headers={"authorization": "Bearer falsch"}).status_code == 401
    assert c.get("/api/persons", headers={"authorization": "Bearer geheim"}).status_code == 200
    c.cookies.set("facerec_token", "geheim")
    assert c.get("/api/persons").status_code == 200
    assert c.get("/health").status_code == 200 and c.get("/").status_code == 200


def test_media_paths(env):
    c, _, _ = env
    crop = rec(c, (50, 1, 20))["faces"][0]["crop"]
    r = c.get(crop)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    for bad in ("/media/face/..%2Ffacerec.db", "/media/x/a.jpg", "/media/face/nicht_da.jpg", "/media/snap/facerec.db"):
        assert c.get(bad).status_code == 404


def test_event_trim_keeps_open_events(env, monkeypatch):
    c, appmod, _ = env
    monkeypatch.setattr(appmod, "KEEP_EVENTS", 3)
    first = rec(c, (60, 1, 20))["event_id"]                    # offen (unbekannt) -> bleibt
    for _ in range(5):
        rec(c, (0, 0, 0))
    ids = [e["id"] for e in c.get("/api/events?limit=50").json()]
    assert first in ids and len(ids) == 4


def test_say():
    from app import say
    assert say([], 0) == "niemand erkennbar"
    assert say(["Anna"], 0) == "Anna"
    assert say(["Anna", "Bob"], 1) == "Anna, Bob und eine unbekannte Person"
    assert say([], 5) == "5 unbekannte Personen"


def test_sentence():
    from app import sentence
    assert sentence([], 0) == "Es klingelt, kein Gesicht erkennbar."
    assert sentence(["Anna"], 0) == "Anna steht vor der Tür."
    assert sentence([], 1) == "Eine unbekannte Person steht vor der Tür."
    assert sentence(["Anna"], 1) == "Anna und eine unbekannte Person stehen vor der Tür."


def test_sentence_english():
    from app import sentence
    assert sentence([], 0, "en") == "Someone is at the door, no face visible."
    assert sentence(["Anna"], 0, "en") == "Anna is at the door."
    assert sentence([], 1, "en") == "An unknown person is at the door."
    assert sentence(["Anna", "Bob"], 2, "en") == "Anna, Bob and two unknown people are at the door."


# ------------------------------------------------------------------ Sprache

def test_i18n_script_and_translated_errors(env):
    c, _, _ = env
    js = c.get("/i18n.js")
    assert js.status_code == 200 and "window.I18N" in js.text and '"Verlauf": "History"' in js.text
    assert c.delete("/api/persons/999").json()["detail"] == "Person nicht gefunden"
    assert c.delete("/api/persons/999", headers={"X-FaceRec-Lang": "en"}).json()["detail"] == "Person not found"
    assert c.delete("/api/persons/999", headers={"Accept-Language": "en-US,en"}).json()["detail"] == "Person not found"
    r = c.post("/api/recent/999/correct", json={"name": "Max"}, headers={"X-FaceRec-Lang": "en"})
    assert r.status_code == 404 and r.json()["detail"] == "Recognition no longer exists (deleted after 48 h)"
    assert c.get("/proxy-module", headers={"Accept-Language": "en"}).json()["links"][0]["label"] == "Open faces"


def test_all_texts_translated():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, os.path.join(root, "scripts", "check_i18n.py")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_event_trim_per_source_and_filter(env, monkeypatch):
    c, appmod, _ = env
    monkeypatch.setattr(appmod, "KEEP_EVENTS", 2)
    with appmod.closing(appmod.db()) as con, con:
        door = con.execute("INSERT INTO events (created, source) VALUES (1, 'image.klingel')").lastrowid
        for i in range(5):
            con.execute("INSERT INTO events (created, source) VALUES (?, 'camera.flur')", (i + 2,))
        appmod._trim_events(con)
    assert [e["id"] for e in c.get("/api/events?source=image.klingel").json()] == [door]
    assert len(c.get("/api/events?source=camera.flur").json()) == 2
    assert c.get("/api/events/sources").json() == ["camera.flur", "image.klingel"]


def test_stream_scan_per_camera(env, monkeypatch):
    c, appmod, _ = env
    started, stopped = [], []

    async def fake_scan(entity_id, url, duration):
        started.append(entity_id)
        try:
            await appmod.asyncio.sleep(60)
        except appmod.asyncio.CancelledError:
            stopped.append(entity_id)
            raise
    monkeypatch.setattr(appmod, "_run_stream_scan", fake_scan)

    class Req:
        def __init__(self, e):
            self.headers, self.e = {"content-type": "application/json"}, e

        async def json(self):
            return {"entity_id": self.e}

    async def run():
        for e in ("camera.a", "camera.b"):
            await appmod.start_stream_scan(Req(e))
        await appmod.asyncio.sleep(0)
        assert stopped == []                                    # andere Kamera laeuft weiter
        await appmod.start_stream_scan(Req("camera.a"))
        await appmod.asyncio.sleep(0)
        assert stopped == ["camera.a"] and started == ["camera.a", "camera.b", "camera.a"]
        for t in appmod.STREAM_SCANS.values():
            t.cancel()
    appmod.asyncio.run(run())


# ------------------------------------------------------------------ Falsch erkannt

def _learn(c, person, name, variant=1):
    uid = rec(c, (person, variant, 20))["faces"][0]["unknown_id"]
    return c.post(f"/api/unknowns/{uid}/assign", json={"name": name}).json()


def test_correct_wrong_person_learns_and_updates_event(env):
    c, appmod, _ = env
    _learn(c, 70, "Max")
    c.post("/api/persons", json={"name": "Lisa"})           # Lisa angelegt, aber noch ohne Gesicht
    # Lisas Gesicht wird faelschlich Max zugeordnet (simuliert: lockere Schwelle)
    appmod.MATCH = -1.0
    r = rec(c, (71, 5, 220), (70, 2, 20))
    appmod.MATCH = 0.45
    wrong = next(f for f in r["faces"] if f["known"] and f["name"] == "Max" and f["bbox"][0] == 0)
    assert r["text"] == "Max"
    out = c.post(f"/api/recent/{wrong['recent_id']}/correct", json={"name": "lisa"}).json()
    assert out["name"] == "Lisa" and out["was"] == "Max" and not out["created"]
    assert out["event"]["names"] == ["Lisa", "Max"] and out["event"]["sentence"] == "Lisa und Max stehen vor der Tür."
    assert out["matched_sample"]["person"] == "Max" and out["matched_sample"]["crop"].startswith("/media/face/")
    counts = {p["name"]: p["count"] for p in c.get("/api/persons").json()}
    assert counts == {"Max": 1, "Lisa": 1}                    # Lisa hat das Gesicht gelernt
    ev = c.get("/api/events").json()[0]
    assert ev["text"] == "Lisa und Max"
    f0 = ev["result"]["faces"][0]
    assert f0["corrected"] and not f0["correctable"] and ev["result"]["faces"][1]["correctable"]
    # zweimal korrigieren geht nicht, gleiche Person auch nicht
    assert c.post(f"/api/recent/{wrong['recent_id']}/correct", json={"name": "Lisa"}).status_code == 404
    other = ev["result"]["faces"][1]["recent_id"]
    assert c.post(f"/api/recent/{other}/correct", json={"name": "max"}).status_code == 400


def test_correct_to_unknown_and_new_person(env):
    c, appmod, tmp = env
    _learn(c, 80, "Max")
    appmod.MATCH = -1.0
    r = rec(c, (81, 1, 20), (82, 1, 20))
    appmod.MATCH = 0.45
    a, b = (f["recent_id"] for f in r["faces"])
    out = c.post(f"/api/recent/{a}/correct", json={"name": ""}).json()
    assert out["name"] is None and out["event"]["sentence"] == "Max und eine unbekannte Person stehen vor der Tür."
    groups = c.get("/api/unknowns").json()
    assert len(groups) == 1 and groups[0]["faces"][0]["crop"]  # wartet jetzt unter "Offen"
    out = c.post(f"/api/recent/{b}/correct", json={"name": "Neu"}).json()
    assert out["created"] and out["event"]["text"] == "Neu und eine unbekannte Person"
    assert {p["name"]: p["count"] for p in c.get("/api/persons").json()} == {"Max": 1, "Neu": 1}
    # die verschobenen Ausschnitte bleiben erhalten (Offen bzw. Lernbild), auch nach dem Aufraeumen
    appmod.RECENT_HOURS = 0
    rec(c, (0, 0, 0))
    for crop in [groups[0]["faces"][0]["crop"]] + [s["crop"] for p in c.get("/api/persons").json() for s in p["samples"]]:
        assert c.get(crop).status_code == 200


def test_correct_bad_input(env):
    c, _, _ = env
    assert c.post("/api/recent/999/correct", json={"name": "x"}).status_code == 404
