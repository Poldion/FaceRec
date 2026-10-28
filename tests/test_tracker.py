"""
Tests der Verfolgung im Flur-Stream (tracker.py) und des Stream-Scans in app.py mit nachgebautem Erkenner.
Szenarien: mehrere Personen gleichzeitig, Gast neben bekannter Person, Einzelbild-Ausreisser,
mehrere Unbekannte, Person kurz aus dem Bild, Sammelfenster.
"""
import asyncio
import importlib
import json
import os
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tracker  # noqa: E402
from tracker import Match, Sample, SceneTracker  # noqa: E402

MATCH, SUGGEST = 0.32, 0.28


def sample(bbox, det=0.9):
    return Sample(tracker.quality(det, bbox), None, b"", det, bbox)


def box(x, y=100, s=80):
    return (x, y, x + s, y + s)


def frame(scene, now, faces):
    """faces: [(bbox, {Name: Wert})] -> (Spuren, neu anzusagende Spuren)"""
    tracks = scene.associate([f[0] for f in faces], now)
    items = [(t, sample(bb), {n: Match(i + 1, v, None) for i, (n, v) in enumerate(sc.items())})
             for t, (bb, sc) in zip(tracks, faces)]
    decided = scene.add_evidence(items, now)
    return tracks, scene.to_announce(decided, now)


def run_until_batch(scene, start, step, faces_fn, frames):
    now = start
    for i in range(frames):
        frame(scene, now, faces_fn(i))
        now += step
        if scene.batch_ready(now):
            return scene.take_batch(), now
    now += tracker.MAX_WINDOW
    return (scene.take_batch() if scene.batch_ready(now) else []), now


def kinds(tracks):
    return sorted((t.decision.kind, t.decision.name) for t in tracks)


# ------------------------------------------------------------------ Logik

def test_two_known_side_by_side_same_speed_one_batch():
    scene = SceneTracker(MATCH, SUGGEST)
    faces = lambda i: [(box(100 + 10 * i), {"Max": 0.36, "Anton": 0.05}),
                       (box(300 + 10 * i), {"Anton": 0.35, "Max": 0.08})]
    batch, _ = run_until_batch(scene, 0.0, 0.1, faces, 40)
    assert kinds(batch) == [("known", "Anton"), ("known", "Max")]


def test_guest_next_to_known_person_is_not_swallowed():
    scene = SceneTracker(MATCH, SUGGEST)
    # Gast aehnelt Max etwas (0.30) - frueher: als "Max bewegt sich" ignoriert
    faces = lambda i: [(box(100 + 8 * i), {"Max": 0.55, "Anton": 0.1}),
                       (box(320 + 8 * i), {"Max": 0.30, "Anton": 0.12})]
    batch, _ = run_until_batch(scene, 0.0, 0.1, faces, 40)
    assert kinds(batch) == [("known", "Max"), ("unknown", None)]


def test_two_similar_faces_never_both_the_same_person():
    scene = SceneTracker(MATCH, SUGGEST)
    faces = lambda i: [(box(100 + 5 * i), {"Max": 0.50}), (box(300 + 5 * i), {"Max": 0.45})]
    batch, _ = run_until_batch(scene, 0.0, 0.1, faces, 40)
    names = [t.decision.name for t in batch if t.decision.kind == "known"]
    assert names == ["Max"]


def test_single_frame_outlier_is_not_enough():
    scene = SceneTracker(MATCH, SUGGEST)
    _, new = frame(scene, 0.0, [(box(100), {"Anton": 0.33})])
    assert new == [] and scene.tracks[0].decision is None
    _, new = frame(scene, 0.1, [(box(105), {"Anton": 0.34})])     # zweites Bild >= MATCH -> bekannt
    assert [t.decision.name for t in new] == ["Anton"]


def test_two_unknown_guests_counted_separately():
    scene = SceneTracker(MATCH, SUGGEST)
    faces = lambda i: [(box(100 + 6 * i), {"Max": 0.05}), (box(320 + 6 * i), {"Anton": 0.1})]
    batch, _ = run_until_batch(scene, 0.0, 0.1, faces, 40)
    assert kinds(batch) == [("unknown", None), ("unknown", None)]


def test_suggestion_after_several_frames():
    scene = SceneTracker(MATCH, SUGGEST)
    faces = lambda i: [(box(100 + 6 * i), {"Anton": 0.29})]
    batch, _ = run_until_batch(scene, 0.0, 0.1, faces, 40)
    assert kinds(batch) == [("suggest", "Anton")]


def test_person_leaving_and_returning_not_greeted_twice():
    scene = SceneTracker(MATCH, SUGGEST)
    _, new = frame(scene, 0.0, [(box(100), {"Max": 0.6})])
    assert len(new) == 1
    scene.take_batch()
    # 3 s spaeter wieder im Bild -> neue Spur, aber schon begruesst
    _, new = frame(scene, 3.0, [(box(500), {"Max": 0.6})])
    assert new == []


def test_batch_waits_for_open_visible_track_but_not_forever():
    scene = SceneTracker(MATCH, SUGGEST)
    frame(scene, 0.0, [(box(100), {"Max": 0.6}), (box(400), {"Anton": 0.25})])   # Anton noch unklar
    t = 0.0
    while t < tracker.MAX_WINDOW - 0.1:
        t += 0.1
        frame(scene, t, [(box(100), {"Max": 0.6}), (box(400), {"Anton": 0.25})])
        if scene.batch_ready(t):
            break
    # Anton war nie eindeutig (0.25: weder bekannt noch unbekannt) -> erst nach AMBIGUOUS_FRAMES entschieden
    assert t >= tracker.MIN_WINDOW


def test_batch_closes_early_when_everyone_is_decided():
    scene = SceneTracker(MATCH, SUGGEST)
    frame(scene, 0.0, [(box(100), {"Max": 0.6})])
    assert not scene.batch_ready(0.5)
    assert scene.batch_ready(tracker.MIN_WINDOW + 0.01)


def test_track_follows_moving_face():
    scene = SceneTracker(MATCH, SUGGEST)
    t1 = scene.associate([box(100)], 0.0)[0]
    t2 = scene.associate([box(140)], 0.1)[0]          # 40 px weiter bei 80-px-Gesicht
    assert t1 is t2
    t3 = scene.associate([box(600)], 0.2)[0]          # weit weg -> neue Person
    assert t3 is not t1


# ------------------------------------------------------------------ Stream-Scan in app.py

def vec(seed):
    v = np.random.default_rng(seed).normal(size=512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def mix(a, b, w):
    v = w * a + (1 - w) * b
    return (v / np.linalg.norm(v)).astype(np.float32)


class StreamEngine:
    """Bild n (Grauwert = n) enthaelt die Gesichter aus SCRIPT[n]: [(x, Abdruck)]."""
    providers = ["Fake"]

    def __init__(self, script):
        self.script = script

    def detect(self, img):
        n = int(img[0, 0, 0])
        return [(np.array([x, 100, x + 80, 180], dtype=np.float32), 0.9, np.zeros((5, 2))) for x, _ in self.script[n]]

    def embed(self, img, kps):
        n = int(img[0, 0, 0])
        x = int(kps[0, 0])
        return self.script[n][x][1]


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACEREC_DATA", str(tmp_path))
    import app as appmod
    appmod = importlib.reload(appmod)
    appmod.ENGINE = StreamEngine([])            # verhindert das Laden der echten Modelle
    with TestClient(appmod.app):
        yield appmod


def test_stream_scan_two_people_and_guest(app_env, monkeypatch):
    appmod = app_env
    max_v, anton_v, guest = vec(1), vec(2), vec(3)
    with appmod.closing(appmod.db()) as con, con:
        for name, v in (("Max", max_v), ("Anton", anton_v)):
            pid = con.execute("INSERT INTO persons (name, created) VALUES (?, 0)", (name,)).lastrowid
            con.execute("INSERT INTO samples (person_id, embedding, created) VALUES (?, ?, 0)", (pid, v.tobytes()))
    # 30 Bilder: Max + Anton nebeneinander, Gast (etwas Max-aehnlich) dahinter
    script = [[(100 + 3 * n, mix(max_v, vec(100 + n), 0.6)), (260 + 3 * n, mix(anton_v, vec(200 + n), 0.6)),
               (420 + 3 * n, mix(max_v, guest, 0.25))] for n in range(30)]
    eng = StreamEngine(script)
    # embed() findet das Gesicht ueber kps[0,0] = Index -> kps passend setzen
    eng.detect = lambda img: [(np.array([x, 100, x + 80, 180], dtype=np.float32), 0.9,
                               np.full((5, 2), i, dtype=np.float32)) for i, (x, _) in enumerate(script[int(img[0, 0, 0])])]
    appmod.ENGINE = eng
    events = []

    async def fake_event(t, data):
        events.append((t, data))

    async def fake_reader(url, headers, duration, latest, new_frame):
        for n in range(30):
            img = np.full((240, 640, 3), n, np.uint8)
            latest["jpeg"] = cv2.imencode(".png", img)[1].tobytes()
            latest["seq"] += 1
            latest["received"] += 1
            new_frame.set()
            await asyncio.sleep(0.1)
        await asyncio.sleep(tracker.MAX_WINDOW)
        latest["done"] = True
        new_frame.set()

    monkeypatch.setattr(appmod, "fire_ha_event", fake_event)
    monkeypatch.setattr(appmod, "_read_frames", fake_reader)
    asyncio.run(appmod._run_stream_scan("camera.test", "http://x/", 10))
    singles = [d for t, d in events if t == "flur_person_erkannt"]
    finals = [d for t, d in events if t == "flur_personen_komplett"]
    # sofort je Person eine Meldung (fuer den schnellen Gruss), danach eine Abschlussmeldung fuer alle
    assert sorted(tuple(d["names"]) for d in singles) == [(), ("Anton",), ("Max",)]
    assert all(d["single"] and len(d["faces"]) == 1 for d in singles)
    assert [d["unknown_count"] for d in singles if not d["names"]] == [1]
    assert len(finals) == 1 and [t for t, _ in events][-1] == "flur_personen_komplett"
    data = finals[0]
    assert sorted(data["names"]) == ["Anton", "Max"]
    assert data["unknown"] is True and data["unknown_count"] == 1
    assert len({d["event_id"] for d in singles + finals}) == 1
    with appmod.closing(appmod.db()) as con:
        assert con.execute("SELECT COUNT(*) FROM unknowns").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM recent").fetchone()[0] == 2
        ev = json.loads(con.execute("SELECT result FROM events").fetchone()[0])
        assert ev["unknown_count"] == 1 and ev["face_count"] == 3


def test_split_frames():
    import app as appmod
    b = b"--BOUND"
    jpg = b"\xff\xd8abc\xff\xd9"
    frames, rest = appmod._split_frames(b + b"\r\nhdr\r\n\r\n" + jpg + b"\r\n" + b + b"\r\nhd" , b)
    assert frames == [jpg] and rest.startswith(b)


def test_second_person_2_6_s_later_joins_the_same_batch():
    # wie am 26.10. 21:39: Anton erschien 2,6 s nach Maxs Erkennung
    scene = SceneTracker(MATCH, SUGGEST)
    frame(scene, 0.0, [(box(100), {"Max": 0.6})])
    t, batch = 0.0, []
    while t < 6.0 and not batch:
        t += 0.1
        faces = [(box(100), {"Max": 0.6})] + ([(box(400), {"Anton": 0.5})] if t >= 2.6 else [])
        frame(scene, t, faces)
        if scene.batch_ready(t):
            batch = scene.take_batch()
    assert kinds(batch) == [("known", "Anton"), ("known", "Max")]


def test_close_faces_with_det_score_0_5_count_as_good():
    assert tracker.quality(0.52, box(0, s=120)) > 0


def _decided_track(appmod, name, pid, emb, scores, other=None):
    t = tracker.Track(1, box(0), 0, 0)
    t.evidence = {name: scores, **({"Anton": [other]} if other is not None else {})}
    t.decision = tracker.Decision("known", name, pid, max(scores), Sample(100, emb, b"jpg", 0.9, box(0)))
    return t


def test_learning_from_stream(app_env):
    appmod = app_env
    max_v = vec(1)
    with appmod.closing(appmod.db()) as con, con:
        pid = con.execute("INSERT INTO persons (name, created) VALUES ('Max', 0)").lastrowid
        con.execute("INSERT INTO samples (person_id, embedding, created) VALUES (?, ?, 0)", (pid, max_v.tobytes()))
        new_view = mix(max_v, vec(50), 0.5)                      # neue Ansicht: Aehnlichkeit < 0,75 zum Lernbild
        assert appmod.learn_from_stream(con, _decided_track(appmod, "Max", pid, new_view, [0.5, 0.48]), 1.0)
        # fast gleiches Bild nochmal -> nicht noch einmal
        assert not appmod.learn_from_stream(con, _decided_track(appmod, "Max", pid, new_view, [0.5, 0.48]), 2.0)
        # nur ein sicheres Bild, zu knapp gegen andere Person, oder zu schwach -> nicht lernen
        other_view = mix(max_v, vec(60), 0.5)
        assert not appmod.learn_from_stream(con, _decided_track(appmod, "Max", pid, other_view, [0.5]), 3.0)
        assert not appmod.learn_from_stream(con, _decided_track(appmod, "Max", pid, other_view, [0.5, 0.48], 0.4), 3.0)
        assert not appmod.learn_from_stream(con, _decided_track(appmod, "Max", pid, other_view, [0.41, 0.40]), 3.0)
        assert con.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 2
