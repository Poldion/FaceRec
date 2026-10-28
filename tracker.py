"""
Verfolgung von Gesichtern ueber mehrere Bilder des Flur-Streams (reine Logik, ohne Datenbank/Netz).

Statt jedes Bild einzeln zu entscheiden, bekommt jedes Gesicht eine Spur (Track), die ueber die
Bildposition von Bild zu Bild weitergereicht wird. Pro Spur werden die Aehnlichkeiten zu allen
Personen gesammelt und erst dann entschieden:
- bekannt:   ein Bild >= STRONG oder mindestens zwei Bilder >= MATCH
- Vorschlag: nach UNKNOWN_FRAMES guten Bildern bester Wert >= SUGGEST, aber nicht bekannt
- unbekannt: nach UNKNOWN_FRAMES guten Bildern alles < UNKNOWN_MAX, oder nach
             AMBIGUOUS_FRAMES guten Bildern alles < SUGGEST
Eindeutig: In einem Bild bekommt jede Person hoechstens ein Gesicht (das aehnlichste), und eine
Person, die schon einer anderen sichtbaren Spur gehoert, kann keiner zweiten zugeordnet werden.
Sammelfenster: Die erste Entscheidung startet es; es endet, sobald keine sichtbare Spur mehr offen
ist (fruehestens MIN_WINDOW, jede neue Spur verlaengert um EXTEND), spaetestens nach MAX_WINDOW.
Sicher erkannte Gesichter (siehe app.learn_from_stream) werden als Lernbild der Person gespeichert.
"""
from dataclasses import dataclass, field

STRONG = 0.40
UNKNOWN_MAX = 0.22
UNKNOWN_FRAMES = 4
AMBIGUOUS_FRAMES = 8
MIN_QUALITY_DET = 0.50       # nahe Gesichter im Flur kommen oft nur auf 0,50-0,55 (2026-10-26)
MIN_QUALITY_SIZE = 32
TRACK_TIMEOUT = 1.5          # s ohne Sichtung -> Spur beendet
VISIBLE = 0.6                # s: Spur gilt als "gerade sichtbar"
MIN_WINDOW = 3.0             # 1,5 s war zu kurz: zweite Person kam 0,2 s nach dem Schliessen (2026-10-26)
EXTEND = 1.0
MAX_WINDOW = 5.0
MAX_EMBEDDINGS = 15          # hoechstens so viele Abdruecke je Spur (spart Rechenzeit)


@dataclass
class Match:
    person_id: int
    score: float
    sample_id: int | None


@dataclass
class Sample:
    quality: float
    embedding: object
    crop: bytes
    det_score: float
    bbox: tuple
    jpeg: bytes | None = None


@dataclass
class Decision:
    kind: str                     # "known" | "suggest" | "unknown"
    name: str | None
    person_id: int | None
    score: float
    sample: Sample
    sample_id: int | None = None


@dataclass
class Track:
    id: int
    bbox: tuple
    first_seen: float
    last_seen: float
    frames: int = 0
    quality_frames: int = 0
    evidence: dict = field(default_factory=dict)       # Name -> [Werte je Bild]
    best: dict = field(default_factory=dict)           # Name -> (Wert, Match, Sample)
    best_sample: Sample | None = None
    decision: Decision | None = None
    announced: bool = False
    embedded: int = 0

    def needs_evidence(self) -> bool:
        """Abdruck nur fuer offene Spuren berechnen; entschiedene Spuren werden nur noch verfolgt."""
        return self.decision is None and self.embedded < MAX_EMBEDDINGS

    def top(self) -> tuple[str | None, float]:
        if not self.evidence:
            return None, 0.0
        name = max(self.evidence, key=lambda n: max(self.evidence[n]))
        return name, max(self.evidence[name])


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a, b) -> float:
    return (((a[0] + a[2]) - (b[0] + b[2])) ** 2 + ((a[1] + a[3]) - (b[1] + b[3])) ** 2) ** 0.5 / 2


def quality(det_score: float, bbox) -> float:
    size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    return det_score * size if det_score >= MIN_QUALITY_DET and size >= MIN_QUALITY_SIZE else 0.0


class SceneTracker:
    def __init__(self, match: float, suggest: float):
        self.match, self.suggest = match, suggest
        self.tracks: list[Track] = []
        self.next_id = 1
        self.announced_names: set[str] = set()
        self.batch: dict | None = None

    # ------------------------------------------------------------------ Zuordnung Bild -> Spuren
    def associate(self, bboxes: list[tuple], now: float) -> list[Track]:
        """Ordnet jedes erkannte Gesicht einer Spur zu (neue Spur, wenn keine passt)."""
        self.tracks = [t for t in self.tracks if now - t.last_seen <= TRACK_TIMEOUT]
        pairs = []
        for di, bb in enumerate(bboxes):
            for t in self.tracks:
                o = iou(bb, t.bbox)
                size = max(t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1])
                d = center_distance(bb, t.bbox)
                if o >= 0.15 or d <= 0.8 * size:
                    pairs.append((o - d / max(size, 1) * 0.1, di, t))
        pairs.sort(key=lambda p: -p[0])
        result: list[Track | None] = [None] * len(bboxes)
        used = set()
        for _, di, t in pairs:
            if result[di] is None and t.id not in used:
                result[di] = t
                used.add(t.id)
        for di, bb in enumerate(bboxes):
            if result[di] is None:
                t = Track(self.next_id, bb, now, now)
                self.next_id += 1
                self.tracks.append(t)
                result[di] = t
                if self.batch:      # neue Person waehrend des Sammelfensters -> etwas laenger warten
                    self.batch["deadline"] = min(self.batch["start"] + MAX_WINDOW,
                                                 max(self.batch["deadline"], now + EXTEND))
            t = result[di]
            t.bbox, t.last_seen = bb, now
            t.frames += 1
        return result

    # ------------------------------------------------------------------ Aehnlichkeiten eines Bildes
    def add_evidence(self, items: list[tuple[Track, Sample, dict]], now: float) -> list[Track]:
        """items: (Spur, Bildausschnitt, {Name: Match}) fuer jedes ausgewertete Gesicht dieses Bildes.
        Gibt die Spuren zurueck, die in diesem Bild entschieden wurden."""
        # Personen, die schon einer anderen sichtbaren Spur gehoeren
        owned = {t.decision.name: t.id for t in self.tracks
                 if t.decision and t.decision.name and now - t.last_seen <= VISIBLE}
        # eindeutig je Bild: jede Person nur fuer das aehnlichste Gesicht
        cand = sorted(((m.score, i, name) for i, (_, _, ms) in enumerate(items) for name, m in ms.items()),
                      reverse=True)
        taken, allowed = set(), [set() for _ in items]
        for score, i, name in cand:
            track = items[i][0]
            if name in taken or owned.get(name, track.id) != track.id:
                continue
            if score >= self.suggest:
                taken.add(name)
            allowed[i].add(name)
        decided = []
        for i, (track, sample, ms) in enumerate(items):
            track.embedded += 1
            if track.best_sample is None or sample.quality > track.best_sample.quality:
                track.best_sample = sample
            if sample.quality > 0:
                track.quality_frames += 1
            for name, m in ms.items():
                if name not in allowed[i]:
                    continue
                track.evidence.setdefault(name, []).append(m.score)
                if name not in track.best or m.score > track.best[name][0]:
                    track.best[name] = (m.score, m, sample)
            if track.decision is None and self._decide(track):
                decided.append(track)
        return decided

    def _decide(self, t: Track) -> bool:
        for name, scores in sorted(t.evidence.items(), key=lambda kv: -max(kv[1])):
            if max(scores) >= STRONG or sum(s >= self.match for s in scores) >= 2:
                score, m, sample = t.best[name]
                t.decision = Decision("known", name, m.person_id, score, sample, m.sample_id)
                return True
        name, top = t.top()
        if t.quality_frames >= UNKNOWN_FRAMES and top >= self.suggest and name:
            score, m, sample = t.best[name]
            t.decision = Decision("suggest", name, m.person_id, score, t.best_sample or sample, m.sample_id)
            return True
        if t.best_sample and t.best_sample.quality > 0 and (
                (t.quality_frames >= UNKNOWN_FRAMES and top < UNKNOWN_MAX)
                or (t.quality_frames >= AMBIGUOUS_FRAMES and top < self.suggest)):
            t.decision = Decision("unknown", None, None, top, t.best_sample)
            return True
        return False

    # ------------------------------------------------------------------ Sammelfenster
    def to_announce(self, decided: list[Track], now: float) -> list[Track]:
        """Entschiedene Spuren, die noch angesagt werden muessen; startet/fuellt das Sammelfenster."""
        new = []
        for t in decided:
            d = t.decision
            if t.announced:
                continue
            t.announced = True
            if d.kind in ("known", "suggest") and d.name in self.announced_names:
                continue        # dieselbe Person schon begruesst (z.B. kurz aus dem Bild und zurueck)
            if d.kind in ("known", "suggest"):
                self.announced_names.add(d.name)
            if self.batch is None:
                self.batch = {"start": now, "deadline": now + MIN_WINDOW, "tracks": []}
            self.batch["tracks"].append(t)
            new.append(t)
        return new

    def batch_ready(self, now: float) -> bool:
        if not self.batch:
            return False
        if now >= self.batch["start"] + MAX_WINDOW:
            return True
        if now < self.batch["deadline"]:
            return False
        # noch offene, gerade sichtbare Spuren mit brauchbaren Bildern? dann noch warten
        return not any(t.decision is None and t.quality_frames > 0 and now - t.last_seen <= VISIBLE
                       for t in self.tracks)

    def take_batch(self) -> list[Track]:
        tracks, self.batch = (self.batch or {}).get("tracks", []), None
        return tracks
