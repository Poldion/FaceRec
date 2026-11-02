# Technik und Hintergründe

🇬🇧 [English](technik.en.md) · 🇩🇪 Deutsch

Ergänzung zur [README](../README.md). Der Dienst (Port **8191**) bekommt beim Klingeln das Bild der Klingel aus Home
Assistant, erkennt die Gesichter und gibt das Ergebnis zurück („Anna steht vor der Tür.“). Unbekannte Gesichter landen
unter „Offen“ und werden per Push-Knopf oder in der Oberfläche einer Person zugewiesen – damit lernt der Dienst dazu.

- **Erkannte Gesichter** werden gemeldet und nur `FACEREC_RECENT_HOURS` (48 h) als Ausschnitt + Abdruck aufgehoben,
  damit sich Verwechslungen korrigieren lassen; danach automatisch gelöscht.
- **Falsch erkannt:** Push-Knopf „Falsch erkannt“ (Texteingabe) oder „Falsch?“ im Verlauf. Name = richtige Person
  lernt das Gesicht (neu, falls nötig), leer/„?“ = unbekannt (kommt nach „Offen“). Ereignis, Satz und
  `sensor.klingel_besucher` werden mitkorrigiert; die Oberfläche zeigt das Lernbild, mit dem verwechselt wurde, und
  bietet an, es zu entfernen.
- **Unbekannte:** Ausschnitt + Standbild bleiben, bis zugewiesen oder verworfen; danach ist das Standbild weg, der
  Ausschnitt wird zum Lernbild der Person. Ähnliche Unbekannte werden **gruppiert** und gemeinsam zugewiesen.
- Ein vorhandener Name (Groß/klein egal) ordnet zu, ein neuer legt die Person an. Personen lassen sich auch per Foto
  anlernen, umbenennen (gleicher Name = zusammenlegen), falsche Gesichter verschieben oder entfernen.
- Alles lokal; nur im eigenen Netz betreiben (optional Schlüssel `FACEREC_TOKEN`).

## Modelle und Leistung

InsightFace `buffalo_l` (offizielles Release v0.7, Prüfsumme im Dockerfile): `det_10g` (SCRFD, Gesichter + 5
Merkmalspunkte) und `w600k_r50` (ArcFace, 512er Gesichtsabdruck). Ohne das insightface-Paket, direkt über ONNX Runtime
(`engine.py`). Lizenz der Modelle: nur nicht-kommerziell.

Gemessen auf einem Ryzen AI Max+ 395 (CPU): ~90 ms pro Bild. Mit erzeugten Testbildern: gleiche Person 0,83–0,92
Ähnlichkeit (auch 640×360-JPEG mit 92-px-Gesicht), verschiedene Personen 0,23–0,31. Schwellen: erkannt ab
`FACEREC_MATCH` 0,35, Vorschlag ab `FACEREC_SUGGEST` 0,28. Echte Klingelbilder (Nacht/IR, schräg) sind oft schlechter –
mehrere Aufnahmen pro Person zuweisen hilft am meisten. Lernen: Sichere Erkennungen (≥ `FACEREC_LEARN_MIN`, 0,45, in
mindestens zwei Bildern) werden als zusätzliche Lernbilder übernommen.

**GPU (AMD, ROCm 7.2 + ONNX Runtime MIGraphX, fp16)** mit `-f docker-compose.rocm.yml`: Gesichtssuche 2,6 ms (CPU
25,8 ms), Abdruck 1,8 ms (CPU 11,9 ms); im Stream-Scan 4 ms/Bild statt 60 ms. Abdrücke zu 99,9 % gleich wie auf der CPU.
Der erste Start kompiliert die Modelle (~2 min) nach `data/migraphx-cache`, danach startet der Dienst in ~3 s.
Startet die GPU nicht oder scheitert der erste Lauf, läuft der Dienst mit CPU weiter (`/health` zeigt das Rechenwerk).
Basis-Image `rocm/pytorch` (wie ComfyUI in ImageGen, Schichten geteilt), `onnxruntime-migraphx` von repo.radeon.com.
GPU-Gruppen des Hosts: `RENDER_GID`/`VIDEO_GID` in `facerec.env`.

**NPU** (`amdxdna`, `/dev/accel/accel0`): bräuchte AMDs Ryzen-AI-Paket (XRT + VitisAI-Execution-Provider,
quantisierte Modelle) und bringt bei einem Bild pro Klingeln nichts Messbares. Vorbereitet über `FACEREC_PROVIDERS`
und einen auskommentierten Geräte-Eintrag in `docker-compose.yml`.

## Stream-Scan (`POST /api/stream_scan`)

Home Assistant startet z. B. beim Öffnen der Haustür `{"entity_id": "camera.flur", "duration": 45}`. Der Dienst liest
den MJPEG-Stream – direkt von der Kamera, wenn `FACEREC_STREAM_URLS` eine Adresse für die Entität hat (z. B. ESP32-CAM),
sonst über Home Assistant (`/api/camera_proxy_stream`) – und meldet erkannte Personen als HA-Ereignis
`facerec_stream_person` (`names`, `unknown`, `unknown_count`, `faces`, `source`, `event_id`), am Ende `facerec_stream_complete`.
Die alten Namen `flur_person_erkannt` und `flur_personen_komplett` werden vorerst zusätzlich gesendet. Je Kamera läuft ein
eigener Scan, ein neuer ersetzt nur den laufenden derselben Kamera.

`tracker.py`:
- **Verfolgung statt Einzelbild:** Jedes Gesicht bekommt eine Spur über die Bildposition. Entschieden wird pro Spur:
  bekannt ab einem Bild ≥ 0,40 oder zwei Bildern ≥ `FACEREC_MATCH`; Vorschlag nach 4 guten Bildern ab
  `FACEREC_SUGGEST`; unbekannt nach 4 guten Bildern unter 0,22 (bzw. 8 Bildern unter `FACEREC_SUGGEST`).
- **Eindeutig:** pro Bild bekommt jede Person höchstens ein Gesicht, und eine Person, die schon einer anderen sichtbaren
  Spur gehört, keiner zweiten. Ein Gast neben einer bekannten Person wird so nicht als deren Bewegung verschluckt, und
  mehrere Unbekannte werden einzeln gezählt.
- **Sammelfenster nach Zustand:** startet mit der ersten Entscheidung, endet sobald keine sichtbare Spur mehr offen ist
  (frühestens 1,5 s, jede neue Spur +1 s), spätestens nach 4 s. Wer später kommt, wird als Nachzügler gemeldet; wer
  kurz aus dem Bild geht, nicht doppelt.
- **Immer das neueste Bild:** Ein Leser behält nur das jüngste Bild, die Auswertung läuft nie hinterher. Abdrücke nur
  für offene Spuren (höchstens 15 je Spur), Lernbild-Abdrücke im Speicher.

Einstellungen: `FACEREC_STREAM_FPS` (12), `FACEREC_DET_SIZE` (640; 480 ist ~40 % schneller bei der Gesichtssuche),
`FACEREC_THREADS` (0 = automatisch), `FACEREC_PROVIDERS` + `FACEREC_PROVIDER_OPTIONS` (JSON). Beim Start werden beide
Modelle einmal aufgewärmt (der erste Aufruf war ~2,5× langsamer).

## Home Assistant

Paket [homeassistant/facerec.yaml](../homeassistant/facerec.yaml) (rest_command + Sensoren) nach `/config/packages/`,
`FACEREC_HOST` darin ersetzen. Die Automationen ([homeassistant/automations.yaml](../homeassistant/automations.yaml))
sind eine Vorlage: Entitäten der eigenen Klingel und des Handys eintragen und in HA übernehmen. Das letzte Ereignisbild
einer Akkuklingel kann auch von Bewegung stammen; nachts (IR) wird nur erkannt, wer zur Klingel schaut.

- `rest_command.facerec_recognize` schickt `{"entity_id": "image.xxx"}`; der Dienst holt das Bild mit `HA_TOKEN` über
  `/api/image_proxy/...` (bzw. `camera_proxy` bei `camera.xxx`).
- Automation „Klingel: Gesicht erkennen“: Klingel-Sensor → bis 10 s auf neues Ereignisbild warten → erkennen → Push mit
  Bild und `sentence`, optional Sprachansage. Bei Unbekannten Knöpfe „Ist <Vorschlag>“, „Name eingeben“, „Verwerfen“.
- Automation „Gesicht zuweisen“ verarbeitet die Knöpfe (`mobile_app_notification_action`).
- Entitäten: `sensor.klingel_besucher` (z. B. „Anna“ / „eine unbekannte Person“, Attribute `satz`, `namen`, `bekannt`,
  `unbekannte`, `gesichter`, `zeitpunkt`) und `sensor.klingel_offene_gesichter` (REST, alle 5 Min. und sofort nach
  Klingeln/Zuweisen/Verwerfen).

Antwort von `/api/recognize`: `sentence`, `text`, `names`, `known`, `unknown_count`, `face_count`, `faces[]`
(`known`, `name`, `score`, `unknown_id`, `suggestion`, `crop`), `snapshot`.

## API

| Methode | Pfad | Zweck |
|---|---|---|
| POST | `/api/recognize` | Bild (Formularfeld `image`, Rohdaten) oder `{"entity_id"}` |
| POST | `/api/stream_scan` | `{"entity_id", "duration", "stream_url"}` – Stream einige Sekunden auswerten |
| GET | `/api/unknowns` | offene Gesichter, gruppiert, mit Vorschlag |
| POST | `/api/unknowns/{id}/assign` | `{"name"}` oder `{"person_id"}`, `"group": false` = nur dieses |
| POST | `/api/unknowns/{id}/ignore` | verwerfen (Standard: ganze Gruppe) |
| POST | `/api/recent/{id}/correct` | falsch erkannt: `{"name"}` richtige Person, leer = unbekannt |
| GET/POST | `/api/persons` | Personen mit Gesichtern / anlegen `{"name"}` |
| PATCH/DELETE | `/api/persons/{id}` | umbenennen (zusammenlegen) / löschen |
| POST | `/api/persons/{id}/photo` | per Foto anlernen |
| DELETE | `/api/samples/{id}` | falsches Gesicht entfernen |
| POST | `/api/samples/{id}/move` | Gesicht anderer Person geben `{"name"}` |
| GET | `/api/events` | Verlauf |
| GET | `/proxy-module`, `/proxy-module/memory` | Modul-Schnittstelle für den [OllamaProxy](https://github.com/Poldion/OllamaProxy) |

## Tests

`./run_tests.sh` (braucht nur Docker): baut das Image und lässt `tests/` mit einem nachgebauten Erkenner laufen
(34 Tests: Erkennen, Zuweisen, Gruppieren, Korrigieren, Aufräumen, Verfolgung im Stream-Scan, Modul-Schnittstelle).
