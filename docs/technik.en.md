# Technical details and background

🇬🇧 English · 🇩🇪 [Deutsch](technik.md)

Supplement to the [README](../README.md). When someone rings, the service (port **8191**) gets the doorbell image from
Home Assistant, recognises the faces and returns the result (“Anna is at the door.”; language via
`FACEREC_LANG`). Unknown faces end up under “Open” and are assigned to a person via a notification button or in the
interface – that is how the service keeps learning.

- **Recognised faces** are reported and kept as crop + embedding only for `FACEREC_RECENT_HOURS` (48 h) so that
  mix-ups can be corrected; then they are deleted automatically.
- **Wrong person:** notification button “Falsch erkannt” (text input) or “Falsch?” in the history. A name = the right
  person learns the face (created if needed), empty/“?” = unknown (goes to “Open”). Event, sentence and
  `sensor.klingel_besucher` are corrected as well; the interface shows the training image it was confused with and
  offers to remove it.
- **Unknown faces:** crop + snapshot stay until assigned or discarded; then the snapshot is gone and the crop becomes a
  training image of the person. Similar unknown faces are **grouped** and assigned together.
- An existing name (case-insensitive) assigns, a new one creates the person. People can also be trained from a photo,
  renamed (same name = merge), wrong faces moved or removed.
- Everything stays local; only run it in your own network (optional key `FACEREC_TOKEN`).

## Models and performance

InsightFace `buffalo_l` (official release v0.7, checksum in the Dockerfile): `det_10g` (SCRFD, faces + 5 landmarks)
and `w600k_r50` (ArcFace, 512-dimensional face embedding). Without the insightface package, directly via ONNX Runtime
(`engine.py`). License of the models: non-commercial only.

Measured on a Ryzen AI Max+ 395 (CPU): ~90 ms per image. With generated test images: same person 0.83–0.92 similarity
(also 640×360 JPEG with a 92 px face), different people 0.23–0.31. Thresholds: recognised from `FACEREC_MATCH` 0.35,
suggestion from `FACEREC_SUGGEST` 0.28. Real doorbell images (night/IR, at an angle) are often worse – assigning several
shots per person helps most. Learning: confident recognitions (≥ `FACEREC_LEARN_MIN`, 0.45, in at least two images) are
taken over as additional training images.

**GPU (AMD, ROCm 7.2 + ONNX Runtime MIGraphX, fp16)** with `-f docker-compose.rocm.yml`: face detection 2.6 ms (CPU
25.8 ms), embedding 1.8 ms (CPU 11.9 ms); in the stream scan 4 ms/image instead of 60 ms. Embeddings are 99.9 % identical
to the CPU. The first start compiles the models (~2 min) into `data/migraphx-cache`; after that the service starts in
~3 s. If the GPU does not start or the first run fails, the service carries on with the CPU (`/health` shows the
execution provider). Base image `rocm/pytorch` (like ComfyUI in ImageGen, layers shared), `onnxruntime-migraphx` from
repo.radeon.com. GPU groups of the host: `RENDER_GID`/`VIDEO_GID` in `facerec.env`.

**NPU** (`amdxdna`, `/dev/accel/accel0`): would need AMD's Ryzen AI package (XRT + VitisAI execution provider, quantised
models) and brings nothing measurable with one image per ring. Prepared via `FACEREC_PROVIDERS` and a commented-out
device entry in `docker-compose.yml`.

## Stream scan (`POST /api/stream_scan`)

Home Assistant starts e.g. `{"entity_id": "camera.flur", "duration": 45}` when the front door opens. The service reads
the MJPEG stream – directly from the camera if `FACEREC_STREAM_URLS` has an address for the entity (e.g. ESP32-CAM),
otherwise via Home Assistant (`/api/camera_proxy_stream`) – and reports recognised people as the HA event
`facerec_stream_person` (`names`, `unknown`, `unknown_count`, `faces`, `source`, `event_id`), and `facerec_stream_complete` at the end.
The old names `flur_person_erkannt` and `flur_personen_komplett` are still sent as well for now. Each camera runs its own
scan; a new one only replaces the running scan of the same camera.

`tracker.py`:
- **Tracking instead of single images:** every face gets a track by its position in the image. Decisions are made per
  track: known from one image ≥ 0.40 or two images ≥ `FACEREC_MATCH`; suggestion after 4 good images from
  `FACEREC_SUGGEST`; unknown after 4 good images below 0.22 (or 8 images below `FACEREC_SUGGEST`).
- **Unambiguous:** per image each person gets at most one face, and a person who already belongs to another visible
  track gets no second one. That way a guest next to a known person is not swallowed as that person's movement, and
  several unknown people are counted individually.
- **Collection window by state:** starts with the first decision, ends as soon as no visible track is open (at the
  earliest after 1.5 s, +1 s per new track), at the latest after 4 s. Whoever comes later is reported as a latecomer;
  whoever briefly leaves the frame is not reported twice.
- **Always the newest image:** a reader only keeps the latest image, the analysis never lags behind. Embeddings only for
  open tracks (at most 15 per track), training image embeddings in memory.

Settings: `FACEREC_STREAM_FPS` (12), `FACEREC_DET_SIZE` (640; 480 is ~40 % faster at face detection),
`FACEREC_THREADS` (0 = automatic), `FACEREC_PROVIDERS` + `FACEREC_PROVIDER_OPTIONS` (JSON). At start both models are
warmed up once (the first call was ~2.5× slower).

## Home Assistant

Copy the package [homeassistant/facerec.yaml](../homeassistant/facerec.yaml) (rest_command + sensors) to
`/config/packages/` and replace `FACEREC_HOST` in it. The automations ([homeassistant/automations.yaml](../homeassistant/automations.yaml))
are a template: enter the entities of your own doorbell and phone and take them over into HA. The last event image of a
battery doorbell can also come from motion; at night (IR) only people looking at the doorbell are recognised.

- `rest_command.facerec_recognize` sends `{"entity_id": "image.xxx"}`; the service fetches the image with `HA_TOKEN`
  via `/api/image_proxy/...` (or `camera_proxy` for `camera.xxx`).
- Automation “Klingel: Gesicht erkennen” (doorbell: recognise face): doorbell sensor → wait up to 10 s for a new event
  image → recognise → push with image and `sentence`, optionally a voice announcement. For unknown people buttons
  “Ist <suggestion>”, “Name eingeben”, “Verwerfen” (is …, enter name, discard).
- Automation “Gesicht zuweisen” (assign face) handles the buttons (`mobile_app_notification_action`).
- Entities: `sensor.klingel_besucher` (e.g. “Anna” / “eine unbekannte Person”, attributes `satz`, `namen`, `bekannt`,
  `unbekannte`, `gesichter`, `zeitpunkt`) and `sensor.klingel_offene_gesichter` (REST, every 5 min and immediately after
  ringing/assigning/discarding).

Response of `/api/recognize`: `sentence`, `text`, `names`, `known`, `unknown_count`, `face_count`, `faces[]`
(`known`, `name`, `score`, `unknown_id`, `suggestion`, `crop`), `snapshot`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/recognize` | image (form field `image`, raw data) or `{"entity_id"}` |
| POST | `/api/stream_scan` | `{"entity_id", "duration", "stream_url"}` – analyse a stream for a few seconds |
| GET | `/api/unknowns` | open faces, grouped, with suggestion |
| POST | `/api/unknowns/{id}/assign` | `{"name"}` or `{"person_id"}`, `"group": false` = only this one |
| POST | `/api/unknowns/{id}/ignore` | discard (default: the whole group) |
| POST | `/api/recent/{id}/correct` | wrong person: `{"name"}` the right person, empty = unknown |
| GET/POST | `/api/persons` | people with faces / create `{"name"}` |
| PATCH/DELETE | `/api/persons/{id}` | rename (merge) / delete |
| POST | `/api/persons/{id}/photo` | train from a photo |
| DELETE | `/api/samples/{id}` | remove a wrong face |
| POST | `/api/samples/{id}/move` | give a face to another person `{"name"}` |
| GET | `/api/events` | history |
| GET | `/proxy-module`, `/proxy-module/memory` | module interface for the [OllamaProxy](https://github.com/Poldion/OllamaProxy) |

## Tests

`./run_tests.sh` (only needs Docker): builds the image and runs `tests/` with a simulated recogniser (34 tests:
recognising, assigning, grouping, correcting, cleaning up, tracking in the stream scan, module interface).
