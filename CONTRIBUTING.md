# Contributing

Thanks for wanting to help! Bug reports, ideas and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/Poldion/FaceRec.git && cd FaceRec
cp facerec.env.example ~/facerec/facerec.env       # HA_URL/HA_TOKEN optional for local tests
docker compose --env-file ~/facerec/facerec.env up -d --build
```

- `app.py` – FastAPI service (API, UI, stream scan, module interface)
- `engine.py` – face detection and embeddings via ONNX Runtime
- `tracker.py` – tracking faces across frames in the stream scan
- `homeassistant/` – Home Assistant package and automation template

## Tests

```bash
./run_tests.sh   # needs only Docker
```

The recognizer is simulated in the tests, so no real faces are needed. Please add a test in `tests/` for new behaviour.

## Guidelines

- Small, self-contained changes – rather several pull requests than one big one.
- Keep the style of the existing code: short comments that explain *why*. The UI and most comments are in German;
  English is fine for new comments.
- Commit messages: one short line saying what changed, details in the body if needed.
- **No personal data** in code, tests or issues: no photos of real people, real names, tokens, IP addresses or entity
  names of your own Home Assistant. Use made-up names and example values.
- Line endings LF (enforced by `.gitattributes`).

By contributing you agree that your contribution is licensed under the project's [MIT License](LICENSE).

## Deutsch

Beiträge auf Deutsch sind willkommen – Issues und Pull Requests dürfen gern deutsch sein. Die ausführliche Doku unter
`docs/` gibt es bisher nur auf Deutsch.
