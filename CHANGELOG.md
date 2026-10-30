# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.1] – 2026-10-30

### Changed

- Donation link in the README

## [0.2.0] – 2026-10-28

### Added

- English user interface: language switch (DE/EN), defaults to the browser language and is remembered in the browser
- `FACEREC_LANG=en` for the announcement and push sentence ("Anna is at the door.")
- Error messages in the language of the request (`X-FaceRec-Lang` from the UI, otherwise `Accept-Language`)
- Translation check `scripts/check_i18n.py` (part of the tests): reports missing translations and wrong placeholders
- English version of the technical notes (`docs/technik.en.md`), linked from the English README

## [0.1.0] – 2026-10-28

### Added

- Initial public release.
- Face recognition with InsightFace `buffalo_l` via ONNX Runtime (CPU or AMD GPU via MIGraphX)
- Learning: assign unknown faces, grouping of similar faces, learning from photos, rename and merge people
- Corrections for wrongly recognized faces; recognized faces kept for 48 h only
- Stream scan of MJPEG cameras with face tracking
- Home Assistant package and automation template
- Module interface for OllamaProxy
- Test suite with a simulated recognizer
