# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

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
