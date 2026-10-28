# Third-party software and models

This repository contains original code under the [MIT License](LICENSE). The following third-party components are
used. None of them is included in the repository; they are fetched from their official sources when the image is
built.

## Runtime software

| Component | License | Note |
|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | GPU variant: `onnxruntime-migraphx` by AMD |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | |
| [OpenCV](https://github.com/opencv/opencv-python) (headless) | Apache-2.0 (bindings: MIT) | |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | |
| [Starlette](https://github.com/encode/starlette) | BSD-3-Clause | |
| [Uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause | |
| [HTTPX](https://github.com/encode/httpx) / httpcore | BSD-3-Clause | |
| [python-multipart](https://github.com/Kludex/python-multipart) | Apache-2.0 | |
| [ROCm](https://github.com/ROCm/ROCm) / [PyTorch](https://github.com/pytorch/pytorch) | mostly MIT/Apache-2.0/BSD, partly proprietary | GPU variant only, via the image `rocm/pytorch` |

## Models

| Model | Source | License (per source) | Note |
|---|---|---|---|
| InsightFace `buffalo_l` (SCRFD `det_10g`, ArcFace `w600k_r50`) | [deepinsight/insightface](https://github.com/deepinsight/insightface), release v0.7 | non-commercial research only | **No commercial use** without a license from InsightFace |

The model is downloaded when the image is built (checksum in the Dockerfile) and is not included. This is a help, not
legal advice; the license of the source is authoritative.

## Images

The people in `docs/bilder/facerec.png` are fictional; their portraits were generated with ImageGen. The screenshot is
licensed under this repository's MIT License.
