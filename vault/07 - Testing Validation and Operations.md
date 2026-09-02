# Testing, Validation, and Operations

## Automated test framework

Most tests use synthetic EKG images and fake model results. This keeps the core suite deterministic and avoids requiring Torch or real weights. Coverage areas include:

| Area | Main tests |
|---|---|
| Preprocessing and anchors | `test_preprocess.py`, `test_binarize.py` |
| Crop geometry and mapping | `test_geometry.py` |
| Pipeline fusion and missing models | `test_pipeline.py` |
| Grid and scale | `test_grid.py`, `test_scale.py` |
| RR export | `test_export.py` |
| Rendering | `test_render.py` |
| Patient registry and import | `test_patients.py`, `test_importer.py` |
| Result persistence/revisions | `test_results.py` |
| Domain comparison | `test_domain.py` |
| API, cache, and UI contracts | `test_web.py`, `test_debug_page.py`, `test_ui_defaults.py` |
| Real weight integration | `test_weights.py` |

## Current local test finding

The non-web test run completed successfully with 169 collected test cases in the available environment. The web suite fails during `TestClient` construction before application assertions run.

Installed by the `pytest.exe` interpreter:

- FastAPI 0.104.1;
- Starlette 0.27.0;
- HTTPX 0.28.1;
- pytest 7.4.4.

Starlette 0.27 passes an `app=` argument that HTTPX 0.28 no longer accepts. The project requirements use broad lower bounds, so an unmanaged environment can resolve an incompatible combination. Container installation may resolve newer compatible versions, but reproducibility should be improved with constraints or a lock file.

## Validation gates

Before accepting a model or preprocessing change:

1. Run the fast synthetic suite.
2. Run API/UI tests in the supported dependency environment.
3. Run integration tests with actual weights.
4. Evaluate a versioned real-image dataset.
5. Compare detection and measurement metrics by subgroup.
6. Inspect overlays for every new failure category.
7. Confirm saved-result invalidation and restart restoration.
8. Update README and this vault with the exact chosen defaults.

## Local operation

- Web: build/start the Compose `web` service and open port 8000.
- Fast tests: use the Compose `test` profile.
- Full tests: use `test-full`, which includes runtime ML dependencies and mounted weights/data.
- CLI: use the Compose `app` profile so OpenCV, Torch, and Ultralytics versions are controlled.

## Container design

- `base`: Python, NumPy, OpenCV, package, web source, and tests.
- `test`: adds development/test requirements without Torch.
- `runtime`: adds CPU Torch, torchvision, Ultralytics, and the CLI entrypoint.
- `web`: adds FastAPI web dependencies and runs Uvicorn.

## Production operation

The production Compose setup and `deploy/` assets place Caddy in front of the application for TLS and optional authentication. Data, output, and model weights must be backed up independently of the container image.

Operational checks should include:

- `/api/health` reports both expected weights and saved-result count;
- data and output volumes are writable where required;
- weights are mounted read-only;
- available memory is adequate for both YOLO models and bounded image caches;
- results are backed up with their source images and configuration context;
- patient data is never committed to Git.

