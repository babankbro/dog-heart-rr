# Data, Web, and CLI

## Patient data model

`data/patients.json` holds registry metadata. Patient images live in folders keyed by validated patient ID. A logical patient contains:

- ID;
- display name;
- group/type;
- note;
- discovered images and image count.

Patient IDs are constrained before becoming paths. Image additions validate extension and actual decodability. Deleting a patient can retain or remove images; deleting images also invalidates associated saved results through the web layer.

The importer understands external folder trees shaped as `<group>/<id name>/<crop>`, cleans group names, separates ID and name, locates a crop subfolder case-insensitively, and supports per-patient limits or replacement.

## FastAPI responsibilities

The server provides:

- health and restricted configuration endpoints;
- image listing and upload;
- single-image detection;
- overlays, preprocessing previews, crops, masks, and CSV;
- patient CRUD, patient-image management, summaries, and batch analysis;
- flat-image migration;
- two-sided A/B debug datasets and rendered comparisons;
- static browser assets.

Only explicitly allowlisted configuration values can be overridden by the browser. Requested image paths are resolved beneath the configured data directory.

## Cache model

There are three related layers:

1. durable JSON results on disk;
2. an in-memory result cache for active images;
3. a bounded encoded-PNG cache for overlays and diagnostic panels.

Only a small number of decoded raw source images remain in memory. Numeric results remain available and raw images are reloaded when a render endpoint needs them.

The server is intentionally a single-machine application. Its caches and loaded models are process-global, not partitioned by authenticated user.

## Browser interface

The main page supports patient selection, groups, upload, whole-patient analysis, single-image analysis, summary cards, RR charts/tables, CSV download, overlay toggles, crop views, point masks, and live preprocessing previews.

The debug page maintains left/right image sets or selections from patient data. Each side keeps its own preprocessing configuration so operators can compare detection outcomes and model-domain appearance.

The frontend uses `fetch`, DOM APIs, and generated SVG. There is no Node or bundling step.

## CLI workflows

| Command | Role |
|---|---|
| `info` | Inspect weight input size and class names |
| `scale` | Estimate scale across images |
| `calibrate` | Measure point-model training scale from a sample |
| `detect` | Run detection over an image directory and export CSV |
| `import-tree` | Import an external grouped patient tree |
| `warm` | Precompute and persist results for every image |
| `domain` | Compare input appearance with saved training mosaics |

Global CLI arguments override weight paths, confidences, class ID, crop/point modes, preprocessing, and scale-related values.

