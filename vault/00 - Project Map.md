# Project Map

## Purpose

The project analyzes scanned or photographed veterinary EKG paper. It detects heartbeat regions and R-peaks, converts their positions into physical and temporal measurements, reports RR intervals and heart rate, and gives the operator visual evidence for checking each result.

The system is deliberately hybrid:

- a crop YOLO model finds candidate beats;
- image processing finds a robust R-peak anchor;
- an optional landmark YOLO model refines the anchor;
- grid analysis converts pixels into millimeters and seconds;
- quality rules flag suspicious intervals rather than silently accepting them.

## Vault outline

1. [[01 - Methodology]] — scientific and engineering method used for this work.
2. [[02 - Overall Framework]] — system boundaries, components, and end-to-end flow.
3. [[03 - Detection Pipeline]] — detailed algorithm from image input to final peaks.
4. [[04 - Image Processing and Geometry]] — binarization, anchor extraction, crops, rows, and mapping.
5. [[05 - Measurement and Export]] — grid scale, RR, BPM, flags, CSV, and saved results.
6. [[06 - Data Web and CLI]] — patient data model, APIs, browser UI, CLI, and debugging tools.
7. [[07 - Testing Validation and Operations]] — test strategy, deployment, monitoring, and operational checks.
8. [[08 - Configuration and Decisions]] — configuration groups, coupled settings, known mismatches, risks, and future work.
9. [[09 - พื้นฐานและวิวัฒนาการของ YOLO]] — หลักการตรวจจับวัตถุ วิวัฒนาการของตระกูล YOLO และสถาปัตยกรรม YOLO11 ที่ใช้ในงานวิจัย
10. [[10 - ผลการทดลอง Preprocessing]] — ผลการทดลองเปรียบเทียบ binarization ก่อนตรวจกรอบ และ preprocessing ก่อนหาจุด R พร้อมวิธีดำเนินการของแต่ละวิธี

## Main source map

| Area | Source |
|---|---|
| Central configuration | `ekg_rpeak/config.py` |
| Preprocessing and anchors | `ekg_rpeak/preprocess.py` |
| Crop and point geometry | `ekg_rpeak/geometry.py` |
| Two-stage detection | `ekg_rpeak/pipeline.py` |
| Paper scale | `ekg_rpeak/scale.py` |
| Major grid lines and RR summaries | `ekg_rpeak/grid.py` |
| Tabular output | `ekg_rpeak/export.py` |
| Visual evidence | `ekg_rpeak/render.py` |
| Patient registry | `ekg_rpeak/patients.py` |
| External folder import | `ekg_rpeak/importer.py` |
| Persistent result cache | `ekg_rpeak/results.py` |
| Domain comparison | `ekg_rpeak/domain.py` |
| Command-line entry point | `ekg_rpeak/cli.py` |
| HTTP API and runtime cache | `webapp/server.py` |
| Browser client | `webapp/static/` |
| Automated checks | `tests/` |
| Containers and production proxy | `Dockerfile`, Compose files, `deploy/` |

## Current implementation status

- The working tree contains a large, uncommitted development set on top of `main`.
- All three local weight files are present: crop, point, and ink-point models.
- Core non-web tests pass in the currently available test environment.
- Web tests are presently blocked by an installed Starlette/HTTPX compatibility mismatch; see [[07 - Testing Validation and Operations]].
- Crop preprocessing now defaults to `tophat_red` and point preprocessing to `red_ink`; both were chosen from the measurements reported in [[10 - ผลการทดลอง Preprocessing]].
