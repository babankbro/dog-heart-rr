# Detection Pipeline

## Inputs and outputs

`detect_r_peaks(image_path, models, cfg)` accepts an image path, loaded model pair, and `Config`. It returns source image data, flat boxes, boxes grouped by row, final peaks, all landmarks, grid details, x-origin, per-row RR summaries, main-row selection, and statistics.

## Stage 1 — image loading and scale estimate

The image is decoded through `imread_u`, which supports paths that OpenCV may not open directly. `resolve_px_per_mm` either uses a configured scale or estimates it from the paper.

Failure to decode is a hard error. Failure to estimate scale is not: peak detection can continue, but millimeter, time, and BPM outputs may be unavailable.

## Stage 2 — beat-box detection

`detect_boxes` sends `crop_preprocess(raw, cfg)` to the crop YOLO using crop confidence, IoU, and image-size settings. Its `xyxy` boxes are rounded into source-image pixel coordinates.

No boxes produces a valid empty result, not a crash.

## Stage 3 — row formation and edge cleanup

`group_rows` clusters boxes by vertical center using median box height and a configured tolerance, then sorts each row left-to-right. This prevents RR calculation across separate leads.

`drop_edge_non_beats` examines boxes at row boundaries. It can remove calibration pulses or partial edge fragments based on amplitude, width, box size, and whether the candidate peak is too close to a box edge.

## Stage 4 — grid and coordinate origin

The pipeline finds major grid lines, optionally refines their periodic fit, and may replace the initial pixel scale with the line-derived value. The first retained row supplies a horizontal reference for choosing a nearby grid origin.

## Stage 5 — anchor and model crop creation

For each box:

1. `find_r_anchor` searches an expanded region for a connected ink peak.
2. If no anchor exists, the box center is used as a conservative fallback.
3. `square_crop` creates the model input according to the selected crop mode.
4. `point_preprocess` converts the crop into the representation expected by the point weights.
5. Mapping metadata records source offset, padding offset, and independent x/y scales.

## Stage 6 — optional point-landmark inference

When a point model exists and mode is not `anchor_only`, crops are predicted in configured batches. Every detected landmark is mapped back to the full image and retained for diagnostics.

`pick_point` selects the R candidate using class filtering, confidence, and distance from the expected training-frame position.

## Stage 7 — fusion policy

The anchor is the initial answer. A model point replaces it when one of these is true:

- point mode is `model_only`;
- the model point lies within `max_refine_ratio * row_pitch` of the anchor;
- model confidence is at least `trust_model_conf`.

A far but highly confident model point is accepted and counted separately. A low-confidence disagreement is rejected, leaving the anchor. In `model_only`, beats without an accepted model point are omitted.

## Stage 8 — deduplication

Overlapping beat crops can predict the same landmark. Diagnostic landmarks are deduplicated within class and row. Final R-peaks are deduplicated per row using a minimum distance proportional to median RR; the higher-confidence point survives.

## Stage 9 — measurements and result assembly

Each peak receives row/index, x/y, confidence, source, class, and optional millimeter x-position. Per-row RR summaries are derived from ordered x coordinates. The main row is the row containing the most retained beat boxes.

Important counters include:

- boxes, rows, and peaks;
- model-accepted and anchor-derived peaks;
- rejected refinements and trusted far refinements;
- duplicate peaks and landmarks;
- dropped edge boxes;
- scale and grid-fit details.

## Failure behavior

| Condition | Behavior |
|---|---|
| Missing crop weights | Model loading fails clearly |
| Missing point weights | Continue in anchor-capable mode |
| Unreadable image | Raise a file/read error |
| No boxes | Return an empty valid result |
| No grid/scale | Retain pixel peaks; omit physical metrics |
| Anchor missing | Start from box center fallback |
| Point disagrees weakly | Reject point and keep anchor |
| Overlapping predictions | Deduplicate within each row |

