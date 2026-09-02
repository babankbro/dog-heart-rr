# Measurement and Export

## Pixel scale

`scale.py` estimates pixels per millimeter from repeating EKG grid structure, with limits guarding against implausible periods. A configured `px_per_mm` can override detection. `check_scale` combines scale and beat pitch to estimate heart rate and reject physically implausible calibration.

## Major-grid model

`grid.py` emphasizes chromatic grid structure and finds major lines, normally representing 5 mm spacing. Refinement fits a periodic line model over many observations and records residual RMS and drift. This is more stable than relying on one small grid interval.

The grid supplies:

- line positions;
- major spacing in pixels;
- pixels per millimeter;
- number of measured lines;
- fit residual and drift;
- a grid-aligned horizontal origin.

## Coordinate conversion

When both origin and scale are available:

`x_mm = (x_px - origin_px) / px_per_mm`

For adjacent peaks in one row:

`RR_mm = delta_x_px / px_per_mm`

At configured paper speed:

`RR_seconds = RR_mm / paper_speed_mm_s`

`BPM = 60 / RR_seconds`

No interval is formed across row boundaries.

## RR summary

Per-row summaries include count, mean, standard deviation, median, minimum, maximum, seconds, and BPM. The image-level UI normally reports the row with the most beats as the main row.

## Tabular export

`result_to_rows` creates a stable record for each peak/interval. It includes image identity, row and peak index, pixel/millimeter position, RR units, BPM, model confidence/source, and a quality flag.

Quality checks use local RR context to identify intervals suggestive of a missed beat, duplicate detection, or other large deviation. These flags direct review; they are not diagnoses.

`write_csv` writes all rows using the defined `FIELDS` order. The web CSV endpoint exports the same rows associated with the displayed cached result.

## Durable result storage

`results.py` stores one JSON payload per image under `out/results/`. It serializes configuration, dimensions, boxes, rows, peaks, landmarks, grid data, RR summaries, and export rows, while excluding the large raw image array.

A saved result is valid only when:

- its schema is readable;
- the source image still exists;
- source modification time matches;
- the requested configuration matches, when a configuration is supplied.

The revision identifier hashes image name, modification time, and configuration. Rendered images can therefore use long-lived immutable browser caching without displaying stale evidence after a restart.

