# Configuration and Decisions

## Configuration groups

### Model contract

- crop and point weight paths;
- confidence and IoU thresholds;
- inference image sizes;
- R landmark class ID;
- point operating mode and batch size.

### Preprocessing contract

- crop representation and its thresholds/kernels;
- hysteresis, closing, and dilation;
- point representation;
- ink brightness, saturation, adaptive limits, connected-component, and run-length rules.

### Geometry contract

- crop mode and output size;
- padding behavior;
- training physical scale;
- training R anchor fractions;
- training frame width/height ratios;
- model-anchor agreement and deduplication distances;
- row grouping tolerance.

### Measurement contract

- optional fixed pixels per millimeter;
- grid spacing and refinement controls;
- paper speed;
- plausible heart-rate range;
- RR flag thresholds and local window.

## Settings that must move together

| Coupled set | Reason |
|---|---|
| Point weights + `point_pre` + class map | Input representation and class IDs are learned properties |
| Crop weights + `crop_pre` + crop `imgsz` | The box model depends on its training image domain |
| Crop mode + training scale + anchor fractions + frame ratios | Together they reproduce the point model's training frame |
| Grid millimeters + paper speed | Together convert pixels into time and BPM |
| Dedup/refinement ratios + typical pitch | These thresholds are relative to rhythm geometry |

## Current notable defaults

- point weights: `models/point_ink_best.pt`;
- point preprocessing: `ink`;
- R class: `5`;
- crop confidence: `0.20`;
- point confidence: `0.05`;
- crop mode: `train_match`;
- point mode: hybrid;
- vertical anchor expansion: `0.35`;
- model input sizes: 512.

Always read `Config` for the complete authoritative values.

## Open documentation decision

The current code sets `crop_pre='tophat_red'` and `point_pre='red_ink'`. Neither matches the training domain of its model — `blackhat` and `ink` do. Both departures are deliberate and measured; see [[10 - ผลการทดลอง Preprocessing]] for the comparison tables and the limitations that apply to them.

This should be resolved explicitly:

1. decide whether production prioritizes training-domain fidelity or the measured recall gain;
2. rerun the complete evaluation on the named 67-image set and relevant subgroups;
3. record false positives, missed beats, R-tip truncation, and downstream RR/BPM impact;
4. make `Config`, README, UI defaults, tests, and this vault agree.

## Technical risks

- Domain mismatch can improve box recall while degrading box geometry or generalization.
- Global process caches mean the web application is not designed for isolated multi-user workloads.
- Broad dependency ranges currently permit an incompatible Starlette/HTTPX test environment.
- Image modification time is part of invalidation; operational copying practices must preserve or intentionally update it.
- Patient files are local filesystem data and require external access control, backup, and retention policies.
- Metrics described in documentation are experiment-specific and should not be interpreted as clinical validation.

## Suggested next work

1. Resolve and document the production crop-preprocessing default.
2. Add a reproducible dependency constraints/lock strategy.
3. Create a versioned, de-identified evaluation manifest with ground truth.
4. Add a repeatable evaluation command that emits box, peak, RR, and subgroup metrics.
5. Define acceptance thresholds and a model/config version identifier.
6. Add privacy, backup, and retention procedures for patient data.
7. Keep this vault updated in the same change that modifies architecture or defaults.

