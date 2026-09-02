# Image Processing and Geometry

## Crop-model preprocessing

`crop_preprocess` supports several representations:

- `blackhat`: morphology emphasizing dark traces on light paper;
- `blackhat_otsu`: blackhat with automatic thresholding;
- `tophat_gray`: background estimate from closing, followed by gray difference thresholding;
- `adaptive`: local adaptive thresholding;
- `ink`: color-aware ink extraction;
- `red`: red-channel separation useful when the grid is printed in red.

Optional hysteresis connects faint pixels to a strong trace core. Closing joins short gaps, and dilation thickens retained lines. These operations change what the crop model sees and therefore must be evaluated as model-domain changes.

## Ink mask

The ink path separates dark, low-saturation waveform pixels from colored grid lines. Adaptive brightness limits prevent a fixed threshold from failing on pale scans. Full ink columns can be removed, small components can be discarded, and a continuity constraint prevents isolated grid pixels from becoming a peak.

`mask_quality` measures whether a candidate mask contains a plausible amount and distribution of ink. `keep_trace` preserves connected components large enough relative to the dominant trace. `solid_ink` applies run-length continuity.

## R-anchor extraction

The anchor region expands the detector box horizontally and vertically because a crop detector trained on thresholded input can place its upper edge below a faint R tip. The anchor algorithm:

1. builds a cleaned ink mask;
2. estimates a baseline within the region;
3. measures vertical trace extent per column;
4. requires sufficient continuous ink;
5. chooses the strongest upward or downward excursion;
6. returns debugging arrays when requested.

This anchor makes point weights optional and provides a geometric reference for rejecting implausible model refinements.

## Crop modes

| Mode | Frame basis | Intended use |
|---|---|---|
| `train_match` | row pitch and measured R amplitude | Reproduce measured training-dataset proportions |
| `mm` | paper scale and training px/mm | Give every crop the same physical coverage |
| `anchored` | padded detector box placed around R anchor | Approximate label-derived framing |
| `height` | box dimensions | Fallback when pitch/anchor evidence is insufficient |
| `pitch` | row beat spacing | Rhythm-relative square frame |
| `box` | detector-box width | Simple box-relative square frame |
| `stretch` | independently expanded box width/height | Preserve rectangular content then stretch to model size |

`train_match` falls back when it cannot derive the required pitch or anchor-debug geometry.

## Border handling and coordinate mapping

The requested crop dimensions remain fixed even at source-image borders. Missing content is either replicated or filled white. An optional strategy shifts the frame inside the source where possible.

The resized model crop stores:

- source top-left coordinates;
- x/y padding offsets;
- source pixels per model pixel in x and y.

`unmap_point` reverses this mapping. Separate scale factors are essential because some training crops were stretched to square.

## Row and distance geometry

- Boxes are grouped by y center before any RR reasoning.
- Row pitch uses the median distance between sorted box centers.
- Point deduplication uses a fraction of median RR.
- Expected R location is the recorded training anchor fraction for training-matched modes and the center for ordinary square modes.

