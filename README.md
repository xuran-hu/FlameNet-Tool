# FlameNet on Earth Engine — Sentinel-1

A browser deployment of **FlameNet**, the Sentinel-1 burned-area model. Earth
Engine supplies a pre/post SAR pair, `onnxruntime-web` runs the trained network
on it, and the map is produced without any imagery or result leaving the
machine.

This directory is a fork of `../wildfire-earth-engine/` (FlameNet-S2), which is
itself a fork of Ali Shibli's wildfire tool; `style.css` still imports that
stylesheet from `../wildfire-demo/`. What changed here is the sensor, and with
it everything the sensor implies.

```
index.html      the page
samples.html    static results for six 2025 fires, no account needed
app.js          Earth Engine queries, the inference loop, rendering
pipeline.js     the numeric core, with no DOM or network: testable in node
model/          the exported graphs and the preprocessing constants
samples/        the gallery's imagery
test/           node tests against the research implementation
```

## What the model requires, and what this reproduces

| | FlameNet (Sentinel-1) |
|---|---|
| Collection | `COPERNICUS/S1_GRD`, IW mode, dual-polarisation VV+VH |
| Radiometry | **σ⁰ in decibels**, as that collection delivers it: thermal-noise removed, calibrated, Range-Doppler terrain corrected on SRTM. **No terrain flattening (so not γ⁰) and no speckle filter** — the model was trained on exactly this |
| Bands | `VH, VV` per phase, in that order. Two channels, not four: radar needs no cloud band |
| Standardisation | the 2017–2023 training means and stds, −13.38 / −6.54 dB and 4.21 / 3.87 |
| No-data | pixels outside the swath are filled with the channel mean, which is zero after standardising — the loader's own rule |
| Grid | 20 m in the scene's local **UTM** zone; a 20 m Web Mercator pixel is 11 m on the ground at 56° N |
| Overpass | one date **and one relative orbit** is one scene; its granules are mosaicked. The training archive mosaicked by date alone, which mixes two geometries on 3 % of its tiles |
| Pre-fire scene | the **previous year's fire season**. A lookback of weeks is wrong in Canada: many fires start in April–May, so the pre-image would be frozen or snow-covered ground and every pixel would read as change |
| Orbit matching | the pre scene is taken from the **post scene's relative orbit** when that orbit flew in the search window (see below) |
| Inference | 256 px patches, stride 128, Gaussian-blended, with the tile context encoder's output sampled at each patch's footprint, then components under 50 px dropped |

Every one of those constants is read at page load from `model/model_meta.json`,
which `scripts/export_onnx.py` in the FLAMENet repository writes directly from
the training code. Nothing about the model is duplicated here as a literal.

### Orbit matching is not cosmetic

On the held-out 2025 season, fires whose pre and post scenes share a relative
orbit score **0.489 event-averaged IoU**; fires whose scenes come from
different orbits score **0.379**. The gap survives stratification by fire size,
and it is not specific to this model — every baseline except the one that
barely works loses accuracy the same way. Only 67 % of the training archive's
pairs match, so this page does better than the archive where the data allow,
and says which orbits it used either way.

### What this page cannot do that the Sentinel-2 page can

Report obscured pixels. Radar sees through cloud, so the only unseen pixels are
those outside the swath; those are filled with the channel mean and will read
as unburned. The no-data fraction of each selected scene is reported instead.

## Samples

`samples.html` shows six fires of the 2025 test season — the Sentinel-1 pair,
the NBAC reference perimeter and the prediction, with the per-tile IoU the
paper reports. It is a starting point, not the evidence: the gallery is
filtered to fires at or above IoU 0.5 (`make_web_gallery.py --min-iou`), while
the demonstrator itself offers all 32 presets, whose notes carry the IoU each
one scored across the full range of the season, and any coordinates and dates
a reader enters. Each card links into the demonstrator with that fire
preselected (`./?preset=CA_2025_…`).

## The preset list

32 fires from the 2025 test season, grouped by burned area, spanning
**176 ha to 142,975 ha**, each carrying the pre and post dates the dataset
loader itself selected for that tile — so running a preset reproduces the pair
the reported metrics came from, and the note on it says what IoU to expect. The
last group pairs scenes from **different relative orbits**, which is what a
third of the training archive looks like.

Window sizes are snapped to the slider's 2.56 km step from 17.92 km, because a
value off that grid is silently snapped when the preset is selected and would
segment a different area than the one the number was measured on.

## Tests

```
npm install          # onnxruntime-web, for the end-to-end test only
npm test
```

`test/pipeline.test.mjs` checks each transcribed numeric function against
vectors taken from the research code. `test/wiring.test.mjs` checks the page
against itself: every element id, the cache-busting version, the preset table's
shape, the model directory. `test/endtoend.test.mjs` runs the whole window loop
over the real ONNX graphs and **skips itself** until a fixture is generated —
the Sentinel-1 fixture has not been produced yet, so that check is currently
not running here.
