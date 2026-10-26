/*
 * FlameNet in the browser.
 *
 * Earth Engine supplies a pre/post Sentinel-1 pair, onnxruntime-web runs the
 * trained network over it, and the burned-area map comes back without anything
 * leaving the machine. The point of the exercise is that the numbers match the
 * research pipeline, so every preprocessing constant comes from
 * ./model/model_meta.json, which scripts/export_onnx.py writes from
 * src/datasets/sar_fire.py at export time. Nothing about the model is
 * duplicated here as a literal.
 *
 * What the model expects, and therefore what this reproduces exactly:
 *
 *   collection   COPERNICUS/S1_GRD, IW mode, dual-polarisation VV+VH. That
 *                collection is already sigma-nought in DECIBELS, terrain
 *                corrected on SRTM; the model was trained on exactly it, with
 *                no terrain flattening and no speckle filter.
 *   bands        VH, VV in that order, per phase. Two channels, not four:
 *                radar has no cloud band and needs none.
 *   values       dB, standardised with the 2017-2023 training means and stds
 *                (-13.38 / -6.54 dB, 4.21 / 3.87). Pixels outside the swath
 *                are filled with the channel mean, which is zero after
 *                standardisation -- the loader's own rule.
 *   grid         20 m in the local UTM zone. A 20 m Web Mercator grid is 20 m
 *                only at the equator: at 56 N it is 11 m on the ground, and the
 *                model would be shown imagery at 1.8x its training resolution.
 *   overpass     one date and one relative orbit is one scene; granules of that
 *                overpass are mosaicked. The archive mosaicked by date alone,
 *                which silently mixes two geometries on 3% of its tiles.
 *   pre scene    the PREVIOUS YEAR'S FIRE SEASON, not a lookback of weeks: a
 *                spring pre-image in Canada is frozen or snow-covered ground
 *                and every pixel reads as change. The pre date is supplied
 *                rather than derived, so the presets carry the one the dataset
 *                loader selected for each fire.
 *   orbit        the pre scene is taken from the POST scene's relative orbit
 *                whenever that orbit was acquired in the search window. Pairs
 *                that match score 0.489 event-averaged IoU on the 2025 season
 *                against 0.379 for pairs that do not, so this is not cosmetic;
 *                the page says which orbits it used.
 *   inference    256 px patches, stride 128, Gaussian-blended, with the tile
 *                context encoder's output sampled at each patch's footprint,
 *                then components under 50 px dropped.
 *
 * What this cannot do that the Sentinel-1 page can: report obscured pixels.
 * Radar sees through cloud, so the only unseen pixels are those outside the
 * swath, and those are filled with the channel mean and will read as unburned.
 * The no-data fraction of each selected scene is reported instead.
 */

import { fromArrayBuffer } from "https://cdn.jsdelivr.net/npm/geotiff@2.1.3/+esm";
import {
  blendWindow, blitChannel, chunkEdges, convertChannel, extractPatch,
  gaussianWeight, normaliseBlend, removeSmallComponents, sarColour, sarRamp,
  sampleContext, standardise, standardiseInto, windowStarts,
// ?v= must match index.html's. A fresh app.js next to a cached pipeline.js is
// worse than both being stale: the numeric core and its caller would disagree.
} from "./pipeline.js?v=4";

const MODEL_DIR = "./model/";
const ORT_ASSET_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const EE_SCOPE = "https://www.googleapis.com/auth/earthengine.readonly";

// Earth Engine refuses a computePixels response over 32 MiB. The detail window
// is eight bands of int16, so ~1024 px square is the most one request can carry
// (17 MB, with headroom); the reflectance DN are integers in the archive anyway
// and dB quantised to 0.01 moves a standardised channel by 2.4e-3.
const DETAIL_CHUNK_PX = 1024;
// The window itself is tiled across as many requests as it needs, so the real
// ceiling is browser memory, not the response cap. At 4096 px the eight float32
// planes are 537 MB before the accumulators, which is why the reflectance is
// standardised in place and why the slider warns past 2048 px.
const DETAIL_MAX_PX = 4096;
// dB x 100 as int16: +-327 dB of range at 0.01 dB, which moves a
// standardised channel by 0.0024. Float32 would double every response.
const DB_QUANT = 100;
const NODATA = -9999;

// Phase-prefixed band names in the order both requests return them.
const PHASES = ["pre", "post"];

// Preset fires, all from the held-out 2025 season of the NBAC Canada archive.
// Every number here is what the research loader derives for the tile rather
// than something typed in: the centre of the burned bounding box, the smallest
// window that holds it, and the two acquisitions sar_fire._collect_triplets
// pairs for that fire. Sizes run 176 ha to 55,504 ha; nothing larger is offered
// because nothing larger fits a 81.92 km window, and a preset that cannot show
// the whole fire is a bad demonstration.
const SCENE_PRESETS = Object.freeze({
  CA_2025_SK_429: {
    label: "2025 · SK_429 · 142,975 ha", group: "Over 10,000 ha",
    latitude: 56.089, longitude: -103.72,
    detailKm: 79.36,
    pre: "2025-04-12", target: "2025-09-03",
    note: "SK · NBAC 142,975 ha · orbit 107 · reported IoU 0.79",
  },
  CA_2025_SK_354: {
    label: "2025 · SK_354 · 56,369 ha", group: "Over 10,000 ha",
    latitude: 56.7566, longitude: -109.2803,
    detailKm: 53.76,
    pre: "2024-07-02", target: "2025-10-07",
    note: "SK · NBAC 56,369 ha · orbit 78 · reported IoU 0.71",
  },
  CA_2025_NT_523: {
    label: "2025 · NT_523 · 55,504 ha", group: "Over 10,000 ha",
    latitude: 62.204, longitude: -112.8975,
    detailKm: 51.2,
    pre: "2024-06-30", target: "2025-11-04",
    note: "NT · NBAC 55,504 ha · orbit 49 · reported IoU 0.75",
  },
  CA_2025_SK_702: {
    label: "2025 · SK_702 · 34,511 ha", group: "Over 10,000 ha",
    latitude: 55.0875, longitude: -102.8676,
    detailKm: 48.64,
    pre: "2025-04-12", target: "2025-09-15",
    note: "SK · NBAC 34,511 ha · orbit 107 · reported IoU 0.57",
  },
  CA_2025_MB_70: {
    label: "2025 · MB_70 · 32,324 ha", group: "Over 10,000 ha",
    latitude: 51.672, longitude: -95.6824,
    detailKm: 38.4,
    pre: "2024-06-03", target: "2025-12-19",
    note: "MB · NBAC 32,324 ha · orbit 12 · reported IoU 0.47",
  },
  CA_2025_ON_646: {
    label: "2025 · ON_646 · 29,563 ha", group: "Over 10,000 ha",
    latitude: 51.5721, longitude: -93.811,
    detailKm: 35.84,
    pre: "2024-07-04", target: "2025-10-27",
    note: "ON · NBAC 29,563 ha · orbit 114 · reported IoU 0.85",
  },
  CA_2025_NT_210: {
    label: "2025 · NT_210 · 20,721 ha", group: "Over 10,000 ha",
    latitude: 63.8262, longitude: -120.535,
    detailKm: 35.84,
    pre: "2024-06-04", target: "2025-10-03",
    note: "NT · NBAC 20,721 ha · orbit 20 · reported IoU 0.67",
  },
  CA_2025_MB_34: {
    label: "2025 · MB_34 · 15,423 ha", group: "Over 10,000 ha",
    latitude: 52.7363, longitude: -95.7213,
    detailKm: 28.16,
    pre: "2025-04-04", target: "2025-11-06",
    note: "MB · NBAC 15,423 ha · orbit 165 · reported IoU 0.62",
  },
  CA_2025_ON_630: {
    label: "2025 · ON_630 · 9,012 ha", group: "1,000 to 10,000 ha",
    latitude: 52.3126, longitude: -93.0555,
    detailKm: 25.6,
    pre: "2025-05-16", target: "2025-09-13",
    note: "ON · NBAC 9,012 ha · orbit 92 · reported IoU 0.52",
  },
  CA_2025_YT_397: {
    label: "2025 · YT_397 · 5,040 ha", group: "1,000 to 10,000 ha",
    latitude: 66.7011, longitude: -134.4777,
    detailKm: 20.48,
    pre: "2024-07-16", target: "2025-08-16",
    note: "YT · NBAC 5,040 ha · orbit 108 · reported IoU 0.82",
  },
  CA_2025_YT_379: {
    label: "2025 · YT_379 · 4,799 ha", group: "1,000 to 10,000 ha",
    latitude: 64.1749, longitude: -135.0947,
    detailKm: 17.92,
    pre: "2024-06-20", target: "2025-07-09",
    note: "YT · NBAC 4,799 ha · orbit 79 · reported IoU 0.66",
  },
  CA_2025_YT_420: {
    label: "2025 · YT_420 · 4,564 ha", group: "1,000 to 10,000 ha",
    latitude: 62.8807, longitude: -139.5503,
    detailKm: 17.92,
    pre: "2024-06-22", target: "2025-09-09",
    note: "YT · NBAC 4,564 ha · orbit 116 · reported IoU 0.71",
  },
  CA_2025_BC_575: {
    label: "2025 · BC_575 · 4,453 ha", group: "1,000 to 10,000 ha",
    latitude: 52.5365, longitude: -125.9503,
    detailKm: 17.92,
    pre: "2024-09-09", target: "2025-12-03",
    note: "BC · NBAC 4,453 ha · orbit 35 · reported IoU 0.55",
  },
  CA_2025_AB_467: {
    label: "2025 · AB_467 · 3,746 ha", group: "1,000 to 10,000 ha",
    latitude: 58.9177, longitude: -118.236,
    detailKm: 17.92,
    pre: "2024-08-15", target: "2025-08-28",
    note: "AB · NBAC 3,746 ha · orbit 20 · reported IoU 0.81",
  },
  CA_2025_PC_655: {
    label: "2025 · PC_655 · 3,695 ha", group: "1,000 to 10,000 ha",
    latitude: 59.7082, longitude: -112.0757,
    detailKm: 17.92,
    pre: "2024-06-06", target: "2025-06-25",
    note: "PC · NBAC 3,695 ha · orbit 49 · reported IoU 0.78",
  },
  CA_2025_AB_510: {
    label: "2025 · AB_510 · 3,234 ha", group: "1,000 to 10,000 ha",
    latitude: 55.8693, longitude: -110.2976,
    detailKm: 20.48,
    pre: "2025-04-22", target: "2025-09-13",
    note: "AB · NBAC 3,234 ha · orbit 78 · reported IoU 0.62",
  },
  CA_2025_NT_176: {
    label: "2025 · NT_176 · 2,579 ha", group: "1,000 to 10,000 ha",
    latitude: 67.347, longitude: -135.1636,
    detailKm: 17.92,
    pre: "2024-06-08", target: "2025-09-19",
    note: "NT · NBAC 2,579 ha · orbit 79 · reported IoU 0.58",
  },
  CA_2025_MB_19: {
    label: "2025 · MB_19 · 2,374 ha", group: "1,000 to 10,000 ha",
    latitude: 54.3324, longitude: -95.5986,
    detailKm: 17.92,
    pre: "2025-04-16", target: "2025-12-12",
    note: "MB · NBAC 2,374 ha · orbit 165 · reported IoU 0.38",
  },
  CA_2025_AB_495: {
    label: "2025 · AB_495 · 1,841 ha", group: "1,000 to 10,000 ha",
    latitude: 56.7868, longitude: -117.201,
    detailKm: 17.92,
    pre: "2024-04-29", target: "2025-10-15",
    note: "AB · NBAC 1,841 ha · orbit 20 · reported IoU 0.74",
  },
  CA_2025_YT_390: {
    label: "2025 · YT_390 · 1,772 ha", group: "1,000 to 10,000 ha",
    latitude: 66.5566, longitude: -136.5145,
    detailKm: 17.92,
    pre: "2024-07-02", target: "2025-07-09",
    note: "YT · NBAC 1,772 ha · orbit 79 · reported IoU 0.45",
  },
  CA_2025_NT_189: {
    label: "2025 · NT_189 · 832 ha", group: "Under 1,000 ha",
    latitude: 63.9591, longitude: -114.2221,
    detailKm: 17.92,
    pre: "2024-06-06", target: "2025-10-11",
    note: "NT · NBAC 832 ha · orbit 49 · reported IoU 0.65",
  },
  CA_2025_SK_306: {
    label: "2025 · SK_306 · 528 ha", group: "Under 1,000 ha",
    latitude: 57.2131, longitude: -108.1599,
    detailKm: 17.92,
    pre: "2024-06-08", target: "2025-07-09",
    note: "SK · NBAC 528 ha · orbit 78 · reported IoU 0.48",
  },
  CA_2025_BC_592: {
    label: "2025 · BC_592 · 462 ha", group: "Under 1,000 ha",
    latitude: 49.3012, longitude: -124.6009,
    detailKm: 17.92,
    pre: "2024-08-04", target: "2025-11-09",
    note: "BC · NBAC 462 ha · orbit 35 · reported IoU 0.34",
  },
  CA_2025_BC_533: {
    label: "2025 · BC_533 · 450 ha", group: "Under 1,000 ha",
    latitude: 52.4373, longitude: -120.7891,
    detailKm: 17.92,
    pre: "2024-09-07", target: "2025-12-07",
    note: "BC · NBAC 450 ha · orbit 13 · reported IoU 0.25",
  },
  CA_2025_NL_947: {
    label: "2025 · NL_947 · 406 ha", group: "Under 1,000 ha",
    latitude: 55.1542, longitude: -60.4353,
    detailKm: 17.92,
    pre: "2024-08-11", target: "2025-08-18",
    note: "NL · NBAC 406 ha · orbit 149 · reported IoU 0.59",
  },
  CA_2025_MB_25: {
    label: "2025 · MB_25 · 342 ha", group: "Under 1,000 ha",
    latitude: 53.3665, longitude: -95.2663,
    detailKm: 17.92,
    pre: "2025-04-16", target: "2025-12-12",
    note: "MB · NBAC 342 ha · orbit 165 · reported IoU 0.08",
  },
  CA_2025_NT_171: {
    label: "2025 · NT_171 · 308 ha", group: "Under 1,000 ha",
    latitude: 60.5652, longitude: -121.0212,
    detailKm: 17.92,
    pre: "2024-06-02", target: "2025-10-07",
    note: "NT · NBAC 308 ha · orbit 166 · reported IoU 0.76",
  },
  CA_2025_SK_341: {
    label: "2025 · SK_341 · 176 ha", group: "Under 1,000 ha",
    latitude: 56.6371, longitude: -109.8194,
    detailKm: 17.92,
    pre: "2024-09-12", target: "2025-10-07",
    note: "SK · NBAC 176 ha · orbit 78 · reported IoU 0.00",
  },
  CA_2025_NT_688: {
    label: "2025 · NT_688 · 128 ha", group: "Under 1,000 ha",
    latitude: 67.3419, longitude: -135.3086,
    detailKm: 17.92,
    pre: "2024-07-02", target: "2025-09-19",
    note: "NT · NBAC 128 ha · orbit 79 · reported IoU 0.20",
  },
  CA_2025_AB_486: {
    label: "2025 · AB_486 · 117 ha", group: "Under 1,000 ha",
    latitude: 59.1094, longitude: -115.2925,
    detailKm: 17.92,
    pre: "2024-07-24", target: "2025-07-19",
    note: "AB · NBAC 117 ha · orbit 49 · reported IoU 0.42",
  },
  CA_2025_MB_36: {
    label: "2025 · MB_36 · 2,556 ha", group: "Different relative orbits",
    latitude: 56.2416, longitude: -99.6141,
    detailKm: 17.92,
    pre: "2025-05-20", target: "2025-09-24",
    note: "MB · NBAC 2,556 ha · orbit 63 post / 136 pre — MISMATCHED · reported IoU 0.89",
  },
  CA_2025_SK_359: {
    label: "2025 · SK_359 · 1,460 ha", group: "Different relative orbits",
    latitude: 56.6955, longitude: -104.0843,
    detailKm: 17.92,
    pre: "2025-04-12", target: "2025-08-17",
    note: "SK · NBAC 1,460 ha · orbit 34 post / 107 pre — MISMATCHED · reported IoU 0.90",
  },
});


/* Every preset is a fire of the held-out 2025 season, carrying the pre- and
 * post-fire dates the dataset loader itself chose for that tile, so running one
 * reproduces the pair the reported metrics came from. The IoU in each note is
 * what checkpoint ep295 scored on that fire at threshold 0.5. They are spread
 * across the accuracy range of their size class on purpose -- the worst of them
 * returns almost nothing -- because a preset list drawn from the top would
 * misrepresent the model. The last group pairs scenes from different relative
 * orbits, which is what a third of the training archive looks like.
 */

const state = {
  meta: null,
  eeConnected: false,
  authPopupReady: false,
  authAttempt: 0,
  ctxSession: null,
  patchSession: null,
  inferenceEngine: null,
  pre: null,          // detail window, reflectance units
  post: null,
  ctxPre: null,       // 256 px context thumbnail, same units
  ctxPost: null,
  probability: null,
  keep: null,         // null when the run is 3-channel or masking is disabled
  geometry: null,
  selected: null,
  sourceBlobs: null,   // one per computePixels request the window needed
  map: null,
  marker: null,
  detailRect: null,
};

const elements = {};
for (const id of [
  "project-id", "client-id", "connect-button", "ee-status", "connection-message",
  "scene-preset", "preset-meta", "latitude", "longitude", "detail-size",
  "detail-value", "pre-date", "target-date",
  "pre-window", "target-window", "map-workspace", "map-expand", "map-coordinate",
  "run-button", "progress-wrap", "progress-text", "progress-value", "progress-bar",
  "message", "results", "threshold", "threshold-value", "pre-date-summary",
  "pre-clear-summary", "current-date-summary", "current-clear-summary",
  "lag-summary", "engine-summary", "burned-summary", "pixel-summary",
  "obscured-summary", "obscured-detail", "window-cost", "pre-canvas", "current-canvas",
  "probability-canvas", "mask-canvas", "download-mask", "download-stack",
  "setup-help", "page-origin", "during-date", "during-date-field",
]) {
  elements[id.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = document.querySelector(`#${id}`);
}

function setMessage(message, error = false, connection = false) {
  const target = connection ? elements.connectionMessage : elements.message;
  target.textContent = message;
  target.style.color = error ? "#8c3218" : "#52605b";
}

function setEeStatus(label, mode = "") {
  elements.eeStatus.className = `engine-status ${mode}`.trim();
  elements.eeStatus.querySelector("b").textContent = label;
}

function setProgress(fraction, message) {
  const value = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  elements.progressWrap.hidden = false;
  elements.progressBar.style.width = `${value}%`;
  elements.progressValue.textContent = `${value}%`;
  elements.progressText.textContent = message;
}

function normalizeError(error) {
  if (typeof error === "string") return error;
  const messages = [
    error?.message,
    error?.details,
    error?.error_description,
    typeof error?.error === "string" ? error.error : error?.error?.message,
    error?.result?.error?.message,
  ].filter(Boolean);
  if (messages.length) return [...new Set(messages)].join(" — ");
  try {
    return JSON.stringify(error);
  } catch {
    return "Unknown error";
  }
}

// ── model metadata ───────────────────────────────────────────────────────────

async function loadMeta() {
  const response = await fetch(`${MODEL_DIR}model_meta.json`);
  if (!response.ok) throw new Error(`model_meta.json is missing (${response.status}).`);
  const meta = await response.json();
  state.meta = meta;
  elements.threshold.value = meta.threshold;
  elements.thresholdValue.value = Number(meta.threshold).toFixed(2);
  return meta;
}

// ── map ──────────────────────────────────────────────────────────────────────

function initializeMap() {
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  state.map = window.L.map("map", { zoomControl: true }).setView([latitude, longitude], 9);
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  const icon = window.L.divIcon({ className: "roi-center-marker", iconSize: [15, 15] });
  state.marker = window.L.marker([latitude, longitude], { icon, interactive: false }).addTo(state.map);
  state.detailRect = window.L.rectangle(squareBounds(latitude, longitude, detailKm()), {
    color: "#ff6b35", weight: 2, fillColor: "#ff6b35", fillOpacity: 0.08,
  }).addTo(state.map);
  state.map.on("click", event => {
    elements.latitude.value = event.latlng.lat.toFixed(5);
    elements.longitude.value = event.latlng.lng.toFixed(5);
    markSceneCustom("Custom scene · map centre set manually; set its dates at right.");
    updateMapSelection(false);
  });
}

const detailKm = () => Number(elements.detailSize.value);
// The tile context encoder saw exactly the area being segmented — in training a
// patch grid tiled a whole fire tile and the thumbnail was that same tile. A
// context wider than the window is therefore not "more context", it is a
// different distribution: on one 71.68 km run, stretching it to 100 km moved
// the burned area from 234,774 ha to 257,344 ha, and shrinking it below the
// window collapsed the periphery altogether. So it is not a control.
const contextKm = () => detailKm();

function squareBounds(latitude, longitude, sizeKm) {
  const halfKm = sizeKm / 2;
  const dLat = halfKm / 111.32;
  const dLon = halfKm / (111.32 * Math.max(0.15, Math.cos(latitude * Math.PI / 180)));
  return [[latitude - dLat, longitude - dLon], [latitude + dLat, longitude + dLon]];
}

function updateMapSelection(pan = true) {
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;
  state.marker.setLatLng([latitude, longitude]);
  state.detailRect.setBounds(squareBounds(latitude, longitude, detailKm()));
  const px = detailPixels();
  elements.detailValue.value = `${detailKm().toFixed(2)} km · ${px}×${px} px`;
  elements.mapCoordinate.textContent = `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
  updateWindowCost(px);
  if (pan) state.map.panTo([latitude, longitude]);
}

/**
 * What the chosen window will cost, before the user spends it.
 *
 * A wide window is several Earth Engine requests and hundreds of model windows,
 * so the run can take minutes and hundreds of megabytes. Saying so up front is
 * cheaper than letting someone drag the slider to the top and wait.
 */
function updateWindowCost(px) {
  if (!state.meta) return;
  const meta = state.meta;
  const chunks = Math.ceil(px / DETAIL_CHUNK_PX) ** 2;
  const windows = Math.ceil(Math.max(1, (px - meta.patch_size) / meta.stride) + 1) ** 2;
  const planesMb = (px * px * 2 * meta.in_ch_per_phase * 4) / 2 ** 20;

  // patch_size / window_size is the scale the tile context is sampled at. In
  // training a patch grid tiled a whole fire tile, which puts this between
  // 0.047 and 0.289 — so a SMALL window is the off-distribution case here, not
  // a large one, which is the opposite of what the control suggests.
  const patchScale = meta.patch_size / px;
  const offScale = patchScale > 0.29;
  const heavy = px > 2048;
  elements.windowCost.textContent =
    `${chunks} Earth Engine request${chunks === 1 ? "" : "s"} · ${windows} model `
    + `window${windows === 1 ? "" : "s"} · ~${Math.round(planesMb)} MB in memory · `
    + `context scale ${patchScale.toFixed(3)}`
    + (offScale ? " — below the 17.9 km the model was trained at; results drift"
      : (heavy ? " · minutes on WebAssembly, and a lot of RAM" : ""));
  elements.windowCost.classList.toggle("is-heavy", heavy || offScale);
}

function detailPixels() {
  const scale = state.meta?.scale_m ?? 20;
  return Math.min(DETAIL_MAX_PX, Math.max(state.meta?.patch_size ?? 256,
    Math.round(detailKm() * 1000 / scale)));
}

function markSceneCustom(message = "Custom scene · coordinates, dates or window adjusted manually.") {
  elements.scenePreset.value = "";
  elements.presetMeta.textContent = message;
  // The series belonged to the preset that was selected. Once the scene is
  // the reader's own, offering that preset's acquisitions would set a date
  // over an area they no longer describe.
  elements.duringDateField.hidden = true;
}

// The fire the page opens on. Not the largest — a first run should finish in a
// reasonable time, and this is the one event here with an independent perimeter
// to compare against.
const DEFAULT_PRESET = "CA_2025_NT_523";

// The search radii a preset falls back to, matching index.html's selected
// options. A preset that does not narrow them must still reset them, or the
// previous selection's narrower window would silently carry over.
const DEFAULT_PRE_WINDOW = 20;
const DEFAULT_TARGET_WINDOW = 10;

/** Fill the preset <select> from SCENE_PRESETS, grouped by burned area.
 *
 * The options are built here rather than written into index.html so the two
 * cannot drift: with this many fires, a hand-maintained list would eventually
 * offer a value SCENE_PRESETS has no entry for, and applyScenePreset would
 * silently do nothing.
 */
function populateScenePresets() {
  const custom = document.createElement("option");
  custom.value = "";
  custom.disabled = true;
  custom.textContent = "Custom scene";
  elements.scenePreset.append(custom);

  let group = null;
  for (const [eventId, preset] of Object.entries(SCENE_PRESETS)) {
    if (!group || group.label !== preset.group) {
      group = document.createElement("optgroup");
      group.label = preset.group;
      elements.scenePreset.append(group);
    }
    const option = document.createElement("option");
    option.value = eventId;
    option.textContent = preset.label;
    group.append(option);
  }
  // samples.html links in as ./?preset=CA_2025_XX_NNN, so a reader who has
  // just looked at a fire's panels lands on that same fire rather than on
  // whatever the default is. An unknown id falls back instead of throwing.
  const asked = new URLSearchParams(window.location.search).get("preset");
  elements.scenePreset.value = (asked && SCENE_PRESETS[asked]) ? asked : DEFAULT_PRESET;
}

/** Offer a during-fire preset's whole series, or hide the control.
 *
 * Kept from the Sentinel-2 page for the progression presets; no Sentinel-1
 * preset defines a series yet, so the control stays hidden.
 */
function fillDuringDates(preset) {
  const field = elements.duringDateField;
  const select = elements.duringDate;
  select.replaceChildren();
  if (!preset.dates) {
    field.hidden = true;
    return;
  }
  preset.dates.forEach((date, i) => {
    const option = document.createElement("option");
    option.value = date;
    const seen = preset.observed?.[i];
    option.textContent = seen === undefined
      ? `${i + 1}/${preset.dates.length} · ${date}`
      : `${i + 1}/${preset.dates.length} · ${date} · ${Math.round(seen * 100)}% seen`;
    select.append(option);
  });
  select.value = preset.target;
  field.hidden = false;
}

function applyScenePreset(eventId) {
  const preset = SCENE_PRESETS[eventId];
  if (!preset) return;
  elements.latitude.value = preset.latitude;
  elements.longitude.value = preset.longitude;
  elements.detailSize.value = preset.detailKm;
  elements.preDate.value = preset.pre;
  elements.targetDate.value = preset.target;
  fillDuringDates(preset);
  // A preset may narrow the searches. The during-fire ones do: their scenes
  // are days apart while the scar is still growing, so a wide target search
  // would return a different stage of the fire, and several of their pre
  // scenes are May acquisitions far enough north that a wide pre search
  // reaches a different relative orbit than the one the post scene used.
  elements.preWindow.value = String(preset.preWindow ?? DEFAULT_PRE_WINDOW);
  elements.targetWindow.value = String(preset.targetWindow ?? DEFAULT_TARGET_WINDOW);
  elements.presetMeta.textContent = `${eventId} · ${preset.note}`;
  updateMapSelection(false);
  state.map.fitBounds(squareBounds(preset.latitude, preset.longitude, preset.detailKm),
    { padding: [40, 40], maxZoom: 11 });
}

function setMapExpanded(expanded) {
  elements.mapWorkspace.classList.toggle("is-map-expanded", expanded && !document.fullscreenElement);
  document.body.classList.toggle("map-expanded", expanded);
  elements.mapExpand.setAttribute("aria-pressed", String(expanded));
  elements.mapExpand.querySelector("span").textContent = expanded ? "Exit enlarged map" : "Enlarge map";
  setTimeout(() => state.map.invalidateSize(), 80);
}

async function toggleMapWorkspace() {
  const fallbackExpanded = elements.mapWorkspace.classList.contains("is-map-expanded");
  if (document.fullscreenElement === elements.mapWorkspace) {
    await document.exitFullscreen();
    return;
  }
  if (fallbackExpanded) {
    setMapExpanded(false);
    return;
  }
  if (elements.mapWorkspace.requestFullscreen) {
    try {
      await elements.mapWorkspace.requestFullscreen();
      return;
    } catch (_) { /* Fall back to an in-page enlarged view. */ }
  }
  setMapExpanded(true);
}

// ── Earth Engine connection ──────────────────────────────────────────────────

function initializeEarthEngine(projectId) {
  setEeStatus("Initializing", "loading");
  window.ee.initialize(
    null,
    null,
    () => {
      state.eeConnected = true;
      elements.runButton.disabled = false;
      elements.connectButton.disabled = false;
      elements.connectButton.querySelector("span").textContent = "Earth Engine connected";
      setEeStatus("Earth Engine ready", "ready");
      setMessage(`Connected through ${projectId}.`, false, true);
    },
    error => {
      state.eeConnected = false;
      elements.connectButton.disabled = false;
      setEeStatus("Connection failed");
      setMessage(`Earth Engine initialization failed: ${normalizeError(error)}`, true, true);
    },
    null,
    projectId,
  );
}

function connectEarthEngine() {
  const projectId = elements.projectId.value.trim();
  const clientId = elements.clientId.value.trim();
  if (!projectId) {
    setMessage("Enter the Earth Engine-enabled Google Cloud project ID.", true, true);
    return;
  }
  if (!clientId.endsWith(".apps.googleusercontent.com")) {
    setMessage("Enter a valid OAuth web client ID ending in .apps.googleusercontent.com.", true, true);
    return;
  }
  if (!window.ee) {
    setMessage("The Earth Engine JavaScript library did not load. Reload the page and try again.", true, true);
    return;
  }
  localStorage.setItem("flamenet-ee-project", projectId);
  localStorage.setItem("flamenet-ee-client-id", clientId);

  if (state.authPopupReady) {
    state.authPopupReady = false;
    elements.connectButton.disabled = true;
    setEeStatus("Sign-in opening", "loading");
    setMessage("Opening Google authorization… Allow the popup if your browser asks.", false, true);
    window.ee.data.authenticateViaPopup(
      () => initializeEarthEngine(projectId),
      error => {
        elements.connectButton.disabled = false;
        elements.connectButton.querySelector("span").textContent = "Continue with Google";
        state.authPopupReady = true;
        setEeStatus("Authorization failed");
        setMessage(`Google authorization failed: ${normalizeError(error)}`, true, true);
      },
    );
    return;
  }

  elements.connectButton.disabled = true;
  setEeStatus("Authorizing", "loading");
  setMessage("Checking for an existing Google authorization…", false, true);
  const attempt = ++state.authAttempt;
  let fallbackTimer;
  const offerPopup = (message = "Click “Continue with Google” to open the authorization window.") => {
    if (attempt !== state.authAttempt || state.eeConnected) return;
    state.authPopupReady = true;
    elements.connectButton.disabled = false;
    elements.connectButton.querySelector("span").textContent = "Continue with Google";
    setEeStatus("Sign-in required");
    setMessage(message, false, true);
  };
  const success = () => {
    clearTimeout(fallbackTimer);
    if (attempt !== state.authAttempt) return;
    state.authPopupReady = false;
    initializeEarthEngine(projectId);
  };
  const failure = error => {
    clearTimeout(fallbackTimer);
    if (attempt !== state.authAttempt) return;
    state.authPopupReady = false;
    elements.connectButton.disabled = false;
    setEeStatus("Authorization failed");
    setMessage(`Google authorization failed: ${normalizeError(error)}`, true, true);
  };
  fallbackTimer = setTimeout(() => {
    offerPopup("The silent sign-in check took too long. Click “Continue with Google” to authorize directly.");
  }, 8000);
  window.ee.data.authenticateViaOauth(
    clientId, success, failure, [EE_SCOPE],
    () => { clearTimeout(fallbackTimer); offerPopup(); },
    true,
  );
}

function eeEvaluate(computed) {
  return new Promise((resolve, reject) => {
    computed.evaluate((value, error) => error ? reject(new Error(normalizeError(error))) : resolve(value));
  });
}

function freshAuthToken() {
  const existing = window.ee.data.getAuthToken();
  if (existing) return Promise.resolve(existing);
  return new Promise((resolve, reject) => {
    window.ee.data.refreshAuthToken(
      () => {
        const refreshed = window.ee.data.getAuthToken();
        if (refreshed) resolve(refreshed);
        else reject(new Error("Google authorization expired. Reconnect Earth Engine."));
      },
      error => reject(new Error(normalizeError(error))),
    );
  });
}

// ── Earth Engine imagery ─────────────────────────────────────────────────────

function utmCrs(longitude, latitude) {
  const zone = Math.floor((longitude + 180) / 6) + 1;
  return `EPSG:${(latitude >= 0 ? 32600 : 32700) + zone}`;
}

/** A square grid of `pixels` cells of `scale` m, centred on (x, y) in `crsCode`. */
function utmGrid(x, y, pixels, scale, crsCode) {
  const span = pixels * scale;
  return {
    dimensions: { width: pixels, height: pixels },
    affineTransform: {
      scaleX: scale, shearX: 0, translateX: x - span / 2,
      shearY: 0, scaleY: -scale, translateY: y + span / 2,
    },
    crsCode,
  };
}

const gridTransformArray = grid => [
  grid.affineTransform.scaleX, grid.affineTransform.shearX, grid.affineTransform.translateX,
  grid.affineTransform.shearY, grid.affineTransform.scaleY, grid.affineTransform.translateY,
];

/**
 * One image per datatake, which is the scene unit the archive was built from:
 * a fire spanning two granules of the same overpass is one scene, not two.
 * Mirrors the archive's per-date mosaic in gee/s1s2.py::get_s1_dict, tightened
 * to group by date AND relative orbit.
 */
/**
 * One overpass is one scene: the granules sharing a date AND a relative orbit
 * are mosaicked together, and nothing else is.
 *
 * Grouping by date alone -- what the training archive did -- merges two orbits
 * whenever both fly over the same day, which happens on 3% of its tiles and
 * puts two incidence angles in one image. Grouping by date and orbit costs one
 * extra property and cannot do that.
 */
function s1Scenes(roi, start, end) {
  const ee = window.ee;
  const meta = state.meta;
  const collection = ee.ImageCollection(meta.collection)
    .filterBounds(roi)
    .filterDate(start, end)
    .filter(ee.Filter.eq("instrumentMode", meta.instrument_mode))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .map(image => {
      const img = ee.Image(image);
      const day = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd");
      const orbit = ee.Number(img.get("relativeOrbitNumber_start")).format("%d");
      return img.set("overpass", day.cat("_").cat(orbit), "acquisition_day", day);
    });
  const keys = ee.List(collection.aggregate_array("overpass")).distinct();
  return ee.ImageCollection(keys.map(key => {
    const subset = collection.filter(ee.Filter.eq("overpass", key));
    const first = ee.Image(subset.first());
    return subset.mosaic().select(meta.bands).set({
      overpass: key,
      "system:time_start": first.get("system:time_start"),
      acquisition_day: first.get("acquisition_day"),
      rel_orbit: first.get("relativeOrbitNumber_start"),
      orbit_pass: first.get("orbitProperties_pass"),
    });
  }));
}

/**
 * Score each overpass by how much of the window it actually covers.
 *
 * Radar has no clarity to rank, so coverage is the whole of it: a pass that
 * clips the corner of the window is useless however good its geometry, and the
 * training archive dropped any date whose mosaic left more than 20% of the ROI
 * empty.
 */
function scoreScenes(collection, roi, scoreScale) {
  const ee = window.ee;
  const meta = state.meta;
  return collection.map(image => {
    const img = ee.Image(image);
    const band = meta.bands[0];
    const covered = img.select(band).mask().reduceRegion({
      reducer: ee.Reducer.mean(), geometry: roi, scale: scoreScale,
      maxPixels: 1e8, bestEffort: true,
    }).get(band);
    const coverage = ee.Number(ee.Algorithms.If(covered, covered, 0));
    return img.set({
      cs_mean: coverage,
      nan_frac: ee.Number(1).subtract(coverage),
      usability: coverage,
    });
  });
}

/** Best coverage wins; a mostly-empty footprint is set aside unless it is all
 * there is. */
function pickScene(collection) {
  const ee = window.ee;
  const usable = collection.filter(ee.Filter.lte("nan_frac", 0.5));
  const pool = ee.ImageCollection(ee.Algorithms.If(usable.size().gt(0), usable, collection));
  return ee.Image(pool.sort("usability", false).first());
}

/**
 * The pre-fire scene, taken from the post scene's relative orbit when that
 * orbit flew in the pre window.
 *
 * Pairs that share an orbit see the fire at the same incidence angle -- 0.01
 * degrees apart at the median, against 7.9 when they do not -- and score 0.489
 * event-averaged IoU on the 2025 season against 0.379 for pairs that do not.
 * The archive matched only 67% of the time; this does better where the data
 * allow, and falls back to best coverage where they do not.
 */
function pickPreScene(collection, postOrbit) {
  const ee = window.ee;
  const matched = collection.filter(ee.Filter.eq("rel_orbit", postOrbit));
  const pool = ee.ImageCollection(
    ee.Algorithms.If(ee.Number(state.meta.prefer_same_relative_orbit ? 1 : 0)
      .and(matched.size().gt(0)), matched, collection));
  return pickScene(pool);
}

/**
 * The scale the scene-ranking reductions run at.
 *
 * Every candidate gets two reduceRegion calls, and a wide window can have 40 of
 * them in flight: at a fixed fine scale Earth Engine answers "Too many
 * concurrent aggregations" and the run dies before any imagery is fetched. The
 * scale therefore grows with the window, keeping each reduction near 150x150
 * cells whatever the extent. It costs nothing real — these numbers only rank
 * scenes, and over a 72 km window the mean coverage moves by 0.002 between 480 m
 * and 1920 m.
 */
function scoringScale(detailPixels) {
  const metres = detailPixels * state.meta.scale_m;
  return Math.max(state.meta.scale_m * 6, Math.round(metres / 150));
}

async function selectScenes(roi, preDate, targetDate, preRadius, targetRadius, scoreScale) {
  const ee = window.ee;
  const window_ = (centre, radius) => [
    ee.Date(centre).advance(-radius, "day"), ee.Date(centre).advance(radius + 1, "day"),
  ];
  const [preStart, preEnd] = window_(preDate, preRadius);
  const [postStart, postEnd] = window_(targetDate, targetRadius);
  const preAll = s1Scenes(roi, preStart, preEnd);
  const postAll = s1Scenes(roi, postStart, postEnd);
  const counts = await Promise.all([eeEvaluate(preAll.size()), eeEvaluate(postAll.size())]);
  if (!counts[0]) {
    throw new Error(`No Sentinel-1 IW VV+VH overpass within ±${preRadius} days of the `
      + `pre-fire date ${preDate}. Widen the pre-fire search or move the date.`);
  }
  if (!counts[1]) {
    throw new Error(`No Sentinel-1 IW VV+VH overpass within ±${targetRadius} days of ${targetDate}.`);
  }
  const postImage = pickScene(scoreScenes(postAll, roi, scoreScale));
  const preImage = pickPreScene(scoreScenes(preAll, roi, scoreScale),
    postImage.get("rel_orbit"));
  const metadata = await eeEvaluate(ee.Dictionary({
    pre_date: preImage.get("acquisition_day"),
    pre_cs: preImage.get("cs_mean"),
    pre_nan: preImage.get("nan_frac"),
    pre_candidates: counts[0],
    pre_orbit: preImage.get("rel_orbit"),
    pre_pass: preImage.get("orbit_pass"),
    post_date: postImage.get("acquisition_day"),
    post_cs: postImage.get("cs_mean"),
    post_nan: postImage.get("nan_frac"),
    post_candidates: counts[1],
    post_orbit: postImage.get("rel_orbit"),
    post_pass: postImage.get("orbit_pass"),
  }));
  return { preImage, postImage, metadata };
}

/** Four bands, pre then post, named so both requests return a fixed order. */
function bandIds() {
  return PHASES.flatMap(phase => state.meta.bands.map(band => `${phase}_${band}`));
}

function stackPhases(preImage, postImage) {
  const ee = window.ee;
  const names = state.meta.bands;
  const rename = (image, phase) => ee.Image(image).select(names, names.map(b => `${phase}_${b}`));
  return rename(preImage, "pre").addBands(rename(postImage, "post"));
}

/**
 * The detail window, at the model's native 20 m, as int16.
 *
 * S1_GRD is sigma-nought in dB, a float. Carrying it as dB x 100 keeps 0.01 dB
 * -- 0.0024 of a standardised unit -- and halves the response against float32,
 * which is what lets the window reach 1024 px inside Earth Engine's 32 MiB cap.
 */
function detailImage(stack) {
  const names = bandIds();
  return stack.select(names).multiply(DB_QUANT).unmask(NODATA).round().int16();
}

/**
 * The context window, averaged onto the 256 px grid the tile context encoder
 * was trained on.
 *
 * reduceResolution against the request grid's own transform is an exact
 * area-weighted mean of the 20 m pixels — the same quantity cv2.INTER_AREA
 * computes in the loader — so the thumbnail is not resampled a second time on
 * the way out.
 */
function contextImage(stack, crsCode, grid, scale) {
  const ee = window.ee;
  const maxPixels = Math.max(64, Math.ceil((scale / state.meta.scale_m) ** 2) * 4);
  return stack
    .reproject({ crs: crsCode, scale: state.meta.scale_m })
    .reduceResolution({ reducer: ee.Reducer.mean(), bestEffort: false, maxPixels })
    .reproject({ crs: crsCode, crsTransform: gridTransformArray(grid) })
    .unmask(NODATA)
    .toFloat();
}

async function computePixels(image, grid, projectId, bands, tag) {
  const token = await freshAuthToken();
  const response = await fetch(
    `https://earthengine.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/image:computePixels`,
    {
      method: "POST",
      headers: {
        Authorization: token,
        "Content-Type": "application/json",
        "X-Goog-User-Project": projectId,
      },
      body: JSON.stringify({
        expression: window.ee.Serializer.encodeCloudApi(image),
        fileFormat: "GEO_TIFF",
        grid,
        bandIds: bands,
        workloadTag: "flamenet-s1-browser",
      }),
    },
  );
  if (!response.ok) {
    let detail = "";
    try { detail = (await response.json()).error?.message || ""; } catch (_) { /* binary or empty */ }
    throw new Error(`Earth Engine ${tag} request failed (${response.status})${detail ? `: ${detail}` : "."}`);
  }
  return response.blob();
}

/**
 * Read an eight-band GeoTIFF into two phases of reflectance, reproducing
 * src/datasets/sar_fire.py's _read_tif: dB as stored, no-data filled with
 * the channel mean, which is zero after standardising rather than an
 * typically clear), then the reflectance channels clipped.
 */
/** How a band of the computePixels response is turned into reflectance. */
function channelOptions(channel, quant) {
  const meta = state.meta;
  return {
    // outside the swath -> the channel mean, i.e. zero after standardising,
    // which is what sar_fire._read_tif does with the GRD's non-finite edge
    fill: meta.mean[channel],
    scale: 1 / quant,
    clip: null,
    nodata: NODATA,
  };
}

async function readRasters(arrayBuffer) {
  const expected = PHASES.length * state.meta.in_ch_per_phase;
  const tiff = await fromArrayBuffer(arrayBuffer);
  const image = await tiff.getImage();
  const samples = image.getSamplesPerPixel();
  if (samples < expected) {
    throw new Error(`Earth Engine returned ${samples} bands; ${expected} were expected.`);
  }
  return {
    width: image.getWidth(),
    height: image.getHeight(),
    bands: await image.readRasters({
      samples: [...Array(expected).keys()], interleave: false,
    }),
  };
}

/** One request's worth: used for the context thumbnail, which always fits. */
async function readStack(arrayBuffer, quant) {
  const nChannels = state.meta.in_ch_per_phase;
  const { width, height, bands } = await readRasters(arrayBuffer);
  const makePhase = offset => ({
    width, height,
    channels: Array.from({ length: nChannels }, (_, c) =>
      convertChannel(bands[offset + c], channelOptions(c, quant))),
  });
  return { pre: makePhase(0), post: makePhase(nChannels) };
}

/**
 * Fetch the detail window, in as many computePixels requests as it takes.
 *
 * Earth Engine caps a response at 32 MiB, which with eight int16 bands is about
 * 1024 px square — far smaller than the fires this model is for, whose scars
 * run 55-75 km across. So the window is tiled, each chunk requested on its own
 * grid cut from the same 20 m UTM lattice, and blitted into one set of planes.
 * Because every chunk shares the parent grid's origin and scale, the reassembled
 * raster is pixel-identical to the single request it replaces.
 */
async function fetchDetailWindow(image, grid, projectId, bands, quant, onProgress) {
  const meta = state.meta;
  const pixels = grid.dimensions.width;
  const a = grid.affineTransform;
  const rows = chunkEdges(pixels, DETAIL_CHUNK_PX);
  const cols = chunkEdges(pixels, DETAIL_CHUNK_PX);
  const total = (rows.length - 1) * (cols.length - 1);

  const nChannels = meta.in_ch_per_phase;
  const plane = pixels * pixels;
  const makePhase = () => ({
    width: pixels, height: pixels,
    channels: Array.from({ length: nChannels }, () => new Float32Array(plane)),
  });
  const pre = makePhase();
  const post = makePhase();

  const blobs = [];
  let done = 0;
  for (let r = 0; r < rows.length - 1; r += 1) {
    for (let c = 0; c < cols.length - 1; c += 1) {
      const row = rows[r];
      const col = cols[c];
      const chunkH = rows[r + 1] - row;
      const chunkW = cols[c + 1] - col;
      const chunkGrid = {
        dimensions: { width: chunkW, height: chunkH },
        affineTransform: {
          ...a,
          translateX: a.translateX + col * a.scaleX,
          translateY: a.translateY + row * a.scaleY,
        },
        crsCode: grid.crsCode,
      };
      onProgress(done, total, `${chunkW}×${chunkH} px`);
      const blob = await computePixels(image, chunkGrid, projectId, bands,
        total > 1 ? `detail chunk ${done + 1}/${total}` : "detail");
      blobs.push({ blob, row, col });
      const raster = await readRasters(await blob.arrayBuffer());
      if (raster.width !== chunkW || raster.height !== chunkH) {
        throw new Error(`Earth Engine returned ${raster.width}×${raster.height} px for a `
          + `${chunkW}×${chunkH} px chunk.`);
      }
      const geom = { targetWidth: pixels, chunkWidth: chunkW, chunkHeight: chunkH, row, col };
      for (const [phase, offset] of [[pre, 0], [post, nChannels]]) {
        for (let ch = 0; ch < nChannels; ch += 1) {
          blitChannel(raster.bands[offset + ch], phase.channels[ch],
            channelOptions(ch, quant), geom);
        }
      }
      done += 1;
    }
  }
  return { pre, post, blobs, requests: total };
}

// ── inference ────────────────────────────────────────────────────────────────

async function createSessions() {
  if (state.ctxSession && state.patchSession) return;
  if (!window.ort) throw new Error("ONNX Runtime did not load. Reload the page and try again.");
  window.ort.env.wasm.wasmPaths = ORT_ASSET_PATH;
  window.ort.env.wasm.numThreads = 1;
  const graphs = state.meta.graphs;
  const create = provider => Promise.all([
    window.ort.InferenceSession.create(`${MODEL_DIR}${graphs.ctx}`, { executionProviders: [provider] }),
    window.ort.InferenceSession.create(`${MODEL_DIR}${graphs.patch}`, { executionProviders: [provider] }),
  ]);
  try {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    [state.ctxSession, state.patchSession] = await create("webgpu");
    state.inferenceEngine = "WebGPU";
  } catch (_) {
    [state.ctxSession, state.patchSession] = await create("wasm");
    state.inferenceEngine = "WebAssembly";
  }
}

async function runInference(geometry) {
  const meta = state.meta;
  const ort = window.ort;
  const patch = meta.patch_size;
  const stride = meta.stride;
  const { width, height } = state.post;

  // In place: see the note in runWorkflow — at the top window size a second
  // copy of the detail planes would be another 537 MB. Anything needing raw
  // reflectance has already been computed by the time this runs.
  const preNorm = standardiseInto(state.pre.channels, meta.mean, meta.std);
  const postNorm = standardiseInto(state.post.channels, meta.mean, meta.std);

  // The thumbnail is small, so it keeps the allocating version.
  const norm = phase => standardise(phase.channels, meta.mean, meta.std);

  // Tile context: one pass over the context thumbnail, reused by every patch.
  const ctxPixels = meta.ctx_size * meta.ctx_size;
  const ctxInput = new Float32Array(2 * meta.in_ch_per_phase * ctxPixels);
  const thumbNorm = [...norm(state.ctxPre), ...norm(state.ctxPost)];
  thumbNorm.forEach((channel, index) => ctxInput.set(channel, index * ctxPixels));
  const ctxOutput = await state.ctxSession.run({
    ctx_in: new ort.Tensor("float32", ctxInput, [1, 2 * meta.in_ch_per_phase, meta.ctx_size, meta.ctx_size]),
  });
  const tileCtx = ctxOutput.tile_ctx.data;
  const [, ctxChannels, ctxH, ctxW] = ctxOutput.tile_ctx.dims;

  const weight = gaussianWeight(patch);
  const accumulated = new Float32Array(width * height);
  const weightSum = new Float32Array(width * height);

  const rows = windowStarts(height, patch, stride);
  const cols = windowStarts(width, patch, stride);
  const total = rows.length * cols.length;
  let done = 0;

  for (const r1 of rows) {
    for (const c1 of cols) {
      const pre = extractPatch(preNorm, width, height, r1, c1, patch);
      const post = extractPatch(postNorm, width, height, r1, c1, patch);
      // Patch position is normalised against the CONTEXT window — the extent
      // the encoder saw — not against the detail window inside it.
      const centreRow = geometry.detailOriginRow + r1 + patch / 2;
      const centreCol = geometry.detailOriginCol + c1 + patch / 2;
      const ctxBtn = sampleContext(
        tileCtx, ctxChannels, ctxH, ctxW,
        (centreRow / geometry.contextPixels) * 2 - 1,
        (centreCol / geometry.contextPixels) * 2 - 1,
        patch / geometry.contextPixels,
        patch / geometry.contextPixels,
        meta.btn_size,
      );
      const output = await state.patchSession.run({
        pre: new ort.Tensor("float32", pre, [1, meta.in_ch_per_phase, patch, patch]),
        post: new ort.Tensor("float32", post, [1, meta.in_ch_per_phase, patch, patch]),
        ctx_btn: new ort.Tensor("float32", ctxBtn, [1, ctxChannels, meta.btn_size, meta.btn_size]),
      });
      blendWindow(accumulated, weightSum, output.probability.data, weight,
        width, height, r1, c1, patch);
      done += 1;
      setProgress(0.62 + 0.36 * done / total, `Running model window ${done} of ${total}…`);
      await new Promise(requestAnimationFrame);
    }
  }
  return normaliseBlend(accumulated, weightSum);
}

// ── run ──────────────────────────────────────────────────────────────────────

function validateRunInputs() {
  if (!state.meta) throw new Error("The model metadata has not loaded yet.");
  if (!state.eeConnected) throw new Error("Connect Earth Engine first.");
  const latitude = Number(elements.latitude.value);
  const longitude = Number(elements.longitude.value);
  if (!Number.isFinite(latitude) || latitude < -85 || latitude > 85) {
    throw new Error("Latitude must be between −85 and 85.");
  }
  if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
    throw new Error("Longitude must be between −180 and 180.");
  }
  const pre = new Date(`${elements.preDate.value}T00:00:00Z`);
  const target = new Date(`${elements.targetDate.value}T00:00:00Z`);
  if (!Number.isFinite(pre.getTime()) || !Number.isFinite(target.getTime())) {
    throw new Error("Choose a valid pre-fire date and target observation.");
  }
  if (target <= pre) throw new Error("The target observation must be after the pre-fire date.");
  return { latitude, longitude, lagDays: Math.round((target - pre) / 86400000) };
}

async function runWorkflow() {
  elements.runButton.disabled = true;
  elements.results.hidden = true;
  setMessage("");
  setProgress(0.02, "Building the Earth Engine query…");
  try {
    const { latitude, longitude, lagDays } = validateRunInputs();
    const meta = state.meta;
    const ee = window.ee;
    const projectId = elements.projectId.value.trim();
    const crsCode = utmCrs(longitude, latitude);

    // Everything is laid out on one 20 m UTM grid: the detail window is a
    // centred crop of the context window, so a patch's position inside the
    // context is exact rather than reprojected.
    const detailPx = detailPixels();
    const contextPx = Math.round(contextKm() * 1000 / meta.scale_m);
    const contextScale = (contextPx * meta.scale_m) / meta.ctx_size;

    setProgress(0.06, "Locating the window in UTM…");
    const [centreX, centreY] = await eeEvaluate(
      ee.Geometry.Point([longitude, latitude]).transform(crsCode, 0.01).coordinates());
    const detailGrid = utmGrid(centreX, centreY, detailPx, meta.scale_m, crsCode);
    const contextGrid = utmGrid(centreX, centreY, meta.ctx_size, contextScale, crsCode);
    const roi = ee.Geometry.Rectangle(
      [centreX - detailPx * meta.scale_m / 2, centreY - detailPx * meta.scale_m / 2,
        centreX + detailPx * meta.scale_m / 2, centreY + detailPx * meta.scale_m / 2],
      crsCode, false);
    const contextRoi = ee.Geometry.Rectangle(
      [centreX - contextPx * meta.scale_m / 2, centreY - contextPx * meta.scale_m / 2,
        centreX + contextPx * meta.scale_m / 2, centreY + contextPx * meta.scale_m / 2],
      crsCode, false);

    setProgress(0.1, `Searching ${meta.collection} for IW VV+VH overpasses…`);
    const selected = await selectScenes(
      roi, elements.preDate.value, elements.targetDate.value,
      Number(elements.preWindow.value), Number(elements.targetWindow.value),
      scoringScale(detailPx));
    state.selected = selected.metadata;
    setProgress(0.22,
      `Selected ${selected.metadata.pre_date} and ${selected.metadata.post_date}`);

    const stack = stackPhases(selected.preImage, selected.postImage);
    const bands = bandIds();

    const detail = await fetchDetailWindow(
      detailImage(stack), detailGrid, projectId, bands, DB_QUANT,
      (done, total, size) => setProgress(0.26 + 0.18 * done / total,
        total > 1
          ? `Fetching ${detailPx}×${detailPx} px at ${meta.scale_m} m — request `
            + `${done + 1} of ${total} (${size})…`
          : `Fetching ${detailPx}×${detailPx} px at ${meta.scale_m} m…`));
    state.pre = detail.pre;
    state.post = detail.post;
    state.sourceBlobs = detail.blobs;

    // Both of these need RAW reflectance, and runInference standardises the
    // planes in place to avoid a second 537 MB copy at the top window size.
    // So the two dual-polarisation panels are
    // rasterised to their canvases now rather than kept as arrays — the
    // canvases are at most 900 px a side, where the planes are up to 4096.
    // No cloud mask: radar sees through cloud. Pixels outside the swath are
    // filled with the channel mean and will read as unburned, which is why the
    // summary reports each scene's no-data fraction.
    state.keep = null;
    const preRamp = sarRamp(state.pre.channels);
    const postRamp = sarRamp(state.post.channels);
    renderToCanvas(elements.preCanvas, i => sarColour(state.pre.channels, preRamp, i));
    renderToCanvas(elements.currentCanvas, i => sarColour(state.post.channels, postRamp, i));

    setProgress(0.46, `Fetching the ${contextKm().toFixed(0)} km context thumbnail…`);
    const contextBlob = await computePixels(
      contextImage(stack, crsCode, contextGrid, contextScale),
      contextGrid, projectId, bands, "context");
    const context = await readStack(await contextBlob.arrayBuffer(), 1);
    state.ctxPre = context.pre;
    state.ctxPost = context.post;

    const geometry = {
      detailPixels: detailPx,
      contextPixels: contextPx,
      // The detail window is centred in the context window.
      detailOriginRow: (contextPx - detailPx) / 2,
      detailOriginCol: (contextPx - detailPx) / 2,
      crsCode,
      lagDays,
    };
    state.geometry = geometry;

    setProgress(0.56, `Loading ${meta.name}…`);
    await createSessions();
    state.probability = await runInference(geometry);

    renderResults();
    setProgress(1, "Earth Engine retrieval and inference complete");
    elements.results.hidden = false;
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error(error);
    elements.progressWrap.hidden = true;
    setMessage(normalizeError(error), true);
  } finally {
    elements.runButton.disabled = !state.eeConnected;
  }
}

// ── rendering ────────────────────────────────────────────────────────────────

function previewGeometry(width, height) {
  const scale = Math.min(1, 900 / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

function renderToCanvas(canvas, pixelFunction, fullResolution = false) {
  const sourceWidth = state.post.width;
  const sourceHeight = state.post.height;
  const target = fullResolution
    ? { width: sourceWidth, height: sourceHeight }
    : previewGeometry(sourceWidth, sourceHeight);
  canvas.width = target.width;
  canvas.height = target.height;
  const context = canvas.getContext("2d");
  const output = context.createImageData(target.width, target.height);
  for (let y = 0; y < target.height; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor(y * sourceHeight / target.height));
    for (let x = 0; x < target.width; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor(x * sourceWidth / target.width));
      const index = sourceY * sourceWidth + sourceX;
      const targetIndex = (y * target.width + x) * 4;
      const colour = pixelFunction(index);
      output.data[targetIndex] = colour[0];
      output.data[targetIndex + 1] = colour[1];
      output.data[targetIndex + 2] = colour[2];
      output.data[targetIndex + 3] = colour.length > 3 ? colour[3] : 255;
    }
  }
  context.putImageData(output, 0, 0);
}

function probabilityColour(value) {
  const stops = [[0, 3, 8, 7], [0.35, 57, 38, 29], [0.6, 223, 75, 32], [0.8, 255, 173, 66], [1, 255, 252, 229]];
  let upper = 1;
  while (upper < stops.length && value > stops[upper][0]) upper += 1;
  const high = stops[Math.min(upper, stops.length - 1)];
  const low = stops[Math.max(0, upper - 1)];
  const amount = high[0] === low[0] ? 0 : (value - low[0]) / (high[0] - low[0]);
  return [1, 2, 3].map(i => Math.round(low[i] + (high[i] - low[i]) * amount));
}

function currentMask() {
  const meta = state.meta;
  const threshold = Number(elements.threshold.value);
  const binary = new Uint8Array(state.probability.length);
  for (let i = 0; i < binary.length; i += 1) binary[i] = state.probability[i] >= threshold ? 1 : 0;
  const filtered = removeSmallComponents(
    binary, state.post.width, state.post.height, meta.min_cc_area);
  if (state.keep) {
    for (let i = 0; i < filtered.length; i += 1) if (!state.keep[i]) filtered[i] = 0;
  }
  return filtered;
}

function renderResults() {
  const meta = state.meta;
  const threshold = Number(elements.threshold.value);
  elements.thresholdValue.value = threshold.toFixed(2);
  const mask = currentMask();

  // The two false-colour panels were rasterised in runWorkflow, while the
  // reflectance was still raw; only the model's own outputs change with the
  // threshold, so only these two are redrawn.
  renderToCanvas(elements.probabilityCanvas, i => state.keep && !state.keep[i]
    ? [86, 96, 92, 255]
    : [...probabilityColour(state.probability[i]), 255]);
  renderToCanvas(elements.maskCanvas, i => state.keep && !state.keep[i]
    ? [86, 96, 92, 255]
    : (mask[i] ? [255, 255, 255, 255] : [0, 0, 0, 255]));

  const pixelHa = (meta.scale_m ** 2) / 10000;
  let burned = 0;
  let obscured = 0;
  for (let i = 0; i < mask.length; i += 1) {
    if (state.keep && !state.keep[i]) obscured += 1;
    else if (mask[i]) burned += 1;
  }
  const observed = mask.length - obscured;

  const selected = state.selected;
  elements.preDateSummary.textContent = selected.pre_date;
  elements.preClearSummary.textContent =
    `orbit ${selected.pre_orbit} ${String(selected.pre_pass).toLowerCase()} `
    + `· ${(100 * Number(selected.pre_nan)).toFixed(1)}% no-data `
    + `· ${selected.pre_candidates} candidate overpass${selected.pre_candidates === 1 ? "" : "es"}`;
  elements.currentDateSummary.textContent = selected.post_date;
  elements.currentClearSummary.textContent =
    `orbit ${selected.post_orbit} ${String(selected.post_pass).toLowerCase()} `
    + `· ${(100 * Number(selected.post_nan)).toFixed(1)}% no-data `
    + `· ${selected.post_candidates} candidate overpass${selected.post_candidates === 1 ? "" : "es"}`;
  elements.lagSummary.textContent = `${state.geometry.lagDays} days`;
  elements.engineSummary.textContent =
    `${state.inferenceEngine} · ${state.geometry.crsCode} · ${state.geometry.detailPixels} px`;
  elements.burnedSummary.textContent = `${(burned * pixelHa).toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })} ha`;
  elements.pixelSummary.textContent = observed
    ? `${(100 * burned / observed).toFixed(1)}% of the observed window`
    : "nothing observed";
  elements.obscuredSummary.textContent = `${(obscured * pixelHa).toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })} ha`;
  elements.obscuredDetail.textContent = Number(selected.pre_orbit) === Number(selected.post_orbit)
    ? `pre and post share relative orbit ${selected.post_orbit} — the geometry the model scores 0.489 IoU on`
    : `pre orbit ${selected.pre_orbit} against post orbit ${selected.post_orbit} — mismatched, worth about 0.11 IoU`;
}

// ── downloads ────────────────────────────────────────────────────────────────

function downloadMask() {
  if (!state.probability) return;
  const mask = currentMask();
  const threshold = Number(elements.threshold.value);
  const canvas = document.createElement("canvas");
  renderToCanvas(canvas, i => state.keep && !state.keep[i]
    ? [86, 96, 92, 255]
    : (mask[i] ? [255, 255, 255, 255] : [0, 0, 0, 255]), true);
  canvas.toBlob(blob => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `flamenet_s1_mask_${state.selected.post_date}_t${threshold.toFixed(2)}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
}

/**
 * The imagery the model was run on, as Earth Engine returned it.
 *
 * A window wider than one request comes back as several chunks, so each is
 * offered separately, named by its pixel offset in the window. They share the
 * parent grid, so gdal_merge or a VRT reassembles them exactly.
 */
function downloadSourceStack() {
  const blobs = state.sourceBlobs;
  if (!blobs?.length) return;
  const stem = `s2_pair_${state.selected.pre_date}_${state.selected.post_date}`;
  blobs.forEach(({ blob, row, col }, index) => {
    // Browsers drop same-tick repeated downloads; space them out.
    setTimeout(() => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = blobs.length === 1 ? `${stem}.tif` : `${stem}_r${row}_c${col}.tif`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, index * 400);
  });
}

// ── wiring ───────────────────────────────────────────────────────────────────

elements.connectButton.addEventListener("click", connectEarthEngine);
elements.runButton.addEventListener("click", runWorkflow);
elements.scenePreset.addEventListener("change", event => applyScenePreset(event.target.value));
// Stepping through a during-fire series moves the target date and nothing
// else — same fire, same window, a later moment in it — so this is not the
// manual edit that markSceneCustom exists for.
elements.duringDate.addEventListener("change", event => {
  elements.targetDate.value = event.target.value;
  const preset = SCENE_PRESETS[elements.scenePreset.value];
  if (!preset) return;
  const i = preset.dates.indexOf(event.target.value);
  elements.presetMeta.textContent =
    `${elements.scenePreset.value} · acquisition ${i + 1} of ${preset.dates.length}`
    + ` · ${preset.note}`;
});
for (const field of [elements.latitude, elements.longitude]) {
  field.addEventListener("change", () => {
    markSceneCustom();
    updateMapSelection();
  });
}
elements.detailSize.addEventListener("input", () => {
  markSceneCustom();
  updateMapSelection(false);
});
elements.preDate.addEventListener("change", () => markSceneCustom());
elements.targetDate.addEventListener("change", () => markSceneCustom());
elements.mapExpand.addEventListener("click", toggleMapWorkspace);
document.addEventListener("fullscreenchange", () => {
  setMapExpanded(document.fullscreenElement === elements.mapWorkspace);
});
// Re-rendering means re-running the component filter over up to a million
// pixels, so the slider updates its readout immediately and the maps catch up
// once the drag settles.
let thresholdTimer = null;
elements.threshold.addEventListener("input", () => {
  elements.thresholdValue.value = Number(elements.threshold.value).toFixed(2);
  if (!state.probability) return;
  clearTimeout(thresholdTimer);
  thresholdTimer = setTimeout(renderResults, 140);
});
elements.downloadMask.addEventListener("click", downloadMask);
elements.downloadStack.addEventListener("click", downloadSourceStack);

elements.projectId.value = localStorage.getItem("flamenet-ee-project") || elements.projectId.value;
elements.clientId.value = localStorage.getItem("flamenet-ee-client-id") || "";

// The origin Google wants in Authorized JavaScript origins is this page's own,
// and getting it wrong is the usual reason a first sign-in fails. Print the
// real one rather than asking the reader to construct it.
elements.pageOrigin.textContent = location.origin;
// Open the setup walkthrough for anyone who has not been through it. A reader
// who already has both fields saved does not need it in their way.
elements.setupHelp.open = !(elements.projectId.value && elements.clientId.value);

populateScenePresets();
initializeMap();

loadMeta()
  .then(() => {
    applyScenePreset(elements.scenePreset.value);
    updateMapSelection(false);
  })
  .catch(error => {
    setMessage(`Could not load the model metadata: ${normalizeError(error)}`, true);
    elements.runButton.disabled = true;
  });
