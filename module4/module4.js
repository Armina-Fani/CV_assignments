/* ============================================================================
   CSc 8830 - Computer Vision | Module 4 (web demo)
   Human boundary extraction in RGB and thermal images -- classical only.

   This file is a line-by-line port of
     module4/rgb_human_segmentation.py
     module4/thermal_human_segmentation.py
     module4/compare_with_sam2.py   (metrics only; SAM2 itself is not run
                                     in the browser -- upload its mask)
   It uses OpenCV.js (loaded by index.html) for filtering, morphology,
   watershed, connected components, distance transforms and contours. Colour
   / texture histograms, Otsu, hysteresis and the metrics are plain JS.

   The algorithm functions (m4SegmentRGB, m4SegmentThermal, m4Compare) take
   and return plain typed arrays, so they also run under Node for testing.
   ========================================================================== */

"use strict";

// ---------------------------------------------------------------------------
// Generic helpers (operate on Uint8Array masks, 0/255, row-major H x W)
// ---------------------------------------------------------------------------

function m4Kernel(cv, size) {
  size = Math.max(3, Math.floor(size) | 1);
  return cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
}

function m4MatFromU8(cv, arr, H, W, type) {
  const m = new cv.Mat(H, W, type === undefined ? cv.CV_8UC1 : type);
  m.data.set(arr);
  return m;
}

function m4Morph(cv, maskArr, H, W, op, size) {
  const src = m4MatFromU8(cv, maskArr, H, W), dst = new cv.Mat(), k = m4Kernel(cv, size);
  cv.morphologyEx(src, dst, op, k, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, new cv.Scalar(0));
  const out = new Uint8Array(dst.data);
  src.delete(); dst.delete(); k.delete();
  return out;
}
function m4Erode(cv, m, H, W, s) { return m4Morph(cv, m, H, W, cv.MORPH_ERODE, s); }
function m4Dilate(cv, m, H, W, s) { return m4Morph(cv, m, H, W, cv.MORPH_DILATE, s); }

// Fill enclosed holes: flood the background from the border (BFS); every
// background pixel that is not reached is a hole.
function m4FillHoles(mask, H, W) {
  const seen = new Uint8Array(H * W), q = new Int32Array(H * W);
  let head = 0, tail = 0;
  const push = i => { if (!seen[i] && !mask[i]) { seen[i] = 1; q[tail++] = i; } };
  for (let x = 0; x < W; x++) { push(x); push((H - 1) * W + x); }
  for (let y = 0; y < H; y++) { push(y * W); push(y * W + W - 1); }
  while (head < tail) {
    const i = q[head++], x = i % W, y = (i / W) | 0;
    if (x > 0) push(i - 1); if (x < W - 1) push(i + 1);
    if (y > 0) push(i - W); if (y < H - 1) push(i + W);
  }
  const out = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) out[i] = (mask[i] || !seen[i]) ? 255 : 0;
  return out;
}

// Connected components with stats -> {n, labels:Int32Array, stats:[[x,y,w,h,area],...]}
function m4Components(cv, mask, H, W) {
  const src = m4MatFromU8(cv, mask, H, W), lab = new cv.Mat(), st = new cv.Mat(), ce = new cv.Mat();
  const n = cv.connectedComponentsWithStats(src, lab, st, ce, 8, cv.CV_32S);
  const labels = new Int32Array(lab.data32S), stats = [];
  for (let i = 0; i < n; i++) stats.push(Array.from(st.data32S.slice(i * 5, i * 5 + 5)));
  src.delete(); lab.delete(); st.delete(); ce.delete();
  return { n, labels, stats };
}

// Otsu threshold on a list of uint8 values (same result as cv2 THRESH_OTSU)
function m4Otsu(values) {
  const hist = new Float64Array(256);
  for (let i = 0; i < values.length; i++) hist[values[i]]++;
  const N = values.length;
  let sumAll = 0; for (let t = 0; t < 256; t++) sumAll += t * hist[t];
  let wB = 0, sumB = 0, best = 0, bestVar = -1;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (wB === 0) continue;
    const wF = N - wB; if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > bestVar) { bestVar = v; best = t; }
  }
  return best;
}

// numpy-style percentile (linear interpolation)
function m4Percentile(values, p) {
  const a = Float64Array.from(values).sort();
  if (!a.length) return 0;
  const pos = (a.length - 1) * p / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

function m4Contours(cv, mask, H, W) {
  const src = m4MatFromU8(cv, mask, H, W), cs = new cv.MatVector(), hi = new cv.Mat();
  cv.findContours(src, cs, hi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
  const out = [];
  for (let i = 0; i < cs.size(); i++) {
    const c = cs.get(i);
    out.push({ pts: Array.from(c.data32S), area: cv.contourArea(c) });
    c.delete();
  }
  src.delete(); cs.delete(); hi.delete();
  return out.sort((a, b) => b.area - a.area).map(c => c.pts);
}

function m4Watershed(cv, rgb3Mat, markers, H, W) {
  const mk = new cv.Mat(H, W, cv.CV_32S);
  mk.data32S.set(markers);
  cv.watershed(rgb3Mat, mk);
  const out = new Int32Array(mk.data32S);
  mk.delete();
  return out;
}

// ---------------------------------------------------------------------------
// PART 1 -- RGB (port of rgb_human_segmentation.py)
// ---------------------------------------------------------------------------

const M4_BINS = [8, 20, 20];
const M4_TEX_BINS = 10;
const M4_ITER = 3;

// separable blur of a 2-D Float64 array with reflect-101 border
function m4Blur2D(a, h, w, sigma) {
  const r = Math.ceil(3 * sigma), k = [];
  let s = 0; for (let i = -r; i <= r; i++) { const v = Math.exp(-i * i / (2 * sigma * sigma)); k.push(v); s += v; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const refl = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i; return i; };
  const tmp = new Float64Array(h * w), out = new Float64Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let j = -r; j <= r; j++) v += k[j + r] * a[y * w + refl(x + j, w)]; tmp[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let j = -r; j <= r; j++) v += k[j + r] * tmp[refl(y + j, h) * w + x]; out[y * w + x] = v;
  }
  return out;
}

function m4ColourHist(colIdx, sample) {
  const [B0, B1, B2] = M4_BINS, P = B1 * B2, h = new Float64Array(B0 * P);
  for (let i = 0; i < colIdx.length; i++) if (sample[i]) h[colIdx[i]]++;
  for (let l = 0; l < B0; l++) h.set(m4Blur2D(h.subarray(l * P, (l + 1) * P), B1, B2, 0.8), l * P);
  const out = new Float64Array(h.length);            // smooth along L (circular, like np.roll)
  for (let l = 0; l < B0; l++) for (let j = 0; j < P; j++)
    out[l * P + j] = 0.25 * h[((l + B0 - 1) % B0) * P + j] + 0.5 * h[l * P + j] + 0.25 * h[((l + 1) % B0) * P + j];
  let s = 0; for (let i = 0; i < out.length; i++) s += out[i];
  s = Math.max(s, 1e-9); for (let i = 0; i < out.length; i++) out[i] /= s;
  return out;
}

function m4TexHist(texIdx, sample) {
  const h = new Float64Array(M4_TEX_BINS);
  for (let i = 0; i < texIdx.length; i++) if (sample[i]) h[texIdx[i]]++;
  const c = new Float64Array(M4_TEX_BINS);          // np.convolve(h,[.25,.5,.25],'same')
  for (let i = 0; i < M4_TEX_BINS; i++)
    c[i] = 0.5 * h[i] + 0.25 * (i > 0 ? h[i - 1] : 0) + 0.25 * (i < M4_TEX_BINS - 1 ? h[i + 1] : 0) + 1e-3;
  let s = 0; for (const v of c) s += v;
  return c.map(v => v / s);
}

function m4KeepSeeded(cv, mask, seed, H, W) {
  const { n, labels, stats } = m4Components(cv, mask, H, W);
  if (n <= 1) return mask;
  const overlap = new Float64Array(n);
  for (let i = 0; i < H * W; i++) if (seed[i]) overlap[labels[i]]++;
  overlap[0] = 0;
  let best = 0;
  for (let i = 1; i < n; i++) if (overlap[i] > overlap[best]) best = i;
  const keep = new Uint8Array(n);
  if (overlap[best] === 0) {
    best = 1; for (let i = 2; i < n; i++) if (stats[i][4] > stats[best][4]) best = i;
    keep[best] = 1;
  } else {
    keep[best] = 1;
    for (let i = 1; i < n; i++) if (i !== best && overlap[i] > 0 && stats[i][4] >= 0.3 * stats[best][4]) keep[i] = 1;
  }
  const out = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) out[i] = keep[labels[i]] ? 255 : 0;
  return out;
}

function m4SegmentOneBox(cv, smoothMat, colIdx, texIdx, rect, H, W, dbg) {
  const [x, y, w, h] = rect, size = Math.hypot(w, h), N = H * W;
  const box = new Uint8Array(N);
  for (let yy = y; yy < y + h; yy++) box.fill(255, yy * W + x, yy * W + x + w);

  // background seed 1: ring outside the box
  const rw = Math.floor(0.15 * Math.max(w, h)) + 3, ring = new Uint8Array(N);
  for (let yy = Math.max(0, y - rw); yy < Math.min(H, y + h + rw); yy++)
    for (let xx = Math.max(0, x - rw); xx < Math.min(W, x + w + rw); xx++)
      if (!box[yy * W + xx]) ring[yy * W + xx] = 255;
  // background seed 2: the four box corners (triangles)
  const corners = new Uint8Array(N), cw = Math.floor(0.22 * w), ch = Math.floor(0.18 * h);
  for (let yy = y; yy < y + h; yy++) for (let xx = x; xx < x + w; xx++) {
    const dx = Math.min(xx - x, x + w - 1 - xx), dy = Math.min(yy - y, y + h - 1 - yy);
    if (cw > 0 && ch > 0 && dx / cw + dy / ch <= 1) corners[yy * W + xx] = 255;
  }
  const bgSeed = new Uint8Array(N);
  for (let i = 0; i < N; i++) bgSeed[i] = (ring[i] || corners[i]) ? 255 : 0;
  // person seed: ellipse around the torso
  const fgSeed = new Uint8Array(N), ecx = x + w / 2, ecy = y + 0.45 * h;
  const ea = Math.max(2, Math.floor(0.16 * w)), eb = Math.max(2, Math.floor(0.28 * h));
  for (let yy = y; yy < y + h; yy++) for (let xx = x; xx < x + w; xx++)
    if (((xx - ecx) / ea) ** 2 + ((yy - ecy) / eb) ** 2 <= 1) fgSeed[yy * W + xx] = 255;

  // first-pass samples: central vertical band vs ring + corners
  let fgSample = new Uint8Array(N), bgSample = bgSeed;
  const bx0 = x + Math.floor(0.3 * w), bx1 = x + Math.floor(0.7 * w);
  for (let yy = y; yy < y + h; yy++) for (let xx = bx0; xx < bx1; xx++)
    if (!corners[yy * W + xx]) fgSample[yy * W + xx] = 255;

  // horizontal spatial prior: 0.65 on the centre line -> 0.35 at the box sides
  const priorRow = new Float64Array(W);
  for (let xx = 0; xx < W; xx++) {
    const t = Math.min(1, Math.abs((xx - (x + w / 2)) / (w / 2)));
    priorRow[xx] = 0.65 - 0.30 * t * t;
  }

  const kSmall = size / 90, kClose = size / 30;
  const scoreMat = new cv.Mat(H, W, cv.CV_32F), blurred = new cv.Mat();
  const sigma = Math.max(1.0, size / 250);
  let mask = fgSeed, firstMask = null, scoreU8 = new Uint8Array(N);

  for (let it = 0; it <= M4_ITER; it++) {
    const pcF = m4ColourHist(colIdx, fgSample), pcB = m4ColourHist(colIdx, bgSample);
    const ptF = m4TexHist(texIdx, fgSample), ptB = m4TexHist(texIdx, bgSample);
    const sd = scoreMat.data32F;
    for (let i = 0; i < N; i++) {
      const pr = priorRow[i % W];
      const lf = pcF[colIdx[i]] * ptF[texIdx[i]] + 1e-12, lb = pcB[colIdx[i]] * ptB[texIdx[i]] + 1e-12;
      sd[i] = lf * pr / (lf * pr + lb * (1 - pr));
    }
    cv.GaussianBlur(scoreMat, blurred, new cv.Size(0, 0), sigma, sigma, cv.BORDER_DEFAULT);
    const bd = blurred.data32F;
    let m = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      const v = box[i] ? bd[i] : 0;
      scoreU8[i] = Math.max(0, Math.min(255, Math.floor(v * 255)));
      m[i] = v > 0.5 ? 255 : 0;
    }
    m = m4Morph(cv, m, H, W, cv.MORPH_OPEN, kSmall);
    m = m4Morph(cv, m, H, W, cv.MORPH_CLOSE, kClose);
    m = m4FillHoles(m, H, W);
    for (let i = 0; i < N; i++) if (!box[i]) m[i] = 0;
    m = m4KeepSeeded(cv, m, fgSeed, H, W);
    if (!firstMask) firstMask = m;
    mask = m;

    let cnt = 0; for (let i = 0; i < N; i++) if (m[i]) cnt++;
    if (cnt > 50) {
      fgSample = m4Erode(cv, m, H, W, kSmall);
      const dil = m4Dilate(cv, m, H, W, kClose), bs = new Uint8Array(N);
      for (let i = 0; i < N; i++) bs[i] = (bgSeed[i] || (box[i] && !dil[i])) ? 255 : 0;
      bgSample = bs;
    }
  }
  scoreMat.delete(); blurred.delete();

  // watershed boundary snapping
  const band = size / 22;
  let sureFg = m4Erode(cv, mask, H, W, band);
  let c = 0; for (let i = 0; i < N; i++) if (sureFg[i]) c++;
  if (c < 20) sureFg = m4Erode(cv, mask, H, W, kSmall);
  const dil = m4Dilate(cv, mask, H, W, band);
  const markers = new Int32Array(N), unknown = new Uint8Array(N), sureBg = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    sureBg[i] = (!dil[i] || !box[i]) ? 1 : 0;
    markers[i] = sureFg[i] ? 2 : (sureBg[i] ? 1 : 0);
    unknown[i] = markers[i] === 0 ? 1 : 0;
  }
  const ws = m4Watershed(cv, smoothMat, markers, H, W);
  let fin = new Uint8Array(N);
  for (let i = 0; i < N; i++) fin[i] = ws[i] === 2 ? 255 : 0;
  fin = m4FillHoles(fin, H, W);
  fin = m4Morph(cv, fin, H, W, cv.MORPH_OPEN, kSmall);
  fin = m4KeepSeeded(cv, fin, sureFg, H, W);

  for (let i = 0; i < N; i++) {
    if (scoreU8[i] > dbg.score[i]) dbg.score[i] = scoreU8[i];
    if (mask[i]) dbg.iterated[i] = 255;
    if (bgSeed[i]) dbg.seeds[i] = 1;
    if (fgSeed[i]) dbg.seeds[i] = 2;
    if (box[i]) dbg.trimap[i] = sureFg[i] ? 3 : (unknown[i] ? 2 : 1);
  }
  return fin;
}

/**
 * rgba: Uint8ClampedArray (H*W*4), rects: [[x,y,w,h], ...] (may be empty)
 * returns {mask, contours, debug}
 */
function m4SegmentRGB(cv, rgba, H, W, rects) {
  const N = H * W;
  if (!rects || !rects.length) {
    const mx = Math.floor(0.05 * W), my = Math.floor(0.05 * H);
    rects = [[mx, my, W - 2 * mx, H - 2 * my]];
  }
  rects = rects.map(([x, y, w, h]) => {
    x = Math.max(0, Math.round(x)); y = Math.max(0, Math.round(y));
    return [x, y, Math.min(Math.round(w), W - x), Math.min(Math.round(h), H - y)];
  }).filter(r => r[2] > 4 && r[3] > 4);

  const src = new cv.Mat(H, W, cv.CV_8UC4); src.data.set(rgba);
  const rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const smooth = new cv.Mat(); cv.bilateralFilter(rgb, smooth, 9, 40, 7, cv.BORDER_DEFAULT);
  const lab = new cv.Mat(); cv.cvtColor(smooth, lab, cv.COLOR_RGB2Lab);
  const labO = new cv.Mat(); cv.cvtColor(rgb, labO, cv.COLOR_RGB2Lab);

  // colour bin index per pixel
  const colIdx = new Int32Array(N), ld = lab.data;
  for (let i = 0; i < N; i++) {
    const li = (ld[3 * i] * M4_BINS[0]) >> 8, ai = (ld[3 * i + 1] * M4_BINS[1]) >> 8, bi = (ld[3 * i + 2] * M4_BINS[2]) >> 8;
    colIdx[i] = (li * M4_BINS[1] + ai) * M4_BINS[2] + bi;
  }
  // texture bin: local std of L in 7x7 (on the unsmoothed image)
  const L = new cv.Mat(H, W, cv.CV_32F), L2 = new cv.Mat(H, W, cv.CV_32F);
  for (let i = 0; i < N; i++) { const v = labO.data[3 * i]; L.data32F[i] = v; L2.data32F[i] = v * v; }
  const mL = new cv.Mat(), mL2 = new cv.Mat();
  cv.blur(L, mL, new cv.Size(7, 7)); cv.blur(L2, mL2, new cv.Size(7, 7));
  const texIdx = new Int32Array(N), den = Math.log1p(40);
  for (let i = 0; i < N; i++) {
    const sd = Math.sqrt(Math.max(mL2.data32F[i] - mL.data32F[i] ** 2, 0));
    texIdx[i] = Math.min(M4_TEX_BINS - 1, Math.max(0, Math.floor(Math.log1p(sd) / den * M4_TEX_BINS)));
  }
  [L, L2, mL, mL2, src, lab, labO, rgb].forEach(m => m.delete());

  const debug = { rects, score: new Uint8Array(N), iterated: new Uint8Array(N),
                  seeds: new Uint8Array(N), trimap: new Uint8Array(N) };
  let mask = new Uint8Array(N);
  for (const r of rects) {
    const m = m4SegmentOneBox(cv, smooth, colIdx, texIdx, r, H, W, debug);
    for (let i = 0; i < N; i++) if (m[i]) mask[i] = 255;
  }
  smooth.delete();
  return { mask, contours: m4Contours(cv, mask, H, W), debug };
}

// ---------------------------------------------------------------------------
// PART 2 -- THERMAL (port of thermal_human_segmentation.py)
// ---------------------------------------------------------------------------

function m4Hysteresis(cv, strength, low, high, H, W) {
  const weak = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) weak[i] = strength[i] > low ? 255 : 0;
  const { n, labels } = m4Components(cv, weak, H, W);
  const strong = new Uint8Array(n);
  for (let i = 0; i < H * W; i++) if (strength[i] > high && labels[i] > 0) strong[labels[i]] = 1;
  const out = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) out[i] = strong[labels[i]] ? 255 : 0;
  return out;
}

/**
 * rgba: Uint8ClampedArray; opts: {blackHot, upright, relHeat, tophatFraction}
 */
function m4SegmentThermal(cv, rgba, H, W, opts) {
  opts = Object.assign({ blackHot: false, upright: true, relHeat: 0.8, tophatFraction: 0.3,
                         minAreaFraction: 0.0008 }, opts || {});
  const N = H * W, diag = Math.hypot(H, W);
  const src = new cv.Mat(H, W, cv.CV_8UC4); src.data.set(rgba);
  // luminance like cv2.cvtColor(BGR2GRAY): the image is RGBA here
  const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  if (opts.blackHot) cv.bitwise_not(gray, gray);
  const den = new cv.Mat(); cv.medianBlur(gray, den, 5);

  // white top-hat with a large disk (opening done on a downscaled copy)
  const disk = opts.tophatFraction * Math.min(H, W), f = Math.min(1, 60 / disk);
  const small = new cv.Mat(), opened = new cv.Mat(), bg = new cv.Mat(), k = m4Kernel(cv, disk * f);
  cv.resize(den, small, new cv.Size(Math.max(1, Math.floor(W * f)), Math.max(1, Math.floor(H * f))), 0, 0, cv.INTER_AREA);
  cv.morphologyEx(small, opened, cv.MORPH_OPEN, k);
  cv.resize(opened, bg, new cv.Size(W, H), 0, 0, cv.INTER_LINEAR);
  const th = new cv.Mat(), thB = new cv.Mat();
  cv.subtract(den, bg, th);
  const s = Math.max(1.0, diag / 500);
  cv.GaussianBlur(th, thB, new cv.Size(0, 0), s, s, cv.BORDER_DEFAULT);
  const tophat = new Uint8Array(thB.data), background = new Uint8Array(bg.data);
  const grayArr = new Uint8Array(gray.data);

  // hysteresis threshold: HIGH = Otsu of the top-hat, LOW = HIGH / 2
  const high = Math.max(m4Otsu(tophat), 8), low = 0.5 * high;
  const initial = m4Hysteresis(cv, tophat, low, high, H, W);

  const kSmall = diag / 300, kClose = diag / 120, minArea = Math.max(30, opts.minAreaFraction * N);
  const clean = m => m4FillHoles(m4Morph(cv, m4Morph(cv, m, H, W, cv.MORPH_OPEN, kSmall), H, W, cv.MORPH_CLOSE, kClose), H, W);
  const personLike = ([x, y, w, h, area]) => {
    let ok = area >= minArea && area <= 0.25 * N && area / (w * h) > 0.25 && y > 0;
    if (opts.upright) ok = ok && h / w >= 1.0 && h / w <= 6.0;
    return ok;
  };
  const kept = new Uint8Array(N), log = { split: 0, rejected: 0 };
  const acceptOrSplit = (mask, depth) => {
    const { n, labels, stats } = m4Components(cv, mask, H, W);
    for (let c = 1; c < n; c++) {
      if (personLike(stats[c])) {
        for (let i = 0; i < N; i++) if (labels[i] === c) kept[i] = 255;
      } else if (depth < 3 && stats[c][4] >= 4 * minArea) {
        const vals = []; for (let i = 0; i < N; i++) if (labels[i] === c) vals.push(tophat[i]);
        const t = m4Otsu(vals), sub = new Uint8Array(N);
        for (let i = 0; i < N; i++) if (labels[i] === c && tophat[i] > t) sub[i] = 255;
        log.split++;
        acceptOrSplit(clean(sub), depth + 1);
      } else log.rejected++;
    }
  };
  acceptOrSplit(clean(initial), 0);

  // relative heat: drop blobs whose peak warmth (90th pct) << warmest blob's
  {
    const { n, labels } = m4Components(cv, kept, H, W);
    if (n > 2) {
      const vals = Array.from({ length: n }, () => []);
      for (let i = 0; i < N; i++) if (labels[i] > 0) vals[labels[i]].push(tophat[i]);
      const peaks = vals.map((v, i) => i === 0 ? -1 : m4Percentile(v, 90));
      const top = Math.max(...peaks);
      for (let c = 1; c < n; c++) if (peaks[c] < opts.relHeat * top) {
        for (let i = 0; i < N; i++) if (labels[i] === c) kept[i] = 0;
        log.rejected++;
      }
    }
  }

  // watershed snapping, one marker label per blob
  const sureFg = m4Erode(cv, kept, H, W, diag / 200), dil = m4Dilate(cv, kept, H, W, diag / 90);
  const markers = new Int32Array(N), unknown = new Uint8Array(N);
  const { n: nb, labels: bl } = m4Components(cv, sureFg, H, W);
  for (let i = 0; i < N; i++) {
    markers[i] = !dil[i] ? 1 : (bl[i] > 0 ? bl[i] + 1 : 0);
    unknown[i] = markers[i] === 0 ? 1 : 0;
  }
  const den3 = new cv.Mat(); cv.cvtColor(den, den3, cv.COLOR_GRAY2RGB);
  const ws = m4Watershed(cv, den3, markers, H, W);
  let fin = new Uint8Array(N);
  for (let i = 0; i < N; i++) fin[i] = ws[i] > 1 ? 255 : 0;
  fin = m4FillHoles(fin, H, W);
  fin = m4Morph(cv, fin, H, W, cv.MORPH_OPEN, kSmall);

  // normalised top-hat for display
  let mx = 1; for (const v of tophat) if (v > mx) mx = v;
  const thNorm = tophat.map(v => Math.round(v * 255 / mx));
  [src, gray, den, small, opened, bg, th, thB, den3, k].forEach(m => m.delete());
  return { mask: fin, contours: m4Contours(cv, fin, H, W),
           debug: { gray: grayArr, background, tophat: thNorm, initial, kept, sureFg, unknown,
                    high, low, split: log.split, rejected: log.rejected, nb: nb - 1 } };
}

// ---------------------------------------------------------------------------
// Comparison metrics (port of compare_with_sam2.py)
// ---------------------------------------------------------------------------

function m4Compare(cv, ours, ref, H, W, tol) {
  tol = tol || 2;
  const N = H * W;
  let a = 0, b = 0, inter = 0, uni = 0;
  for (let i = 0; i < N; i++) {
    const A = ours[i] > 0, B = ref[i] > 0;
    a += A; b += B; inter += A && B; uni += A || B;
  }
  const res = {
    IoU: uni ? inter / uni : 1, Dice: (a + b) ? 2 * inter / (a + b) : 1,
    Precision: a ? inter / a : 0, Recall: b ? inter / b : 0,
  };
  // 1-px boundaries via 3x3 rectangular erosion (same as the Python script)
  const bnd = m => {
    const src = new cv.Mat(H, W, cv.CV_8UC1); for (let i = 0; i < N; i++) src.data[i] = m[i] ? 1 : 0;
    const er = new cv.Mat(), k = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.erode(src, er, k, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, new cv.Scalar(0));
    const out = new Uint8Array(N); for (let i = 0; i < N; i++) out[i] = src.data[i] && !er.data[i] ? 1 : 0;
    src.delete(); er.delete(); k.delete();
    return out;
  };
  const dist = bd => {
    const src = new cv.Mat(H, W, cv.CV_8UC1); for (let i = 0; i < N; i++) src.data[i] = bd[i] ? 0 : 1;
    const d = new cv.Mat(); cv.distanceTransform(src, d, cv.DIST_L2, 5);
    const out = new Float32Array(d.data32F); src.delete(); d.delete(); return out;
  };
  const bA = bnd(ours), bB = bnd(ref);
  let hasA = false, hasB = false; for (let i = 0; i < N; i++) { hasA = hasA || bA[i]; hasB = hasB || bB[i]; }
  if (hasA && hasB) {
    const dB = dist(bB), dA = dist(bA), all = [];
    let pa = 0, na = 0, pb = 0, nbb = 0;
    for (let i = 0; i < N; i++) {
      if (bA[i]) { na++; if (dB[i] <= tol) pa++; all.push(dB[i]); }
      if (bB[i]) { nbb++; if (dA[i] <= tol) pb++; all.push(dA[i]); }
    }
    const bp = pa / na, br = pb / nbb;
    res["Boundary F1"] = (bp + br) ? 2 * bp * br / (bp + br) : 0;
    res["ASSD (px)"] = all.reduce((s, v) => s + v, 0) / all.length;
    res["HD95 (px)"] = m4Percentile(all, 95);
  } else {
    res["Boundary F1"] = 0; res["ASSD (px)"] = NaN; res["HD95 (px)"] = NaN;
  }
  return res;
}

// ---------------------------------------------------------------------------
// Browser UI (skipped under Node)
// ---------------------------------------------------------------------------

const M4_MAX_DIM = 640;                 // browser processing size
const M4_OPENCV_URLS = [
  "https://docs.opencv.org/4.10.0/opencv.js",
  "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js",
];

let m4cv = null, m4CvLoading = false;
const m4CvWaiters = [];

function m4LoadOpenCV(onReady) {
  if (m4cv) { onReady(m4cv); return; }
  m4CvWaiters.push(onReady);
  if (m4CvLoading) return;
  m4CvLoading = true;
  document.querySelectorAll(".m4-status").forEach(e => { e.textContent = "Loading OpenCV.js (≈8 MB, first time only)…"; });
  const ready = mod => {
    m4cv = mod;
    document.querySelectorAll(".m4-status").forEach(e => { e.textContent = "OpenCV.js ready. Upload an image or pick a sample."; });
    m4CvWaiters.splice(0).forEach(f => f(m4cv));
  };
  const tryUrl = idx => {
    const s = document.createElement("script");
    s.src = M4_OPENCV_URLS[idx]; s.async = true;
    s.onload = () => {
      const c = window.cv;
      if (c instanceof Promise) c.then(ready);
      else if (c && c.Mat) ready(c);
      else c.onRuntimeInitialized = () => ready(c);
    };
    s.onerror = () => {
      if (idx + 1 < M4_OPENCV_URLS.length) tryUrl(idx + 1);
      else document.querySelectorAll(".m4-status").forEach(e => { e.textContent = "Could not load OpenCV.js (check your connection)."; });
    };
    document.head.appendChild(s);
  };
  tryUrl(0);
}

function m4Jet(v) {                     // OpenCV-style JET colour map, v in 0..255
  const t = v / 255, r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3))),
        g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2))), b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
  return [r * 255, g * 255, b * 255];
}
function m4Inferno(v) {
  const t = v / 255;
  return [255 * Math.min(1, 1.6 * t), 255 * Math.max(0, Math.min(1, 1.9 * t - 0.8)), 255 * Math.max(0, Math.min(1, t < 0.4 ? 1.2 * t : 2.4 * t - 1.6))];
}

function m4PutPixels(canvas, W, H, fn) {
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const [r, g, b] = fn(i);
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return ctx;
}

function m4DrawContours(ctx, contours, color, width) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
  for (const pts of contours) {
    if (pts.length < 4) continue;
    ctx.beginPath(); ctx.moveTo(pts[0] + 0.5, pts[1] + 0.5);
    for (let j = 2; j < pts.length; j += 2) ctx.lineTo(pts[j] + 0.5, pts[j + 1] + 0.5);
    ctx.closePath(); ctx.stroke();
  }
}

function m4DrawBoundary(canvas, rgba, W, H, mask, contours, color) {
  const tint = color === "#00ff00" ? [0, 255, 0] : [255, 160, 0];
  const ctx = m4PutPixels(canvas, W, H, i => {
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    return mask && mask[i] ? [0.65 * r + 0.35 * tint[0], 0.65 * g + 0.35 * tint[1], 0.65 * b + 0.35 * tint[2]] : [r, g, b];
  });
  m4DrawContours(ctx, contours, color, Math.max(2, Math.hypot(W, H) / 450));
}

// Sets up one application block. kind = "rgb" | "thermal"
function m4SetupApp(kind) {
  const $ = id => document.getElementById(`m4${kind}_${id}`);
  const state = { rgba: null, W: 0, H: 0, rects: [], drag: null, result: null, name: "image" };
  const inputCanvas = $("canvasInput");

  function loadImage(img, name) {
    const sc = Math.min(1, M4_MAX_DIM / Math.max(img.naturalWidth, img.naturalHeight));
    state.W = Math.max(1, Math.round(img.naturalWidth * sc));
    state.H = Math.max(1, Math.round(img.naturalHeight * sc));
    const off = document.createElement("canvas");
    off.width = state.W; off.height = state.H;
    const octx = off.getContext("2d");
    octx.drawImage(img, 0, 0, state.W, state.H);
    state.rgba = octx.getImageData(0, 0, state.W, state.H).data;
    state.rects = []; state.result = null; state.name = name.replace(/\.[^.]+$/, "");
    redrawInput();
    ["canvasA", "canvasB", "canvasFinal", "canvasAgree"].forEach(id => { const c = $(id); if (c) { c.width = 1; c.height = 1; } });
    $("result").style.display = "none"; $("metrics").style.display = "none";
    $("run").disabled = !m4cv; $("download").disabled = true; $("sam2").disabled = true;
    $("status").textContent = `Loaded ${state.W}×${state.H} (processing size).` +
      (kind === "rgb" ? " Drag a box around each person (optional), then click “Find boundary”." : " Click “Find boundary”.");
    m4LoadOpenCV(() => { $("run").disabled = false; });
  }

  function redrawInput() {
    if (!state.rgba) return;
    const ctx = m4PutPixels(inputCanvas, state.W, state.H, i => [state.rgba[4 * i], state.rgba[4 * i + 1], state.rgba[4 * i + 2]]);
    ctx.lineWidth = Math.max(2, state.W / 300); ctx.strokeStyle = "#ffb400";
    const all = state.drag ? state.rects.concat([state.drag]) : state.rects;
    for (const [x, y, w, h] of all) ctx.strokeRect(x, y, w, h);
  }

  $("file").addEventListener("change", e => {
    const f = e.target.files[0]; if (!f) return;
    const img = new Image(); img.onload = () => loadImage(img, f.name); img.src = URL.createObjectURL(f);
  });
  document.querySelectorAll(`button.m4-sample[data-kind="${kind}"]`).forEach(btn => btn.addEventListener("click", () => {
    const img = new Image();
    img.onload = () => loadImage(img, btn.dataset.src.split("/").pop());
    img.onerror = () => { $("status").textContent = "Could not load the sample image."; };
    img.src = btn.dataset.src;
  }));

  if (kind === "rgb") {                                   // box drawing
    const pos = e => {
      const r = inputCanvas.getBoundingClientRect(), p = e.touches ? e.touches[0] : e;
      return [(p.clientX - r.left) * state.W / r.width, (p.clientY - r.top) * state.H / r.height];
    };
    const start = e => { if (!state.rgba) return; e.preventDefault(); const [x, y] = pos(e); state.drag = [x, y, 0, 0]; state.anchor = [x, y]; };
    const move = e => {
      if (!state.drag) return; e.preventDefault();
      const [x, y] = pos(e), [ax, ay] = state.anchor;
      state.drag = [Math.min(x, ax), Math.min(y, ay), Math.abs(x - ax), Math.abs(y - ay)]; redrawInput();
    };
    const end = () => {
      if (!state.drag) return;
      if (state.drag[2] > 8 && state.drag[3] > 8) state.rects.push(state.drag.map(Math.round));
      state.drag = null; redrawInput();
      $("status").textContent = `${state.rects.length} box(es) drawn.`;
    };
    inputCanvas.addEventListener("mousedown", start); inputCanvas.addEventListener("touchstart", start);
    window.addEventListener("mousemove", move); inputCanvas.addEventListener("touchmove", move);
    window.addEventListener("mouseup", end); inputCanvas.addEventListener("touchend", end);
    $("clear").addEventListener("click", () => { state.rects = []; redrawInput(); $("status").textContent = "Boxes cleared."; });
  }

  $("run").addEventListener("click", () => {
    if (!state.rgba || !m4cv) return;
    $("run").disabled = true; $("status").textContent = "Processing…";
    setTimeout(() => {
      const t0 = performance.now();
      const { W, H } = state;
      let res;
      try {
        res = kind === "rgb"
          ? m4SegmentRGB(m4cv, state.rgba, H, W, state.rects)
          : m4SegmentThermal(m4cv, state.rgba, H, W, {
              blackHot: $("blackhot").checked, upright: !$("anypose").checked,
              relHeat: parseFloat($("relheat").value), tophatFraction: parseFloat($("tophat").value) });
      } catch (err) {
        $("status").textContent = "Error: " + err; $("run").disabled = false; return;
      }
      const ms = performance.now() - t0;
      state.result = res;
      const d = res.debug;
      if (kind === "rgb") {
        m4PutPixels($("canvasA"), W, H, i => d.seeds[i] === 2 ? [60, 230, 60] : d.seeds[i] === 1 ? [230, 60, 60]
                     : [state.rgba[4 * i] / 2, state.rgba[4 * i + 1] / 2, state.rgba[4 * i + 2] / 2]);
        m4PutPixels($("canvasB"), W, H, i => m4Jet(d.score[i]));
      } else {
        m4PutPixels($("canvasA"), W, H, i => m4Inferno(d.tophat[i]));
        m4PutPixels($("canvasB"), W, H, i => d.kept[i] ? [255, 255, 255] : d.initial[i] ? [110, 110, 110] : [20, 20, 20]);
      }
      m4DrawBoundary($("canvasFinal"), state.rgba, W, H, res.mask, res.contours, "#00ff00");
      let area = 0; for (const v of res.mask) if (v) area++;
      $("result").style.display = "block";
      $("result").innerHTML =
        `<strong>${res.contours.length}</strong> boundary region(s) found, covering ${(100 * area / (W * H)).toFixed(1)}% of the image; ` +
        `boundary length(s): ${res.contours.map(c => c.length / 2).join(", ")} px. Processing time: ${ms.toFixed(0)} ms.` +
        (kind === "thermal" ? `<br><small>Hysteresis thresholds on the top-hat: low ${d.low.toFixed(1)}, high ${d.high}. ` +
          `Blobs re-thresholded: ${d.split}; rejected: ${d.rejected}.</small>` : "");
      $("status").textContent = "Done. Download the mask, or upload a SAM2 mask to compare.";
      $("run").disabled = false; $("download").disabled = false; $("sam2").disabled = false;
    }, 20);
  });

  $("download").addEventListener("click", () => {
    if (!state.result) return;
    const c = document.createElement("canvas");
    m4PutPixels(c, state.W, state.H, i => { const v = state.result.mask[i]; return [v, v, v]; });
    const a = document.createElement("a"); a.download = `${state.name}_mask.png`; a.href = c.toDataURL("image/png"); a.click();
  });

  $("sam2").addEventListener("change", e => {
    const f = e.target.files[0]; if (!f || !state.result) return;
    const img = new Image();
    img.onload = () => {
      const { W, H } = state, c = document.createElement("canvas");
      c.width = W; c.height = H;
      const ctx = c.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, W, H);
      const px = ctx.getImageData(0, 0, W, H).data;
      let transparent = false; for (let i = 3; i < px.length; i += 4) if (px[i] < 255) { transparent = true; break; }
      const ref = new Uint8Array(W * H);
      for (let i = 0; i < W * H; i++) {
        const v = transparent ? px[4 * i + 3] : 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
        ref[i] = v > 127 ? 255 : 0;
      }
      if ($("invert") && $("invert").checked) for (let i = 0; i < W * H; i++) ref[i] = 255 - ref[i];
      const ours = state.result.mask, m = m4Compare(m4cv, ours, ref, H, W, 2);
      m4PutPixels($("canvasAgree"), W, H, i => {
        const A = ours[i] > 0, B = ref[i] > 0, r = state.rgba[4 * i] / 3, g = state.rgba[4 * i + 1] / 3, b = state.rgba[4 * i + 2] / 3;
        return A && B ? [60, 200, 60] : A ? [230, 60, 60] : B ? [40, 120, 230] : [r, g, b];
      });
      const rows = Object.entries(m).map(([k, v]) => `<tr><th>${k}</th><td>${Number.isFinite(v) ? v.toFixed(4) : "—"}</td></tr>`).join("");
      $("metrics").style.display = "block";
      $("metricsTable").innerHTML = rows;
    };
    img.src = URL.createObjectURL(f);
  });
}

function m4Init() {
  if (typeof document === "undefined" || !document.getElementById("m4rgb_file")) return;
  m4SetupApp("rgb");
  m4SetupApp("thermal");
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { m4SegmentRGB, m4SegmentThermal, m4Compare, m4Otsu, m4Percentile };
} else if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", m4Init);
  else m4Init();
}
