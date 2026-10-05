/* ============================================================================
   CSc 8830 - Computer Vision | Assignment 6 (web demo, Part A)
   Browser versions of
     module6/optical_flow.py     dense flow video + motion analysis
     module6/track_validate.py   two-frame Lucas-Kanade (with our own bilinear
                                 interpolation) vs. actual pixel locations
   Uses OpenCV.js (loaded by module4.js's loader when the tab opens).
   Every element lookup is optional-safe, so removing parts of the HTML
   never breaks the rest of the demo.
   ========================================================================== */

"use strict";

const M6 = {
  cv: null,
  procWidth: 320,
  step: 1 / 15,            // seconds between analysed frames (set from fps)
  fps: 30,                 // detected frame rate of the video
  rate: 15,                // requested analysed frames per second
  running: false,
  video: null,
  prevGray: null,          // cv.Mat of previous frame (gray)
  prevRGBA: null,
  W: 0, H: 0,
  series: [],              // {t, speed, panX, panY, objects}
};

const m6$ = id => document.getElementById(id);
const m6Set = (id, txt) => { const e = m6$(id); if (e) e.textContent = txt; };

function m6WithCV(fn) {
  if (M6.cv) { fn(M6.cv); return; }
  const done = cv => { M6.cv = cv; fn(cv); };
  if (typeof m4LoadOpenCV === "function") { m4LoadOpenCV(done); return; }
  if (window.cv && window.cv.Mat) { done(window.cv); return; }
  const s = document.createElement("script");               // fallback loader
  s.src = "https://docs.opencv.org/4.10.0/opencv.js";
  s.onload = () => {
    const c = window.cv;
    if (c instanceof Promise) c.then(done);
    else if (c.Mat) done(c);
    else c.onRuntimeInitialized = () => done(c);
  };
  document.head.appendChild(s);
}

// ---------------------------------------------------------------------------
// Frame grabbing: seek the <video> to exact times and copy the frame
// ---------------------------------------------------------------------------

function m6Seek(video, t) {
  return new Promise(resolve => {
    const done = () => { video.removeEventListener("seeked", done); resolve(); };
    video.addEventListener("seeked", done);
    video.currentTime = Math.min(t, Math.max(0, video.duration - 0.001));
  });
}

function m6Grab(video) {
  const c = document.createElement("canvas");
  c.width = M6.W; c.height = M6.H;
  const ctx = c.getContext("2d");
  ctx.drawImage(video, 0, 0, M6.W, M6.H);
  return ctx.getImageData(0, 0, M6.W, M6.H);
}

// Mean absolute difference between two frames (0 = identical, e.g. when the
// analysis rate is higher than the video's own frame rate).
function m6Diff(a, b) {
  let s = 0;
  for (let i = 0; i < a.data.length; i += 16) s += Math.abs(a.data[i] - b.data[i]);
  return s / (a.data.length / 16);
}

// Seek forward in steps of `gap` until the frame actually changes.
async function m6NextNewFrame(video, t, gap, ref, maxSteps = 8) {
  for (let k = 1; k <= maxSteps; k++) {
    const tk = t + k * gap;
    if (tk > video.duration) break;
    await m6Seek(video, tk);
    const f = m6Grab(video);
    if (m6Diff(f, ref) > 0.3) return { frame: f, t: tk };
  }
  return null;
}

// Estimate the video's frame rate: seek in 1/240 s steps over ~0.6 s and
// count how often the picture changes. Snap to common rates.
async function m6DetectFps(video) {
  const t0 = Math.min(1.0, video.duration / 3), span = Math.min(0.6, video.duration / 3), dt = 1 / 240;
  await m6Seek(video, t0);
  let prev = m6Grab(video), changes = [];
  for (let t = t0 + dt; t <= t0 + span; t += dt) {
    await m6Seek(video, t);
    const f = m6Grab(video);
    if (m6Diff(f, prev) > 0.3) changes.push(t);
    prev = f;
  }
  if (changes.length < 2) return 30;
  const est = (changes.length - 1) / (changes[changes.length - 1] - changes[0]);
  const common = [10, 12, 15, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120];
  return common.reduce((a, c) => (Math.abs(c - est) < Math.abs(a - est) ? c : a), common[0]);
}

// Time of the middle of frame n (seeking there gives exactly frame n).
const m6FrameTime = n => (n + 0.5) / M6.fps;

function m6GrayMat(cv, imgData) {
  const rgba = cv.matFromImageData(imgData), g = new cv.Mat();
  cv.cvtColor(rgba, g, cv.COLOR_RGBA2GRAY);
  rgba.delete();
  return g;
}

// ---------------------------------------------------------------------------
// Camera motion: similarity transform x' = a x - b y + tx, y' = b x + a y + ty
// fitted with RANSAC to tracked corners (OpenCV.js has no estimateAffinePartial2D)
// ---------------------------------------------------------------------------

function m6FitSimilarity(P, Q, idx) {
  // least squares on the given indices: unknowns [a, b, tx, ty]
  let n = idx.length, sx = 0, sy = 0, su = 0, sv = 0;
  for (const i of idx) { sx += P[i][0]; sy += P[i][1]; su += Q[i][0]; sv += Q[i][1]; }
  const mx = sx / n, my = sy / n, mu = su / n, mv = sv / n;
  let num_a = 0, num_b = 0, den = 0;
  for (const i of idx) {
    const x = P[i][0] - mx, y = P[i][1] - my, u = Q[i][0] - mu, v = Q[i][1] - mv;
    num_a += x * u + y * v; num_b += x * v - y * u; den += x * x + y * y;
  }
  if (den < 1e-9) return null;
  const a = num_a / den, b = num_b / den;
  return { a, b, tx: mu - (a * mx - b * my), ty: mv - (b * mx + a * my) };
}

function m6Ransac(P, Q, thr = 1.5, iters = 200) {
  const n = P.length;
  if (n < 6) return { a: 1, b: 0, tx: 0, ty: 0 };
  let best = null, bestIn = [];
  for (let k = 0; k < iters; k++) {
    const i = Math.floor(Math.random() * n); let j = Math.floor(Math.random() * n);
    if (i === j) continue;
    const m = m6FitSimilarity(P, Q, [i, j]); if (!m) continue;
    const inl = [];
    for (let r = 0; r < n; r++) {
      const ex = m.a * P[r][0] - m.b * P[r][1] + m.tx - Q[r][0];
      const ey = m.b * P[r][0] + m.a * P[r][1] + m.ty - Q[r][1];
      if (ex * ex + ey * ey < thr * thr) inl.push(r);
    }
    if (inl.length > bestIn.length) { bestIn = inl; best = m; }
  }
  if (bestIn.length < 6) return { a: 1, b: 0, tx: 0, ty: 0 };
  return m6FitSimilarity(P, Q, bestIn) || best;
}

function m6CameraMotion(cv, g1, g2) {
  const pts = new cv.Mat(), nxt = new cv.Mat(), st = new cv.Mat(), err = new cv.Mat();
  cv.goodFeaturesToTrack(g1, pts, 300, 0.01, 8);
  let model = { a: 1, b: 0, tx: 0, ty: 0 };
  if (pts.rows >= 10) {
    cv.calcOpticalFlowPyrLK(g1, g2, pts, nxt, st, err, new cv.Size(21, 21), 3);
    const P = [], Q = [];
    for (let i = 0; i < pts.rows; i++) if (st.data[i] === 1) {
      P.push([pts.data32F[2 * i], pts.data32F[2 * i + 1]]);
      Q.push([nxt.data32F[2 * i], nxt.data32F[2 * i + 1]]);
    }
    model = m6Ransac(P, Q);
  }
  [pts, nxt, st, err].forEach(m => m.delete());
  return model;
}

// ---------------------------------------------------------------------------
// One flow step: Farneback + analysis (same as optical_flow.py)
// ---------------------------------------------------------------------------

function m6Analyse(cv, g1, g2) {
  const W = M6.W, H = M6.H, N = W * H;
  const flowMat = new cv.Mat();
  cv.calcOpticalFlowFarneback(g1, g2, flowMat, 0.5, 3, 15, 3, 5, 1.2, 0);
  const flow = new Float32Array(flowMat.data32F); flowMat.delete();

  const cam = m6CameraMotion(cv, g1, g2);
  const eig = new cv.Mat(); cv.cornerMinEigenVal(g1, eig, 7, 3);
  const lam = eig.data32F, sorted = Float32Array.from(lam).sort();
  const lamThr = Math.max(1e-6, 0.05 * sorted[Math.floor(0.95 * (sorted.length - 1))]);
  const cx = W / 2, cy = H / 2;
  const camDx = cam.a * cx - cam.b * cy + cam.tx - cx, camDy = cam.b * cx + cam.a * cy + cam.ty - cy;
  const thr = Math.max(1.0, 0.3 * Math.hypot(camDx, camDy));

  const moving = new cv.Mat(H, W, cv.CV_8UC1);
  let movingSum = 0, movingCnt = 0;
  for (let y = 0, i = 0; y < H; y++) for (let x = 0; x < W; x++, i++) {
    const fx = flow[2 * i], fy = flow[2 * i + 1];
    const m = Math.hypot(fx, fy);
    if (m > 1) { movingSum += m; movingCnt++; }
    const rx = fx - (cam.a * x - cam.b * y + cam.tx - x), ry = fy - (cam.b * x + cam.a * y + cam.ty - y);
    moving.data[i] = (Math.hypot(rx, ry) > thr && lam[i] > lamThr) ? 255 : 0;
  }
  eig.delete();
  const k3 = cv.Mat.ones(3, 3, cv.CV_8U), k15 = cv.Mat.ones(15, 15, cv.CV_8U);
  cv.morphologyEx(moving, moving, cv.MORPH_OPEN, k3);
  cv.morphologyEx(moving, moving, cv.MORPH_CLOSE, k15);
  const lab = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
  const n = cv.connectedComponentsWithStats(moving, lab, stats, cent, 8, cv.CV_32S);
  const objects = [];
  for (let c = 1; c < n; c++) {
    const s = stats.data32S.slice(c * 5, c * 5 + 5);
    if (s[4] < 0.002 * N) continue;
    let sx = 0, sy = 0, cnt = 0;
    for (let i = 0; i < N; i++) if (lab.data32S[i] === c) {
      const x = i % W, y = (i / W) | 0;
      sx += flow[2 * i] - (cam.a * x - cam.b * y + cam.tx - x);
      sy += flow[2 * i + 1] - (cam.b * x + cam.a * y + cam.ty - y); cnt++;
    }
    objects.push({ box: [s[0], s[1], s[2], s[3]], dx: sx / cnt, dy: sy / cnt });
  }
  [moving, k3, k15, lab, stats, cent].forEach(m => m.delete());
  return {
    flow, objects, camDx, camDy,
    zoom: Math.hypot(cam.a, cam.b), rot: Math.atan2(cam.b, cam.a) * 180 / Math.PI,
    speed: movingCnt ? movingSum / movingCnt : 0, movingFrac: movingCnt / N,
  };
}

function m6Dir(dx, dy) {
  const ang = (Math.atan2(-dy, dx) * 180 / Math.PI + 360) % 360;
  return ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"][Math.floor(((ang + 22.5) % 360) / 45)];
}

function m6HsvToRgb(h, s, v) {
  const c = v * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = v - c;
  const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

function m6Draw(res, rgba) {
  const W = M6.W, H = M6.H, perS = 1 / M6.step;
  const left = m6$("m6_canvasFrame"), right = m6$("m6_canvasFlow");
  if (left) {
    left.width = W; left.height = H;
    const ctx = left.getContext("2d");
    ctx.putImageData(rgba, 0, 0);
    ctx.strokeStyle = "#00ff00"; ctx.lineWidth = 1;
    for (let y = 8; y < H; y += 12) for (let x = 8; x < W; x += 12) {
      const i = y * W + x, fx = res.flow[2 * i], fy = res.flow[2 * i + 1];
      if (fx * fx + fy * fy < 1) continue;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 3 * fx, y + 3 * fy); ctx.stroke();
    }
    ctx.font = "11px Arial"; ctx.lineWidth = 2;
    for (const o of res.objects) {
      const [x, y, w, h] = o.box;
      ctx.strokeStyle = "#ffb400"; ctx.strokeRect(x, y, w, h);
      const label = `${m6Dir(o.dx, o.dy)} ${(Math.hypot(o.dx, o.dy) * perS).toFixed(0)}px/s`;
      ctx.fillStyle = "#000"; ctx.fillText(label, x + 1, Math.max(11, y - 3) + 1);
      ctx.fillStyle = "#ffb400"; ctx.fillText(label, x, Math.max(11, y - 3));
    }
  }
  if (right) {
    right.width = W; right.height = H;
    const ctx = right.getContext("2d"), img = ctx.createImageData(W, H);
    const mags = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) mags[i] = Math.hypot(res.flow[2 * i], res.flow[2 * i + 1]);
    const mmax = Math.max(1e-3, Float32Array.from(mags).sort()[Math.floor(0.99 * (mags.length - 1))]);
    for (let i = 0; i < W * H; i++) {
      const ang = (Math.atan2(res.flow[2 * i + 1], res.flow[2 * i]) * 180 / Math.PI + 360) % 360;
      const [r, g, b] = m6HsvToRgb(ang, 1, Math.min(1, mags[i] / mmax));
      img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  const camSpeed = Math.hypot(res.camDx, res.camDy) * perS, parts = [];
  if (camSpeed > 5) parts.push(`pans ${m6Dir(res.camDx, res.camDy)} ${camSpeed.toFixed(0)} px/s`);
  if (Math.abs(res.zoom - 1) > 0.002) parts.push(res.zoom > 1 ? "zooms in" : "zooms out");
  if (Math.abs(res.rot) > 0.1) parts.push(`rotates ${(res.rot * perS).toFixed(0)} deg/s`);
  m6Set("m6_info",
    `t = ${M6.video.currentTime.toFixed(2)} s | camera: ${parts.length ? parts.join(", ") : "still"} | ` +
    `moving objects: ${res.objects.length} | speed of moving pixels: ${(res.speed * perS).toFixed(0)} px/s | ` +
    `moving area: ${(100 * res.movingFrac).toFixed(1)}%`);
}

function m6DrawChart() {
  const c = m6$("m6_chart"); if (!c) return;
  const W = c.width = c.clientWidth || 600, H = c.height = 170;
  const ctx = c.getContext("2d"); ctx.clearRect(0, 0, W, H);
  const S = M6.series; if (S.length < 2) return;
  const perS = 1 / M6.step, t0 = S[0].t, t1 = S[S.length - 1].t;
  const lines = [
    { key: s => s.speed * perS, color: "#2a6fdb", name: "speed of moving pixels (px/s)" },
    { key: s => s.panX * perS, color: "#2a9d5c", name: "camera pan, horizontal (px/s)" },
    { key: s => s.panY * perS, color: "#8e44ad", name: "camera pan, vertical (px/s)" },
  ];
  let lo = 0, hi = 1;
  for (const l of lines) for (const s of S) { const v = l.key(s); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const X = t => 40 + (W - 50) * (t - t0) / Math.max(1e-6, t1 - t0), Y = v => 10 + (H - 40) * (1 - (v - lo) / (hi - lo));
  ctx.strokeStyle = "#c9d6f2"; ctx.beginPath(); ctx.moveTo(40, Y(0)); ctx.lineTo(W - 10, Y(0)); ctx.stroke();
  ctx.fillStyle = "#4a6bb5"; ctx.font = "11px Arial";
  ctx.fillText(hi.toFixed(0), 4, 14); ctx.fillText(lo.toFixed(0), 4, H - 30);
  ctx.fillText(`${t0.toFixed(1)} s`, 40, H - 14); ctx.fillText(`${t1.toFixed(1)} s`, W - 50, H - 14);
  lines.forEach((l, k) => {
    ctx.strokeStyle = l.color; ctx.lineWidth = 1.5; ctx.beginPath();
    S.forEach((s, i) => { const x = X(s.t), y = Y(l.key(s)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.fillStyle = l.color; ctx.fillText(l.name, 50 + k * 190, H - 2);
  });
}

async function m6Run() {
  const cv = M6.cv, v = M6.video;
  if (!cv || !v || M6.running) return;
  M6.running = true; m6UpdateButtons();
  const dur = parseFloat((m6$("m6_duration") || {}).value || "30");
  M6.rate = parseFloat((m6$("m6_fps") || {}).value || "15");
  const every = Math.max(1, Math.round(M6.fps / M6.rate));      // analyse every n-th frame
  M6.step = every / M6.fps;                                       // seconds between analysed frames
  const n0 = Math.floor(v.currentTime * M6.fps), nEnd = Math.floor(Math.min(v.duration, v.currentTime + dur) * M6.fps) - 1;
  const startT = n0 / M6.fps;
  M6.series = [];
  if (M6.prevGray) M6.prevGray.delete();
  await m6Seek(v, m6FrameTime(n0));
  M6.prevRGBA = m6Grab(v); M6.prevGray = m6GrayMat(cv, M6.prevRGBA);
  let n = n0, t = startT;
  while (M6.running && n + every <= nEnd) {
    n += every; t = n / M6.fps;
    await m6Seek(v, m6FrameTime(n));
    const rgba = m6Grab(v);
    if (m6Diff(rgba, M6.prevRGBA) < 0.05) continue;               // duplicate frame (fps guess off): skip
    const g = m6GrayMat(cv, rgba);
    const res = m6Analyse(cv, M6.prevGray, g);
    m6Draw(res, M6.prevRGBA);
    M6.series.push({ t: t - M6.step, speed: res.speed, panX: res.camDx, panY: res.camDy, objects: res.objects.length });
    if (M6.series.length % 5 === 0) m6DrawChart();
    M6.prevGray.delete(); M6.prevGray = g; M6.prevRGBA = rgba;
    m6Set("m6_status", `Analysing… ${(t - startT).toFixed(1)} / ${((nEnd - n0) / M6.fps).toFixed(1)} s`);
    await new Promise(r => setTimeout(r, 0));
  }
  m6DrawChart();
  const S = M6.series;
  if (S.length) {
    const perS = 1 / M6.step, mean = f => S.reduce((a, s) => a + f(s), 0) / S.length;
    m6Set("m6_status", `Done: ${(t - startT).toFixed(1)} s analysed (${S.length} frame pairs). Average speed of moving pixels ` +
      `${(mean(s => s.speed) * perS).toFixed(0)} px/s, camera pan ${(mean(s => Math.hypot(s.panX, s.panY)) * perS).toFixed(0)} px/s, ` +
      `${mean(s => s.objects).toFixed(1)} moving objects per frame on average.`);
  }
  M6.running = false; m6UpdateButtons();
}

function m6UpdateButtons() {
  const has = !!(M6.cv && M6.video && M6.video.readyState >= 1 && M6.fps);
  const set = (id, en) => { const e = m6$(id); if (e) e.disabled = !en; };
  set("m6_run", has && !M6.running); set("m6_stop", M6.running); set("m6_track", has && !M6.running);
}

// ---------------------------------------------------------------------------
// Two-frame tracking check (port of track_validate.py)
// ---------------------------------------------------------------------------

const M6_WIN = 21, M6_LEVELS = 3, M6_ITERS = 20, M6_EPS = 0.01;

function m6Bilinear(I, W, H, x, y) {
  x = Math.min(Math.max(x, 0), W - 1.001); y = Math.min(Math.max(y, 0), H - 1.001);
  const j = Math.floor(x), i = Math.floor(y), a = x - j, b = y - i, p = i * W + j;
  return (1 - a) * (1 - b) * I[p] + a * (1 - b) * I[p + 1] + (1 - a) * b * I[p + W] + a * b * I[p + W + 1];
}

function m6Gradients(I, W, H) {
  const Ix = new Float64Array(W * H), Iy = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 1; x < W - 1; x++) Ix[y * W + x] = (I[y * W + x + 1] - I[y * W + x - 1]) / 2;
  for (let y = 1; y < H - 1; y++) for (let x = 0; x < W; x++) Iy[y * W + x] = (I[(y + 1) * W + x] - I[(y - 1) * W + x]) / 2;
  return [Ix, Iy];
}

function m6Pyramid(cv, gMat, levels) {
  const out = [];
  let cur = gMat.clone();
  for (let L = 0; L <= levels; L++) {
    out.push({ I: Float64Array.from(cur.data), W: cur.cols, H: cur.rows });
    if (L < levels) { const d = new cv.Mat(); cv.pyrDown(cur, d); cur.delete(); cur = d; }
  }
  cur.delete();
  return out;
}

function m6LKLevel(L1, L2, G, x, y, u, v, record) {
  const r = M6_WIN >> 1; let a11 = 0, a12 = 0, a22 = 0;
  const T = [], GX = [], GY = [], WX = [], WY = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    const wx = x + dx, wy = y + dy;
    const gx = m6Bilinear(G[0], L1.W, L1.H, wx, wy), gy = m6Bilinear(G[1], L1.W, L1.H, wx, wy);
    T.push(m6Bilinear(L1.I, L1.W, L1.H, wx, wy)); GX.push(gx); GY.push(gy); WX.push(wx); WY.push(wy);
    a11 += gx * gx; a12 += gx * gy; a22 += gy * gy;
  }
  const det = a11 * a22 - a12 * a12, tr = a11 + a22;
  const lmin = tr / 2 - Math.sqrt(Math.max(0, tr * tr / 4 - det));
  if (lmin < 1e-3 * M6_WIN * M6_WIN) return { u, v, ok: false };
  let first = null;
  for (let k = 0; k < M6_ITERS; k++) {
    let b1 = 0, b2 = 0;
    for (let n = 0; n < T.length; n++) {
      const It = m6Bilinear(L2.I, L2.W, L2.H, WX[n] + u, WY[n] + v) - T[n];
      b1 -= GX[n] * It; b2 -= GY[n] * It;
    }
    const du = (a22 * b1 - a12 * b2) / det, dv = (-a12 * b1 + a11 * b2) / det;
    if (record && !first) first = { A: [[a11, a12], [a12, a22]], b: [b1, b2], d: [du, dv], start: [u, v] };
    u += du; v += dv;
    if (Math.hypot(du, dv) < M6_EPS) break;
  }
  return { u, v, ok: true, first };
}

function m6NCC(cv, g1, g2, x, y, search = 40, half = 10) {
  const W = g1.cols, H = g1.rows, xi = Math.round(x), yi = Math.round(y);
  if (xi < half || yi < half || xi >= W - half || yi >= H - half) return null;
  const x0 = Math.max(0, xi - half - search), y0 = Math.max(0, yi - half - search);
  const x1 = Math.min(W, xi + half + search + 1), y1 = Math.min(H, yi + half + search + 1);
  const tpl = g1.roi(new cv.Rect(xi - half, yi - half, 2 * half + 1, 2 * half + 1));
  const roi = g2.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  const R = new cv.Mat();
  cv.matchTemplate(roi, tpl, R, cv.TM_CCOEFF_NORMED);
  const mm = cv.minMaxLoc(R), mx = mm.maxLoc.x, my = mm.maxLoc.y, Rw = R.cols, f = R.data32F;
  const sub = (cm, c0, cp) => { const den = cm - 2 * c0 + cp; return Math.abs(den) < 1e-9 ? 0 : 0.5 * (cm - cp) / den; };
  const ox = mx > 0 && mx < Rw - 1 ? sub(f[my * Rw + mx - 1], f[my * Rw + mx], f[my * Rw + mx + 1]) : 0;
  const oy = my > 0 && my < R.rows - 1 ? sub(f[(my - 1) * Rw + mx], f[my * Rw + mx], f[(my + 1) * Rw + mx]) : 0;
  const out = { x: x0 + mx + ox + half + (x - xi), y: y0 + my + oy + half + (y - yi), score: mm.maxVal };
  [tpl, roi, R].forEach(m => m.delete());
  return out;
}

async function m6Track() {
  const cv = M6.cv, v = M6.video; if (!cv || !v) return;
  const gapFrames = parseInt((m6$("m6_gap") || {}).value || "1", 10);
  m6Set("m6_trackStatus", "Tracking…");
  const n1 = Math.floor(v.currentTime * M6.fps);
  await m6Seek(v, m6FrameTime(n1)); const f1 = m6Grab(v);
  await m6Seek(v, m6FrameTime(n1 + gapFrames)); let f2 = m6Grab(v);
  let t = n1 / M6.fps, t2 = (n1 + gapFrames) / M6.fps;
  if (m6Diff(f1, f2) < 0.05) {                            // same picture: frame-rate guess was off
    const nxt = await m6NextNewFrame(v, m6FrameTime(n1), 1 / M6.fps, f1);
    if (!nxt) { m6Set("m6_trackStatus", "The video does not change after this moment; try another one."); return; }
    f2 = nxt.frame; t2 = nxt.t;
  }
  const g1 = m6GrayMat(cv, f1), g2 = m6GrayMat(cv, f2), W = M6.W, H = M6.H;

  // points: corners that actually move (> 1 px), like --moving_only
  const pts = new cv.Mat(); cv.goodFeaturesToTrack(g1, pts, 200, 0.01, 10);
  const cand = [];
  for (let i = 0; i < pts.rows; i++) {
    const x = pts.data32F[2 * i], y = pts.data32F[2 * i + 1];
    if (x > 30 && y > 30 && x < W - 30 && y < H - 30) cand.push([x, y]);
  }
  pts.delete();
  const P1 = m6Pyramid(cv, g1, M6_LEVELS), P2 = m6Pyramid(cv, g2, M6_LEVELS);
  const G = P1.map(L => m6Gradients(L.I, L.W, L.H));
  const rows = [];
  for (const [x, y] of cand) {
    let u = 0, vv = 0, ok = true, first = null;
    for (let L = M6_LEVELS; L >= 0; L--) {
      const s = 2 ** L, r = m6LKLevel(P1[L], P2[L], G[L], x / s, y / s, u, vv, L === 0);
      ok = ok && r.ok; u = r.u; vv = r.v; if (L === 0) first = r.first;
      if (L > 0) { u *= 2; vv *= 2; }
    }
    const act = m6NCC(cv, g1, g2, x, y);
    if (!ok || !act || act.score < 0.9) continue;
    rows.push({ x, y, px: x + u, py: y + vv, ax: act.x, ay: act.y, score: act.score,
                err: Math.hypot(x + u - act.x, y + vv - act.y), disp: Math.hypot(act.x - x, act.y - y), first });
  }
  g1.delete(); g2.delete();
  let use = rows.filter(r => r.disp > 1); if (use.length < 5) use = rows;
  use = use.slice(0, 30);
  if (!use.length) { m6Set("m6_trackStatus", "No trackable points here; try another moment."); return; }

  // worked example: among the points that moved at least the median amount,
  // the one with the best-textured window (largest smallest eigenvalue of A)
  const medDisp = use.map(r => r.disp).sort((p, q) => p - q)[Math.floor(use.length / 2)];
  const lmin = r => { if (!r.first) return 0; const [[a, b], [, c]] = r.first.A; return (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b); };
  const ex = use.filter(r => r.disp >= medDisp).reduce((a, r) => (lmin(r) > lmin(a) ? r : a));
  const c = m6$("m6_canvasTrack");
  if (c) {
    c.width = W; c.height = H; const ctx = c.getContext("2d"); ctx.putImageData(f2, 0, 0);
    for (const r of use) {
      ctx.strokeStyle = "#ffff00"; ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(r.px, r.py); ctx.stroke();
      ctx.strokeStyle = "#00ff00"; ctx.beginPath(); ctx.arc(r.ax, r.ay, 4, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = "#ff0000"; ctx.beginPath();
      ctx.moveTo(r.px - 3, r.py); ctx.lineTo(r.px + 3, r.py); ctx.moveTo(r.px, r.py - 3); ctx.lineTo(r.px, r.py + 3); ctx.stroke();
    }
  }
  const z = m6$("m6_canvasZoom");
  if (z) {                                     // 25 x 25 px around the example, nearest-neighbour zoom
    const S = 12, R = 12; z.width = z.height = (2 * R + 1) * S; const ctx = z.getContext("2d");
    const cx0 = Math.round(ex.x) - R, cy0 = Math.round(ex.y) - R;
    for (let yy = 0; yy <= 2 * R; yy++) for (let xx = 0; xx <= 2 * R; xx++) {
      const X = Math.min(W - 1, Math.max(0, cx0 + xx)), Y = Math.min(H - 1, Math.max(0, cy0 + yy)), p = 4 * (Y * W + X);
      ctx.fillStyle = `rgb(${f2.data[p]},${f2.data[p + 1]},${f2.data[p + 2]})`; ctx.fillRect(xx * S, yy * S, S, S);
    }
    const P = (x, y) => [(x - cx0 + 0.5) * S, (y - cy0 + 0.5) * S];
    ctx.lineWidth = 2;
    let [a, b] = P(ex.x, ex.y); ctx.strokeStyle = "#ffff00"; ctx.beginPath(); ctx.arc(a, b, 7, 0, 2 * Math.PI); ctx.stroke();
    [a, b] = P(ex.ax, ex.ay); ctx.strokeStyle = "#00ff00"; ctx.beginPath(); ctx.arc(a, b, 10, 0, 2 * Math.PI); ctx.stroke();
    [a, b] = P(ex.px, ex.py); ctx.strokeStyle = "#ff0000"; ctx.beginPath();
    ctx.moveTo(a - 9, b); ctx.lineTo(a + 9, b); ctx.moveTo(a, b - 9); ctx.lineTo(a, b + 9); ctx.stroke();
  }
  const mean = f => use.reduce((s, r) => s + f(r), 0) / use.length;
  const fx = ex.first;
  m6Set("m6_trackStatus",
    `Frames ${n1} and ${n1 + gapFrames} (at ${M6.fps} fps: ${t.toFixed(3)} s and ${t2.toFixed(3)} s): ${use.length} points, mean displacement ` +
    `${mean(r => r.disp).toFixed(2)} px, LK error vs actual: mean ${mean(r => r.err).toFixed(3)} px, ` +
    `max ${Math.max(...use.map(r => r.err)).toFixed(3)} px.`);
  const tb = m6$("m6_trackTable");
  if (tb) {
    let html = "<tr><th>#</th><th>start (x, y)</th><th>LK prediction</th><th>actual (NCC)</th><th>error (px)</th></tr>";
    use.slice(0, 10).forEach((r, i) => {
      html += `<tr><td>${i + 1}</td><td>(${r.x.toFixed(1)}, ${r.y.toFixed(1)})</td><td>(${r.px.toFixed(2)}, ${r.py.toFixed(2)})</td>` +
              `<td>(${r.ax.toFixed(2)}, ${r.ay.toFixed(2)})</td><td>${r.err.toFixed(3)}</td></tr>`;
    });
    tb.innerHTML = html;
  }
  const we = m6$("m6_worked");
  if (we && fx) {
    we.textContent =
      `Zoomed point: start (${ex.x.toFixed(2)}, ${ex.y.toFixed(2)})\n` +
      `A = [[${fx.A[0][0].toFixed(0)}, ${fx.A[0][1].toFixed(0)}], [${fx.A[1][0].toFixed(0)}, ${fx.A[1][1].toFixed(0)}]]   ` +
      `b = [${fx.b[0].toFixed(0)}, ${fx.b[1].toFixed(0)}]\n` +
      `first full-resolution step d = A⁻¹b = (${fx.d[0].toFixed(3)}, ${fx.d[1].toFixed(3)}) from guess (${fx.start[0].toFixed(2)}, ${fx.start[1].toFixed(2)})\n` +
      `final flow (u, v) = (${(ex.px - ex.x).toFixed(3)}, ${(ex.py - ex.y).toFixed(3)}) → predicted (${ex.px.toFixed(2)}, ${ex.py.toFixed(2)}), ` +
      `actual (${ex.ax.toFixed(2)}, ${ex.ay.toFixed(2)}), error ${ex.err.toFixed(3)} px`;
  }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

function m6Init() {
  const file = m6$("m6_file"); if (!file) return;
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.preload = "auto";
  M6.video = v;
  const preview = m6$("m6_preview");

  file.addEventListener("change", e => {
    const f = e.target.files[0]; if (!f) return;
    M6.running = false;
    const url = URL.createObjectURL(f);
    v.src = url; if (preview) preview.src = url;
    m6Set("m6_status", "Loading video…");
    v.onloadedmetadata = () => {
      M6.procWidth = parseInt((m6$("m6_width") || {}).value || "320", 10);
      M6.W = M6.procWidth; M6.H = Math.round(v.videoHeight * M6.W / v.videoWidth / 2) * 2;
      m6WithCV(async () => {
        m6Set("m6_status", "Measuring the video's frame rate…");
        M6.fps = await m6DetectFps(v);
        m6Set("m6_status", `Video ${v.videoWidth}×${v.videoHeight}, ${v.duration.toFixed(1)} s, about ${M6.fps} fps. ` +
          `Seek the preview to where you want to start, then click “Analyse”.`);
        m6UpdateButtons();
      });
    };
    v.onerror = () => m6Set("m6_status", "This browser can't decode the video (try an H.264 .mp4).");
  });
  if (preview) preview.addEventListener("seeked", () => { if (!M6.running) v.currentTime = preview.currentTime; });
  const width = m6$("m6_width");
  if (width) width.addEventListener("change", () => {
    M6.procWidth = parseInt(width.value, 10);
    if (v.videoWidth) { M6.W = M6.procWidth; M6.H = Math.round(v.videoHeight * M6.W / v.videoWidth / 2) * 2; }
  });
  const on = (id, fn) => { const e = m6$(id); if (e) e.addEventListener("click", fn); };
  on("m6_run", () => { if (preview) v.currentTime = preview.currentTime; m6Run(); });
  on("m6_stop", () => { M6.running = false; });
  on("m6_track", () => { if (preview) v.currentTime = preview.currentTime; m6Track(); });
  m6UpdateButtons();
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", m6Init);
  else m6Init();
}
