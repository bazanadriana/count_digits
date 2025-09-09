import fs from "node:fs";
import path from "node:path";
import { glob } from "glob";
import sharp from "sharp";
import mnist from "mnist";

/* -------------------- Load a small MNIST subset -------------------- */
function getMnistData(n = 6000) {
  const set = mnist.set(n, 0);
  const X = set.training.map(s => s.input.map(v => v * 16)); // 0..1 -> 0..16
  const y = set.training.map(s => s.output.findIndex(v => v === 1));
  return { X, y };
}

/* ------------------------- Z-score helpers ------------------------- */
function zscoreFit(X) {
  const n = X.length, d = X[0].length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j];
    mean[j] = s / n;
  }
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const diff = X[i][j] - mean[j];
      s += diff * diff;
    }
    std[j] = Math.sqrt(s / Math.max(1, n - 1)) || 1;
  }
  return { mean, std };
}
const zscoreTransform = (X, stats) =>
  X.map(row => row.map((v, j) => (v - stats.mean[j]) / stats.std[j]));

/* -------------------- Tiny KNN (no dependencies) ------------------- */
function euclid(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}
function predictKNN(Xtrain, ytrain, x, k = 3) {
  // find k nearest neighbors
  const dists = [];
  for (let i = 0; i < Xtrain.length; i++) {
    dists.push([euclid(Xtrain[i], x), ytrain[i]]);
  }
  dists.sort((p, q) => p[0] - q[0]);
  // inverse-distance weighted vote
  const votes = new Array(10).fill(0);
  const kk = Math.min(k, dists.length);
  for (let i = 0; i < kk; i++) {
    const [d, label] = dists[i];
    const w = 1 / (1e-6 + d);
    votes[label] += w;
  }
  let best = 0, bestScore = -1;
  for (let c = 0; c < 10; c++) {
    if (votes[c] > bestScore) { bestScore = votes[c]; best = c; }
  }
  return best;
}

/* -------- Image -> centered 28x28 binary (Otsu) -> vector 0..16 ----- */
async function imageToVector(filePath) {
  const meta = await sharp(filePath).metadata();
  const w = meta.width ?? 0, h = meta.height ?? 0;

  let img = sharp(filePath).removeAlpha().greyscale();

  // pad to square (black background) then resize
  const side = Math.max(1, w, h);
  const left = Math.floor((side - w) / 2);
  const top = Math.floor((side - h) / 2);
  const right = side - w - left;
  const bottom = side - h - top;

  img = img.extend({ top, bottom, left, right, background: { r: 0, g: 0, b: 0 } })
           .resize(28, 28, { fit: "fill" })
           .blur(0.5);

  // 28x28 grayscale
  const buf = await img.raw().toBuffer();

  // Otsu threshold
  const hist = new Array(256).fill(0);
  for (const v of buf) hist[v]++;
  const total = buf.length;
  let sumAll = 0; for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  let sumB = 0, wB = 0, maxBetween = 0, thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (wB === 0) continue;
    const wF = total - wB; if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxBetween) { maxBetween = between; thr = t; }
  }
  // binarize and map to 0..16
  const out = new Array(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = buf[i] > thr ? 16 : 0;
  return out;
}

/* ------------------------------- Main -------------------------------- */
async function main() {
  let folderArg = process.argv[2];
  const previewFlag = process.argv.includes("--preview");
  if (!folderArg || folderArg === "--preview") {
    console.error("Usage: node count_digits.js /path/to/digits [--preview]");
    process.exit(1);
  }
  folderArg = folderArg.replace(/^"(.*)"$/, "$1").replace(/^'(.*)'$/, "$1");
  const folder = path.resolve(folderArg);
  if (!fs.existsSync(folder) || !fs.statSync(folder).isDirectory()) {
    console.error("Folder not found:", folder); process.exit(1);
  }

  // Training data
  const { X, y } = getMnistData(6000);
  const stats = zscoreFit(X);
  const Xs = zscoreTransform(X, stats);

  // Collect images
  const patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.gif", "**/*.tif", "**/*.tiff"];
  let files = [];
  for (const p of patterns) files = files.concat(await glob(path.join(folder, p), { nodir: true }));
  files.sort();
  if (!files.length) { console.error("No images found in:", folder); process.exit(1); }

  // Predict
  const counts = Array(10).fill(0);
  const previews = [];
  for (const f of files) {
    try {
      const vec = await imageToVector(f);
      const z = zscoreTransform([vec], stats)[0];
      const pred = predictKNN(Xs, y, z, 3);
      counts[pred] += 1;
      if (previewFlag && previews.length < 20) previews.push(`${path.basename(f)} -> ${pred}`);
    } catch (e) {
      console.warn(`Skipped ${f}: ${e?.message || e}`);
    }
  }

  console.log("\n✅ Final 10-element array [0..9]:");
  console.log(counts);
  console.log(`\nTotal files counted: ${counts.reduce((a, b) => a + b, 0)}\n`);
  if (previewFlag) {
    console.log("🔎 Sample predictions:");
    previews.forEach(line => console.log(line));
  }

  fs.writeFileSync(
    path.resolve("digit_counts.csv"),
    `${[...Array(10).keys()].join(",")}\n${counts.join(",")}\n`,
    "utf8"
  );
  console.log("Saved -> digit_counts.csv");
}

main().catch(err => { console.error(err); process.exit(1); });