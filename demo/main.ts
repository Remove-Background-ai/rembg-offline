import {
  removeBackground,
  subscribeToProgress,
  getCapabilityLadder,
  getActiveCapability,
  init,
} from "../dist/index.js";

const $ = (id: string) => document.getElementById(id)!;

subscribeToProgress(({ phase, progress }) => {
  $("progress").textContent = `${phase} ${progress}%`;
});

// Allow forcing a tier via ?device=wasm&dtype=q8 for testing
const params = new URLSearchParams(location.search);
const forcedDevice = params.get("device");
const forcedDtype = params.get("dtype");

async function setup() {
  const ladder = await getCapabilityLadder();
  $("cap").textContent = ladder.map((c: any) => `${c.device}/${c.dtype}`).join(" → ") +
    (forcedDevice ? ` (forced: ${forcedDevice}/${forcedDtype})` : "");
}

function makeSampleImage(): Promise<string> {
  // Dark "figure" on a light background — a clearly salient subject for RMBG
  const canvas = document.createElement("canvas");
  canvas.width = 800; canvas.height = 600;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#e8e2d8";
  ctx.fillRect(0, 0, 800, 600);
  ctx.fillStyle = "#20304a";
  ctx.beginPath(); ctx.arc(400, 190, 80, 0, Math.PI * 2); ctx.fill();       // head
  ctx.beginPath(); ctx.ellipse(400, 430, 150, 180, 0, 0, Math.PI * 2); ctx.fill(); // body
  return new Promise((resolve, reject) =>
    canvas.toBlob(b => b ? resolve(URL.createObjectURL(b)) : reject(new Error("toBlob failed")), "image/png")
  );
}

async function alphaStats(blobUrl: string): Promise<{ nonzeroPct: number; partialPct: number }> {
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = blobUrl; });
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let nonzero = 0, partial = 0;
  const total = canvas.width * canvas.height;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] !== 0) nonzero++;
    if (data[i] > 0 && data[i] < 255) partial++;
  }
  return { nonzeroPct: (nonzero / total) * 100, partialPct: (partial / total) * 100 };
}

async function run(url: string) {
  const status = $("status");
  status.className = "";
  status.textContent = "running…";
  ($("src") as HTMLImageElement).src = url;
  $("src").hidden = false;
  try {
    if (forcedDevice && forcedDtype) {
      await init({ capability: { device: forcedDevice, dtype: forcedDtype } as any });
    }
    const result = await removeBackground(url);
    const stats = await alphaStats(result.blobUrl);
    const active = getActiveCapability();
    const summary = {
      ok: true,
      backend: active ? `${active.device}/${active.dtype}` : "unknown",
      seconds: Number(result.processingTimeSeconds.toFixed(2)),
      width: result.width,
      height: result.height,
      nonzeroAlphaPct: Number(stats.nonzeroPct.toFixed(1)),
      partialAlphaPct: Number(stats.partialPct.toFixed(1)),
    };
    ($("out") as HTMLImageElement).src = result.blobUrl;
    $("out").hidden = false;
    status.textContent = JSON.stringify(summary);
    status.className = "done";
  } catch (e: any) {
    console.error(e);
    status.textContent = `ERROR: ${e?.message || e}`;
    status.className = "error";
  }
}

$("run-sample").addEventListener("click", async () => run(await makeSampleImage()));
($("file") as HTMLInputElement).addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) run(URL.createObjectURL(f));
});

setup();
