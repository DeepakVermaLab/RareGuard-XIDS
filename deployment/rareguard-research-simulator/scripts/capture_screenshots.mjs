#!/usr/bin/env node
/* Capture deterministic, high-resolution paper screenshots through Chrome CDP. */

import { spawn } from "node:child_process";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const siteRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const outputDir = join(siteRoot, "screenshots");
const profileDir = await mkdtemp(join(tmpdir(), "rareguard-capture-"));
const port = 9337;
const baseUrl = "http://127.0.0.1:4173/";

await mkdir(outputDir, { recursive: true });

const chrome = spawn(chromePath, [
  "--headless=new",
  "--disable-gpu",
  "--hide-scrollbars",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profileDir}`,
  "--no-first-run",
  "--no-default-browser-check",
  baseUrl,
], { stdio: "ignore" });

const sleep = (ms) => new Promise((resolvePromise) => setTimeout(resolvePromise, ms));

async function waitForJson(url, attempts = 60) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
    } catch {}
    await sleep(100);
  }
  throw new Error(`Chrome debugging endpoint did not become ready: ${url}`);
}

const targets = await waitForJson(`http://127.0.0.1:${port}/json/list`);
const pageTarget = targets.find((target) => target.type === "page");
if (!pageTarget) throw new Error("No Chrome page target found");

const socket = new WebSocket(pageTarget.webSocketDebuggerUrl);
await new Promise((resolvePromise, reject) => {
  socket.addEventListener("open", resolvePromise, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let messageId = 0;
const pending = new Map();
const eventWaiters = new Map();

socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const { resolve: resolvePending, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolvePending(message.result || {});
    return;
  }
  if (message.method && eventWaiters.has(message.method)) {
    const waiters = eventWaiters.get(message.method);
    eventWaiters.delete(message.method);
    waiters.forEach((resolveWaiter) => resolveWaiter(message.params || {}));
  }
});

function command(method, params = {}) {
  messageId += 1;
  const id = messageId;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolvePromise, reject) => pending.set(id, { resolve: resolvePromise, reject }));
}

function waitForEvent(method) {
  return new Promise((resolvePromise) => {
    const waiters = eventWaiters.get(method) || [];
    waiters.push(resolvePromise);
    eventWaiters.set(method, waiters);
  });
}

async function evaluate(expression, awaitPromise = false) {
  const result = await command("Runtime.evaluate", {
    expression,
    awaitPromise,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || "Runtime evaluation failed");
  return result.result?.value;
}

async function navigate(url) {
  const loaded = waitForEvent("Page.loadEventFired");
  await command("Page.navigate", { url });
  await loaded;
  await evaluate(`new Promise((resolve) => {
    const started = Date.now();
    const check = () => {
      const overlay = document.getElementById('loadingOverlay');
      if (overlay && overlay.classList.contains('hidden')) return requestAnimationFrame(() => requestAnimationFrame(resolve));
      if (Date.now() - started > 10000) return resolve();
      setTimeout(check, 50);
    };
    check();
  })`, true);
  await evaluate("new Promise((resolve) => setTimeout(resolve, 400))", true);
}

async function captureViewport(name, sectionId) {
  await command("Emulation.setDeviceMetricsOverride", {
    width: 1920,
    height: 1200,
    deviceScaleFactor: 2,
    mobile: false,
  });
  await navigate(baseUrl);
  const scrollInfo = await evaluate(`(() => {
    const target = document.getElementById(${JSON.stringify(sectionId)});
    const top = ${JSON.stringify(sectionId)} === 'top' ? 0 : Math.max(0, target.offsetTop - 72);
    document.documentElement.style.scrollBehavior = 'auto';
    window.scrollTo({top, left: 0, behavior: 'instant'});
    return {top, scrollY: window.scrollY, height: document.documentElement.scrollHeight};
  })()`);
  await evaluate("new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))", true);
  const shot = await command("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: true,
    clip: { x: 0, y: scrollInfo.top, width: 1920, height: 1200, scale: 1 },
  });
  const outputPath = join(outputDir, `${name}.png`);
  await writeFile(outputPath, Buffer.from(shot.data, "base64"));
  console.log(`${name}.png scroll=${scrollInfo.scrollY} pageHeight=${scrollInfo.height}`);
}

async function captureFullPage() {
  await command("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await navigate(baseUrl);
  await evaluate("window.scrollTo(0, 0)");
  const metrics = await command("Page.getLayoutMetrics");
  const width = Math.ceil(metrics.cssContentSize.width);
  const height = Math.ceil(metrics.cssContentSize.height);
  const shot = await command("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width, height, scale: 1 },
  });
  await writeFile(join(outputDir, "06_full_console.png"), Buffer.from(shot.data, "base64"));
  console.log(`06_full_console.png fullPage=${width}x${height}`);
}

try {
  await command("Page.enable");
  await command("Runtime.enable");
  await captureViewport("01_overview", "top");
  await captureViewport("02_live_replay", "simulation");
  await captureViewport("03_comparative_evidence", "evidence");
  await captureViewport("04_stress_tests", "robustness");
  await captureViewport("05_reproducibility", "reproducibility");
  await captureFullPage();
} finally {
  socket.close();
  chrome.kill("SIGTERM");
  await Promise.race([
    new Promise((resolvePromise) => chrome.once("exit", resolvePromise)),
    sleep(1500),
  ]);
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      await rm(profileDir, { recursive: true, force: true });
      break;
    } catch {
      await sleep(250);
    }
  }
}
