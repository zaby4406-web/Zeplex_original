/**
 * Zeplex Cloudflare Workers AI backend
 *
 * Routes:
 *   GET  /api/health
 *   GET  /api/v1/models
 *   POST /api/v1/chat/completions
 *
 * Cloudflare Worker bindings/settings:
 *   AI                 Workers AI binding (image generation + PDF conversion)
 *   CF_ACCOUNT_ID      Text variable
 *   CF_API_TOKEN       Secret with Account > Workers AI > Read permission
 *   ALLOWED_ORIGINS    Comma-separated exact website origins
 *
 * Do not put CF_API_TOKEN in index.html.
 */
const API_PREFIX = "/api/v1";
const API_BASE = "https://api.cloudflare.com/client/v4/accounts/";
// Account ID is an identifier, not a secret. Override with CF_ACCOUNT_ID if needed.
const DEFAULT_ACCOUNT_ID = "5d04576517bb25ffcf314059bf17a239";
const MAX_BODY_BYTES = 16 * 1024 * 1024;
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_MESSAGES = 80;
const MAX_TEXT_CHARS = 250_000;
const MAX_OUTPUT_TOKENS = 12_000;

const TEXT_MODEL = "@cf/zai-org/glm-4.7-flash";
const VISION_MODEL = "@cf/google/gemma-4-26b-a4b-it";
const REASONING_MODEL = "@cf/nvidia/nemotron-3-120b-a12b";
const IMAGE_MODEL = "@cf/black-forest-labs/flux-2-klein-4b";

// All four are currently eligible for the shared 10,000-Neuron daily
// allocation. That allocation is shared across the account, not per model.
const MODEL_CATALOG = [
  {
    id: TEXT_MODEL,
    name: "GLM 4.7 Flash",
    context_length: 131072,
    architecture: { input_modalities: ["text"], output_modalities: ["text"], modality: "text->text" },
    supported_parameters: ["temperature", "max_tokens", "reasoning", "include_reasoning"],
    top_provider: { max_completion_tokens: MAX_OUTPUT_TOKENS },
    eligible_for_daily_free_allocation: true,
    neurons_per_million_input_tokens: 5500,
    neurons_per_million_output_tokens: 36400
  },
  {
    id: VISION_MODEL,
    name: "Gemma 4 26B A4B",
    context_length: 256000,
    architecture: { input_modalities: ["text", "image"], output_modalities: ["text"], modality: "text+image->text" },
    supported_parameters: ["temperature", "max_tokens", "reasoning", "include_reasoning"],
    top_provider: { max_completion_tokens: MAX_OUTPUT_TOKENS },
    eligible_for_daily_free_allocation: true,
    neurons_per_million_input_tokens: 9091,
    neurons_per_million_output_tokens: 27273
  },
  {
    id: REASONING_MODEL,
    name: "NVIDIA Nemotron 3 Super",
    context_length: 256000,
    architecture: { input_modalities: ["text"], output_modalities: ["text"], modality: "text->text" },
    supported_parameters: ["temperature", "max_tokens", "reasoning", "include_reasoning"],
    top_provider: { max_completion_tokens: MAX_OUTPUT_TOKENS },
    eligible_for_daily_free_allocation: true,
    neurons_per_million_input_tokens: 45455,
    neurons_per_million_output_tokens: 136364
  },
  {
    id: IMAGE_MODEL,
    name: "FLUX.2 Klein 4B · Image generation",
    context_length: 0,
    architecture: { input_modalities: ["text"], output_modalities: ["image"], modality: "text->image" },
    supported_parameters: ["width", "height", "guidance", "seed"],
    top_provider: { max_completion_tokens: 0 },
    eligible_for_daily_free_allocation: true,
    neurons_per_output_512_tile: 26.05
  }
].map((model) => ({
  ...model,
  object: "model",
  created: 0,
  owned_by: "cloudflare-workers-ai"
}));

const MODEL_BY_ID = new Map(MODEL_CATALOG.map((model) => [model.id, model]));
const localBuckets = new Map();

function json(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", ...headers }
  });
}

function fail(message, status = 400) {
  return json({ error: { message, type: "zeplex_worker_error", code: status } }, status);
}

function allowedOrigins(env) {
  const value = String(env.ALLOWED_ORIGINS || "").trim();
  return (value ? value.split(",") : ["https://zeplex-ai.netlify.app"])
    .map((origin) => origin.trim().replace(/\/+$/, ""))
    .filter(Boolean);
}

function originAllowed(request, env) {
  const origin = request.headers.get("Origin");
  if (!origin) return new URL(request.url).pathname === "/api/health";
  return allowedOrigins(env).includes(origin.replace(/\/+$/, ""));
}

function cors(request, env) {
  const origin = request.headers.get("Origin");
  const headers = {
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "Content-Type, Authorization",
    "access-control-max-age": "86400",
    "vary": "Origin"
  };
  if (origin && allowedOrigins(env).includes(origin.replace(/\/+$/, ""))) {
    headers["access-control-allow-origin"] = origin;
  }
  return headers;
}

function withCors(response, request, env) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(cors(request, env))) headers.set(key, value);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

async function rateLimit(request, env) {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  if (env.ZEPLEX_LIMITER && typeof env.ZEPLEX_LIMITER.limit === "function") {
    const result = await env.ZEPLEX_LIMITER.limit({ key: ip });
    return Boolean(result?.success);
  }
  // Fallback is isolate-local only. Configure a Cloudflare rate-limiting rule
  // for /api/* before making this endpoint broadly available.
  const now = Date.now();
  let bucket = localBuckets.get(ip);
  if (!bucket || now - bucket.start > 60_000) bucket = { start: now, count: 0 };
  bucket.count++;
  localBuckets.set(ip, bucket);
  if (localBuckets.size > 1500) {
    for (const [key, value] of localBuckets) if (now - value.start > 60_000) localBuckets.delete(key);
  }
  return bucket.count <= 20;
}

async function readJson(request) {
  const raw = await request.text();
  if (new TextEncoder().encode(raw).byteLength > MAX_BODY_BYTES) {
    throw Object.assign(new Error("The request is too large."), { status: 413 });
  }
  try { return JSON.parse(raw); }
  catch { throw new Error("The request body must be valid JSON."); }
}

function parseDataUrl(value) {
  const match = /^data:([\w.+-]+\/[\w.+-]+);base64,([A-Za-z0-9+/=\s]+)$/i.exec(value || "");
  if (!match) throw new Error("The attached document is not a valid base64 data URL.");
  const binary = atob(match[2].replace(/\s/g, ""));
  if (binary.length > MAX_FILE_BYTES) throw new Error("An attached file exceeds the 10 MB limit.");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return { mimeType: match[1].toLowerCase(), bytes };
}

function validateMessages(input) {
  if (!Array.isArray(input) || input.length < 1 || input.length > MAX_MESSAGES) {
    throw new Error("messages must contain between 1 and " + MAX_MESSAGES + " items.");
  }
  let chars = 0;
  const messages = input.map((message) => {
    if (!message || !["system", "developer", "user", "assistant", "tool"].includes(message.role)) {
      throw new Error("A message has an invalid role.");
    }
    if (typeof message.content === "string") {
      chars += message.content.length;
      return { role: message.role, content: message.content };
    }
    if (!Array.isArray(message.content)) throw new Error("A message must contain text or supported attachment parts.");
    const content = message.content.map((part) => {
      if (!part || typeof part !== "object") throw new Error("An attachment part is invalid.");
      if (part.type === "text" && typeof part.text === "string") {
        chars += part.text.length;
        return { type: "text", text: part.text };
      }
      if (part.type === "image_url" && typeof part.image_url?.url === "string") {
        const url = part.image_url.url;
        if (!/^https:\/\//i.test(url) && !/^data:image\/(png|jpeg|webp);base64,/i.test(url)) {
          throw new Error("Images must be HTTPS URLs or PNG, JPEG or WebP data URLs.");
        }
        return { type: "image_url", image_url: { url } };
      }
      if (part.type === "file" && typeof part.file?.file_data === "string") {
        return { type: "file", file: {
          filename: String(part.file.filename || "document.pdf").slice(0, 160),
          file_data: part.file.file_data
        }};
      }
      throw new Error("This attachment format is not supported.");
    });
    return { role: message.role, content };
  });
  if (chars > MAX_TEXT_CHARS) throw new Error("The conversation text is too large. Split the attachment or shorten the chat.");
  return messages;
}

async function convertPdfs(messages, env) {
  const docs = [];
  const slots = [];
  for (const message of messages) {
    if (!Array.isArray(message.content)) continue;
    const rebuilt = [];
    for (const part of message.content) {
      if (part.type !== "file") {
        rebuilt.push(part);
        continue;
      }
      if (typeof env.AI?.toMarkdown !== "function") {
        throw new Error("PDF conversion is unavailable. Enable the Workers AI binding named AI.");
      }
      const { mimeType, bytes } = parseDataUrl(part.file.file_data);
      const marker = { index: docs.length };
      docs.push({ name: part.file.filename, blob: new Blob([bytes], { type: mimeType }) });
      rebuilt.push(marker);
      slots.push({ rebuilt, marker, filename: part.file.filename });
    }
    message.content = rebuilt;
  }
  if (!docs.length) return messages;
  const result = await env.AI.toMarkdown(docs);
  const converted = Array.isArray(result) ? result : [result];
  for (const slot of slots) {
    const item = converted[slot.marker.index];
    if (!item || item.format === "error" || typeof item.data !== "string") {
      throw new Error("Could not extract " + slot.filename + ": " + (item?.error || "conversion failed"));
    }
    const index = slot.rebuilt.indexOf(slot.marker);
    slot.rebuilt.splice(index, 1, {
      type: "text",
      text: "\n\n[Extracted contents of " + slot.filename + "]\n" + item.data.slice(0, 120_000) + "\n[End of extracted document]"
    });
  }
  return messages;
}

function hasImage(messages) {
  return messages.some((m) => Array.isArray(m.content) && m.content.some((p) => p.type === "image_url"));
}

function latestUserText(messages) {
  const message = [...messages].reverse().find((m) => m.role === "user");
  if (!message) return "";
  if (typeof message.content === "string") return message.content.trim();
  return message.content.filter((p) => p.type === "text").map((p) => p.text).join("\n").trim();
}

function clamp(value, min, max, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
}

function bytesToBase64(value) {
  const bytes = value instanceof Uint8Array ? value :
    value instanceof ArrayBuffer ? new Uint8Array(value) :
    ArrayBuffer.isView(value) ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength) : null;
  if (!bytes) throw new Error("The image model returned an unsupported image format.");
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function upstreamError(text, status) {
  let message = "The AI request failed.";
  try {
    const data = JSON.parse(text);
    message = data.error?.message || data.errors?.[0]?.message || data.message || message;
  } catch { if (text) message = text.slice(0, 400); }
  if (/neuron|daily limit|quota|usage limit/i.test(message)) {
    return "The shared 10,000-Neuron daily allocation may have been exhausted. It resets at 00:00 UTC. Check Cloudflare usage.";
  }
  if (status === 401) return "Cloudflare rejected CF_API_TOKEN. Check the token and its Workers AI permissions.";
  if (status === 403) return "Cloudflare denied this model or account. Check model availability and billing settings.";
  if (status === 429 || /capacity|overloaded|temporarily exceeded/i.test(message)) {
    return "The model is busy or rate-limited. Retry in a moment.";
  }
  return message;
}

async function imageCompletion(body, env) {
  if (typeof env.AI?.run !== "function") return fail("Image generation is unavailable. Enable the Workers AI binding named AI.", 503);
  const messages = validateMessages(body.messages);
  const prompt = latestUserText(messages);
  if (!prompt) return fail("Add a description of the image you want to create.");
  const input = {
    prompt: prompt.slice(0, 8000),
    width: Math.round(clamp(body.width, 256, 1920, 1024)),
    height: Math.round(clamp(body.height, 256, 1920, 768)),
    guidance: clamp(body.guidance, 0, 20, 4)
  };
  if (Number.isInteger(body.seed)) input.seed = body.seed;
  const result = await env.AI.run(IMAGE_MODEL, input);
  if (!result?.image) throw new Error("The image model completed without returning an image.");
  const image = typeof result.image === "string"
    ? (result.image.startsWith("data:image/") ? result.image : "data:image/png;base64," + result.image)
    : "data:image/png;base64," + bytesToBase64(result.image);
  return json({
    id: "chatcmpl-" + crypto.randomUUID(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: IMAGE_MODEL,
    choices: [{ index: 0, message: {
      role: "assistant",
      content: "Here is the image I generated for you.",
      images: [{ image_url: { url: image } }]
    }, finish_reason: "stop" }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  });
}

async function chatCompletion(request, env) {
  let body;
  try { body = await readJson(request); }
  catch (error) { return fail(error.message, error.status || 400); }

  if (Array.isArray(body.modalities) && body.modalities.includes("image")) {
    try { return await imageCompletion(body, env); }
    catch (error) { return fail(error.message || "Image generation failed.", 502); }
  }
  const accountId = env.CF_ACCOUNT_ID || DEFAULT_ACCOUNT_ID;
  if (!accountId || !env.CF_API_TOKEN) {
    return fail("Cloudflare chat is not configured. Add the CF_API_TOKEN secret to this Worker.", 503);
  }

  let messages;
  try {
    messages = validateMessages(body.messages);
    messages = await convertPdfs(messages, env);
  } catch (error) { return fail(error.message || "The messages or attachments are invalid.", 400); }

  const imageInput = hasImage(messages);
  const requested = typeof body.model === "string" ? body.model : "";
  const selected = MODEL_BY_ID.has(requested) ? requested : (imageInput ? VISION_MODEL : TEXT_MODEL);
  const metadata = MODEL_BY_ID.get(selected);
  if (selected === IMAGE_MODEL) return fail("Use modalities:['image','text'] for image generation.", 400);
  if (imageInput && !metadata.architecture.input_modalities.includes("image")) {
    return fail("The selected model cannot analyse image attachments. Refresh the model catalogue and retry.", 400);
  }

  const payload = {
    model: selected,
    messages,
    stream: body.stream === true,
    max_tokens: Math.round(clamp(body.max_tokens ?? body.max_completion_tokens, 256, MAX_OUTPUT_TOKENS, 4096)),
    temperature: clamp(body.temperature, 0, 2, 0.7)
  };
  const wantsThinking = Boolean(body.include_reasoning || body.reasoning?.effort === "high" || body.reasoning_effort === "high");
  if (wantsThinking && [VISION_MODEL, REASONING_MODEL].includes(selected)) {
    payload.chat_template_kwargs = { enable_thinking: true };
  }

  const endpoint = API_BASE + encodeURIComponent(accountId) + "/ai/v1/chat/completions";
  let upstream;
  try {
    upstream = await fetch(endpoint, {
      method: "POST",
      headers: {
        authorization: "Bearer " + env.CF_API_TOKEN,
        "content-type": "application/json",
        accept: payload.stream ? "text/event-stream" : "application/json"
      },
      body: JSON.stringify(payload)
    });
  } catch {
    return fail("Could not reach Cloudflare Workers AI. Check the Worker network and account configuration.", 502);
  }

  if (!upstream.ok) return fail(upstreamError(await upstream.text(), upstream.status), upstream.status);
  const headers = new Headers({
    "cache-control": "no-cache, no-transform",
    "x-accel-buffering": "no"
  });
  const contentType = upstream.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  return new Response(upstream.body, { status: upstream.status, headers });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") {
      if (!originAllowed(request, env)) return new Response(null, { status: 403 });
      return new Response(null, { status: 204, headers: cors(request, env) });
    }
    if (!originAllowed(request, env)) {
      return withCors(fail("This origin is not allowed to use the Zeplex API.", 403), request, env);
    }
    if (url.pathname === "/api/health" && request.method === "GET") {
      return withCors(json({
        ok: true,
        service: "zeplex-cloudflare-ai",
        chatConfigured: Boolean((env.CF_ACCOUNT_ID || DEFAULT_ACCOUNT_ID) && env.CF_API_TOKEN),
        imageBindingConfigured: typeof env.AI?.run === "function",
        documentBindingConfigured: typeof env.AI?.toMarkdown === "function"
      }), request, env);
    }
    if (url.pathname === API_PREFIX + "/models" && request.method === "GET") {
      return withCors(json({ object: "list", data: MODEL_CATALOG }), request, env);
    }
    if (url.pathname === API_PREFIX + "/chat/completions" && request.method === "POST") {
      if (!(await rateLimit(request, env))) {
        return withCors(fail("Too many requests. Please wait a minute and try again.", 429), request, env);
      }
      return withCors(await chatCompletion(request, env), request, env);
    }
    return withCors(fail("Route not found.", 404), request, env);
  }
};
