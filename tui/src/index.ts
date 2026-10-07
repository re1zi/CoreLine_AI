import {
  BoxRenderable,
  createCliRenderer,
  FrameBufferRenderable,
  InputRenderable,
  InputRenderableEvents,
  RGBA,
  TextRenderable,
} from "@opentui/core";

type BridgeEvent =
  | { type: "ready" }
  | { type: "shutdown" }
  | { type: "status"; web_enabled: boolean; run_enabled: boolean }
  | {
      type: "avatar_state";
      state: string;
      art?: string | null;
      pixels?: number[][][] | null;
    }
  | { type: "message"; role: string; text: string; id?: string }
  | { type: "stream_start" }
  | { type: "stream_chunk"; chunk: string }
  | { type: "stream_end" }
  | { type: "truncate"; id: string }
  | { type: "turn_done" }
  | { type: "confirm_request"; command: string }
  | { type: "error"; message: string };

type UiMessage = { role: string; text: string; id?: string };

const renderer = await createCliRenderer({
  exitOnCtrlC: true,
  targetFps: 20,
});

const PALETTE = {
  bgRoot: "#1a2a3a",
  panelBg: "#1a2a3a",
  contentBg: "#142332",
  border: "#5a9fd4",
  borderSoft: "#3f6f95",
  title: "#5a9fd4",
  prompt: "#7ab8d9",
  text: "#b8d4e8",
  textUser: "#c8dde8",
  textAssistant: "#a8c8d8",
  textSystem: "#98a8b8",
  placeholder: "#6f8ea5",
};

// Avatar layout controls (cells). Actual render uses half-block (2 vertical pixels per cell).
const AVATAR_PANEL_RATIO = 0.4; // left panel width fraction
const AVATAR_MIN_W = 18;
const AVATAR_MIN_H = 10;
const AVATAR_MAX_W = 130;
const AVATAR_MAX_H = 85;

const root = new BoxRenderable(renderer, {
  id: "root",
  width: "100%",
  height: "100%",
  border: false,
  flexDirection: "column",
  gap: 0,
  backgroundColor: PALETTE.bgRoot,
});

const statusBox = new BoxRenderable(renderer, {
  id: "status-box",
  width: "100%",
  height: 3,
  border: true,
  borderStyle: "single",
  padding: 1,
  borderColor: PALETTE.borderSoft,
  backgroundColor: PALETTE.panelBg,
});
const statusText = new TextRenderable(renderer, {
  id: "status-text",
  content: "CoreLine OpenTUI | connecting backend...",
  fg: PALETTE.prompt,
});
statusBox.add(statusText);

const body = new BoxRenderable(renderer, {
  id: "body",
  width: "100%",
  height: "auto",
  border: false,
  flexDirection: "row",
  gap: 1,
  flexGrow: 1,
});

const rightColumn = new BoxRenderable(renderer, {
  id: "right-column",
  width: "60%",
  height: "100%",
  border: false,
  flexDirection: "column",
  gap: 1,
});

const avatarPanel = new BoxRenderable(renderer, {
  id: "avatar-panel",
  width: "40%",
  height: "auto",
  border: true,
  borderStyle: "single",
  padding: 1,
  flexDirection: "column",
  borderColor: PALETTE.borderSoft,
  backgroundColor: PALETTE.contentBg,
});
const avatarTitle = new TextRenderable(renderer, {
  id: "avatar-title",
  content: "Avatar",
  fg: PALETTE.title,
});
const avatarStateText = new TextRenderable(renderer, {
  id: "avatar-state",
  content: "state: idle",
  fg: PALETTE.textSystem,
});
const avatarFaceText = new TextRenderable(renderer, {
  id: "avatar-face",
  content: "(・‿・)",
  fg: PALETTE.prompt,
});
const avatarCanvas = new FrameBufferRenderable(renderer, {
  id: "avatar-canvas",
  width: AVATAR_MIN_W,
  height: AVATAR_MIN_H,
});
avatarPanel.add(avatarTitle);
avatarPanel.add(avatarStateText);
avatarPanel.add(avatarCanvas);
avatarPanel.add(avatarFaceText);

const chatPanel = new BoxRenderable(renderer, {
  id: "chat-panel",
  width: "100%",
  height: "auto",
  border: true,
  borderStyle: "single",
  padding: 1,
  borderColor: PALETTE.borderSoft,
  backgroundColor: PALETTE.contentBg,
});
const chatText = new TextRenderable(renderer, {
  id: "chat-text",
  content: "Starting CoreLine backend...\n",
  fg: PALETTE.text,
});
chatPanel.add(chatText);

rightColumn.add(chatPanel);

const inputBox = new BoxRenderable(renderer, {
  id: "input-box",
  width: "100%",
  height: 5,
  border: true,
  borderStyle: "single",
  padding: 1,
  flexDirection: "column",
  borderColor: PALETTE.borderSoft,
  backgroundColor: PALETTE.panelBg,
});
const inputHint = new TextRenderable(renderer, {
  id: "input-hint",
  content: "Enter | *d [N] удалить | *e [N] [текст] правка | *r [N] заново",
  fg: PALETTE.prompt,
});
const input = new InputRenderable(renderer, {
  id: "main-input",
  width: "auto",
  placeholder: "Type and press Enter...",
  focusedBackgroundColor: PALETTE.contentBg,
});
inputBox.add(inputHint);
inputBox.add(input);

rightColumn.add(inputBox);

body.add(avatarPanel);
body.add(rightColumn);

root.add(statusBox);
root.add(body);
renderer.root.add(root);

const messages: UiMessage[] = [];
let streamBuffer = "";
let pendingConfirm: string | null = null;
let rewriteTarget: { id: string; n: number } | null = null;
let turnBusy = false;
let webEnabled = false;
let runEnabled = false;
let avatarState = "idle";
let avatarArt = "(・‿・)";
let avatarImageLoaded = false;
let avatarImageError = "";
const avatarFaces: Record<string, string> = {
  idle: "(・‿・)",
  thinking: "(¬‿¬)",
  speaking: "(＾▽＾)",
  sleeping: "(-_-) zZ",
  joy: "ヽ(°〇°)ﾉ",
  satisfaction: "(◡ ‿ ◡)",
  indifference: "( ͡° ͜ʖ ͡°)",
  anger: "(╬ಠ益ಠ)",
  sadness: "(╥﹏╥)",
  fear: "(° △ °)",
  disgust: "(´Д`)",
  surprise: "(⊙_⊙)",
  contempt: "(¬_¬)",
  blush: "(⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)",
};

function rolePrefix(role: string): string {
  if (role === "user") return "You";
  if (role === "coreline") return "CoreLine";
  if (role === "system") return "System";
  return role;
}

function refreshStatus() {
  statusText.content =
    `CoreLine OpenTUI | web:${webEnabled ? "ON" : "OFF"} ` +
    `run:${runEnabled ? "ON" : "OFF"} | ` +
    `img:${avatarImageLoaded ? "ok" : "fallback"} | ` +
    `${pendingConfirm ? "awaiting run confirm" : turnBusy ? "busy" : rewriteTarget ? `editing #${rewriteTarget.n}` : "ready"}`;
  avatarStateText.content = `state: ${avatarState}`;
  avatarFaceText.content = avatarImageLoaded
    ? ""
    : `${avatarArt || avatarFaces[avatarState] || avatarFaces.idle}${
        avatarImageError ? `\n${avatarImageError}` : ""
      }`;
}

function drawAvatarFallback() {
  avatarImageLoaded = false;
  avatarCanvas.frameBuffer.clear(RGBA.fromHex(PALETTE.contentBg));
}

function toHex(r: number, g: number, b: number): string {
  const h = (v: number) => Math.max(0, Math.min(255, v)).toString(16).padStart(2, "0");
  return `#${h(r)}${h(g)}${h(b)}`;
}

async function drawAvatarImage(state: string) {
  // Image rendering now comes from backend pixels. Keep this for startup fallback.
  avatarImageError = `awaiting backend pixels for state: ${state}`;
  drawAvatarFallback();
  refreshStatus();
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function computeAvatarCellSize() {
  // Approximation based on terminal size (good enough + updates on resize)
  const panelW = Math.floor(renderer.width * AVATAR_PANEL_RATIO) - 6; // borders/padding
  const usableH = renderer.height - 3 - 5 - 6; // header + input + borders/gaps
  const w = clamp(panelW, AVATAR_MIN_W, AVATAR_MAX_W);
  const h = clamp(usableH, AVATAR_MIN_H, AVATAR_MAX_H);
  return { w, h };
}

function applyAvatarCellSize(w: number, h: number) {
  // Resize framebuffer to actually occupy available space.
  // (Changing only renderable props via `as any` may not trigger an internal buffer resize.)
  try {
    avatarCanvas.frameBuffer.resize(w, h);
  } catch {
    // fallback: best-effort update
    (avatarCanvas as any).width = w;
    (avatarCanvas as any).height = h;
  }
  drawAvatarFallback();
  sendBackend({ type: "avatar_config", cell_width: w, cell_height: h });
}

function rgbaFromInts(r: number, g: number, b: number) {
  return RGBA.fromInts(
    Math.max(0, Math.min(255, r)),
    Math.max(0, Math.min(255, g)),
    Math.max(0, Math.min(255, b)),
    255,
  );
}

function drawAvatarPixels(pixels: number[][][] | null | undefined) {
  if (!pixels || !pixels.length) {
    avatarImageError = "no pixel payload";
    drawAvatarFallback();
    refreshStatus();
    return;
  }
  try {
    const fb = avatarCanvas.frameBuffer;
    fb.clear(RGBA.fromHex(PALETTE.contentBg));

    // Half-block: each terminal cell encodes 2 vertical pixels via fg/bg on '▀'
    const cellH = fb.height;
    const cellW = fb.width;
    for (let y = 0; y < cellH; y++) {
      const topRow = pixels[y * 2] ?? [];
      const botRow = pixels[y * 2 + 1] ?? [];
      for (let x = 0; x < cellW; x++) {
        const top = topRow[x];
        const bot = botRow[x];
        const tr = top?.[0] ?? 0;
        const tg = top?.[1] ?? 0;
        const tb = top?.[2] ?? 0;
        const ta = top?.[3] ?? 0;
        const br = bot?.[0] ?? 0;
        const bg = bot?.[1] ?? 0;
        const bb = bot?.[2] ?? 0;
        const ba = bot?.[3] ?? 0;

        if (ta < 20 && ba < 20) {
          fb.setCell(x, y, " ", RGBA.fromHex(PALETTE.contentBg), RGBA.fromHex(PALETTE.contentBg));
          continue;
        }

        fb.setCell(x, y, "▀", rgbaFromInts(tr, tg, tb), rgbaFromInts(br, bg, bb));
      }
    }
    avatarImageLoaded = true;
    avatarImageError = "";
  } catch (err) {
    avatarImageError = `pixel draw error: ${err instanceof Error ? err.message : String(err)}`;
    drawAvatarFallback();
  }
  refreshStatus();
}

function isEditable(m: UiMessage): boolean {
  return !!m.id && (m.role === "user" || m.role === "coreline" || m.role === "assistant");
}

function editableMessages(): UiMessage[] {
  return messages.filter(isEditable);
}

function defaultHint(): string {
  if (rewriteTarget) {
    return `Правка #${rewriteTarget.n} — Enter отправить, пустой ввод отмена`;
  }
  return "Enter | *d [N] удалить | *e [N] [текст] правка | *r [N] заново";
}

function setHint(text?: string) {
  inputHint.content = text || defaultHint();
}

function refreshChat() {
  const editable = editableMessages();
  const visible = messages.slice(-24);
  const lines = visible.map((m) => {
    const idx = isEditable(m) ? editable.indexOf(m) + 1 : 0;
    const num = idx > 0 ? `[${idx}] ` : "";
    return `${num}${rolePrefix(m.role)}: ${m.text}`;
  });
  if (streamBuffer) {
    lines.push(`CoreLine: ${streamBuffer}`);
  }
  if (pendingConfirm) {
    lines.push(`System: Run command? [y/n] -> ${pendingConfirm}`);
  }
  chatText.content = lines.join("\n");
}

function pushMessage(role: string, text: string, id?: string) {
  const msg: UiMessage = { role, text };
  if (id) msg.id = id;
  messages.push(msg);
  if (messages.length > 200) {
    messages.splice(0, messages.length - 200);
  }
  refreshChat();
}

function truncateFromId(id: string) {
  const start = messages.findIndex((m) => m.id === id);
  if (start < 0) return;
  messages.splice(start);
  streamBuffer = "";
  refreshChat();
}

function resolveEditable(n?: number, role?: "user" | "assistant"): UiMessage | null {
  const list = editableMessages();
  if (!list.length) return null;
  if (n != null) {
    const item = list[n - 1];
    if (!item) return null;
    if (role === "user" && item.role !== "user") return null;
    if (role === "assistant" && item.role !== "coreline" && item.role !== "assistant") return null;
    return item;
  }
  if (role === "user") {
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].role === "user") return list[i];
    }
    return null;
  }
  if (role === "assistant") {
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].role === "coreline" || list[i].role === "assistant") return list[i];
    }
    return null;
  }
  return list[list.length - 1];
}

const pyProc = Bun.spawn({
  cmd: ["python", "coreline_backend_bridge.py"],
  cwd: "..",
  stdout: "pipe",
  stderr: "pipe",
  stdin: "pipe",
});

async function readBackendStderr() {
  if (!pyProc.stderr) return;
  const stderrText = await new Response(pyProc.stderr).text();
  if (stderrText.trim()) {
    pushMessage("system", `backend stderr:\n${stderrText.trim()}`);
  }
}

function sendBackend(event: Record<string, unknown>) {
  if (!pyProc.stdin) return;
  const line = `${JSON.stringify(event)}\n`;
  pyProc.stdin.write(line);
}

function handleBridgeEvent(event: BridgeEvent) {
  if (event.type === "status") {
    webEnabled = !!event.web_enabled;
    runEnabled = !!event.run_enabled;
    refreshStatus();
    return;
  }
  if (event.type === "avatar_state") {
    avatarState = event.state || "idle";
    avatarArt = avatarFaces[avatarState] || avatarFaces.idle;
    drawAvatarPixels(event.pixels);
    refreshStatus();
    return;
  }
  if (event.type === "message") {
    pushMessage(event.role, event.text, event.id);
    return;
  }
  if (event.type === "truncate") {
    if (event.id) truncateFromId(event.id);
    return;
  }
  if (event.type === "turn_done") {
    turnBusy = false;
    rewriteTarget = null;
    setHint();
    refreshStatus();
    return;
  }
  if (event.type === "stream_start") {
    streamBuffer = "";
    refreshChat();
    return;
  }
  if (event.type === "stream_chunk") {
    streamBuffer += (event.chunk || "").replace(/\[TIME\]/gi, "");
    refreshChat();
    return;
  }
  if (event.type === "stream_end") {
    // Final assistant message comes as `type: "message"` from backend.
    // Keep stream_end for cleanup only to avoid duplicated chat lines.
    streamBuffer = "";
    refreshChat();
    return;
  }
  if (event.type === "confirm_request") {
    pendingConfirm = event.command;
    refreshStatus();
    refreshChat();
    return;
  }
  if (event.type === "ready") {
    pushMessage("system", "Backend ready.");
    return;
  }
  if (event.type === "error") {
    pushMessage("system", `backend error: ${event.message}`);
    return;
  }
  if (event.type === "shutdown") {
    pushMessage("system", "Backend stopped.");
  }
}

async function readBridgeEvents() {
  if (!pyProc.stdout) return;
  const reader = pyProc.stdout.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx = buffer.indexOf("\n");
    while (idx >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (line) {
        try {
          handleBridgeEvent(JSON.parse(line) as BridgeEvent);
        } catch {
          pushMessage("system", `invalid backend event: ${line}`);
        }
      }
      idx = buffer.indexOf("\n");
    }
  }
}

input.on(InputRenderableEvents.ENTER, (value: string) => {
  const v = (value || "").trim();
  input.value = "";
  if (!v) {
    if (rewriteTarget) {
      rewriteTarget = null;
      setHint();
      refreshStatus();
    }
    return;
  }
  if (pendingConfirm) {
    sendBackend({ type: "confirm_response", answer: v });
    pendingConfirm = null;
    refreshStatus();
    refreshChat();
    return;
  }
  if (turnBusy) {
    pushMessage("system", "Подождите, пока закончится текущий ответ.");
    return;
  }

  if (rewriteTarget && !/^\*(?:d|del|удалить|r|regen|заново|e|edit|правка)\b/i.test(v)) {
    turnBusy = true;
    refreshStatus();
    sendBackend({ type: "rewrite", id: rewriteTarget.id, text: v });
    rewriteTarget = null;
    setHint();
    return;
  }

  const delMatch = v.match(/^\*(?:d|del|удалить)(?:\s+(\d+))?$/i);
  if (delMatch) {
    const n = delMatch[1] ? Number(delMatch[1]) : undefined;
    const target = resolveEditable(n);
    if (!target?.id) {
      pushMessage("system", "Нечего удалять. Укажите номер из чата, например *d 2");
      return;
    }
    turnBusy = true;
    refreshStatus();
    sendBackend({ type: "delete", id: target.id });
    return;
  }

  const regenMatch = v.match(/^\*(?:r|regen|заново)(?:\s+(\d+))?$/i);
  if (regenMatch) {
    const n = regenMatch[1] ? Number(regenMatch[1]) : undefined;
    const target = resolveEditable(n, "assistant");
    if (!target?.id) {
      pushMessage("system", "Нечего перегенерировать. Укажите номер ответа, например *r 2");
      return;
    }
    turnBusy = true;
    refreshStatus();
    sendBackend({ type: "regenerate", id: target.id });
    return;
  }

  const editMatch = v.match(/^\*(?:e|edit|правка)(?:\s+(\d+))?(?:\s+([\s\S]+))?$/i);
  if (editMatch) {
    const n = editMatch[1] ? Number(editMatch[1]) : undefined;
    const rest = (editMatch[2] || "").trim();
    const target = resolveEditable(n, "user");
    if (!target?.id) {
      pushMessage("system", "Нечего переписывать. Укажите номер своего сообщения, например *e 1");
      return;
    }
    const num = editableMessages().indexOf(target) + 1;
    if (rest) {
      turnBusy = true;
      refreshStatus();
      sendBackend({ type: "rewrite", id: target.id, text: rest });
      return;
    }
    rewriteTarget = { id: target.id, n: num };
    input.value = target.text;
    setHint();
    refreshStatus();
    return;
  }

  turnBusy = true;
  refreshStatus();
  sendBackend({ type: "user_input", text: v });
});

renderer.keyInput.on("keypress", (key) => {
  if (key.ctrl && key.name === "c") {
    sendBackend({ type: "exit" });
  }
});

input.focus();
{
  const { w, h } = computeAvatarCellSize();
  applyAvatarCellSize(w, h);
}
setHint();
refreshStatus();
refreshChat();
readBridgeEvents();
readBackendStderr();

// Best-effort resize handling
renderer.keyInput.on("keypress", (key) => {
  if (key.name === "resize") {
    const { w, h } = computeAvatarCellSize();
    applyAvatarCellSize(w, h);
  }
});

