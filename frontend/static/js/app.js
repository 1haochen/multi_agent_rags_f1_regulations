/**
 * F1 Regulations Assistant — chat UI with SSE streaming.
 */

const chatEl = document.getElementById("chat");
const welcomeEl = document.getElementById("welcome");
const formEl = document.getElementById("composer-form");
const inputEl = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

const tplUser = document.getElementById("tpl-user-bubble");
const tplAssistant = document.getElementById("tpl-assistant-bubble");

const SESSION_KEY = "f1_regs_session_id";

function getSessionId() {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    id = `sess_${crypto.randomUUID?.() || String(Date.now())}`;
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function hideWelcome() {
  if (welcomeEl && welcomeEl.parentNode) {
    welcomeEl.remove();
  }
}

function appendUserMessage(text) {
  hideWelcome();
  const node = tplUser.content.cloneNode(true);
  const bubble = node.querySelector(".msg__bubble--user");
  bubble.textContent = text;
  chatEl.appendChild(node);
  scrollToBottom();
  return bubble;
}

function appendAssistantShell() {
  hideWelcome();
  const node = tplAssistant.content.cloneNode(true);
  chatEl.appendChild(node);
  const root = chatEl.lastElementChild;
  const thinking = root.querySelector(".msg__thinking");
  const thinkingLabel = root.querySelector(".msg__thinking-label");
  const body = root.querySelector(".msg__body");
  const meta = root.querySelector(".msg__meta");
  thinking.hidden = false;
  body.classList.add("is-streaming");
  scrollToBottom();
  return { root, thinking, thinkingLabel, body, meta };
}

function setComposerEnabled(on) {
  sendBtn.disabled = !on;
  inputEl.disabled = !on;
}

function autoResizeTextarea() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
}

inputEl.addEventListener("input", autoResizeTextarea);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

/**
 * Parse SSE stream from fetch body.
 * @param {ReadableStream<Uint8Array>} stream
 * @param {(obj: object) => void} onEvent
 */
async function readSSE(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const line = part
        .split("\n")
        .find((l) => l.startsWith("data: "));
      if (!line) continue;
      const raw = line.slice(6).trim();
      if (raw === "[DONE]") return;
      try {
        onEvent(JSON.parse(raw));
      } catch {
        /* ignore */
      }
    }
  }
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  appendUserMessage(text);
  inputEl.value = "";
  autoResizeTextarea();

  const { thinking, thinkingLabel, body, meta } = appendAssistantShell();
  setComposerEnabled(false);
  scrollToBottom();

  let fullAnswer = "";
  let metaPayload = null;

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        message: text,
        session_id: getSessionId(),
      }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || res.statusText);
    }

    if (!res.body) {
      throw new Error("No response body");
    }

    await readSSE(res.body, (ev) => {
      if (ev.type === "phase" && ev.label) {
        thinkingLabel.textContent = ev.label;
      }
      if (ev.type === "error") {
        thinking.hidden = true;
        body.classList.remove("is-streaming");
        body.textContent = ev.message || "Something went wrong.";
      }
      if (ev.type === "token" && ev.text != null) {
        if (thinking && !thinking.hidden) {
          thinking.hidden = true;
        }
        fullAnswer += ev.text;
        body.textContent = fullAnswer;
        scrollToBottom();
      }
      if (ev.type === "meta") {
        metaPayload = ev;
      }
    });

    body.classList.remove("is-streaming");

    if (metaPayload) {
      meta.hidden = false;
      const rows = [];
      if (metaPayload.resolved_query) {
        rows.push(
          `<div class="msg__meta-row"><strong>Resolved query</strong> ${escapeHtml(metaPayload.resolved_query)}</div>`
        );
      }
      if (metaPayload.query_type) {
        rows.push(
          `<div class="msg__meta-row"><strong>Type</strong> ${escapeHtml(metaPayload.query_type)}</div>`
        );
      }
      if (metaPayload.answer_supported != null) {
        rows.push(
          `<div class="msg__meta-row"><strong>Evidence</strong> ${metaPayload.answer_supported ? "supported" : "limited / check sources"}</div>`
        );
      }
      if (metaPayload.support_notes) {
        rows.push(
          `<div class="msg__meta-row"><strong>Notes</strong> ${escapeHtml(metaPayload.support_notes)}</div>`
        );
      }
      if (Array.isArray(metaPayload.citations) && metaPayload.citations.length) {
        const cites = metaPayload.citations
          .slice(0, 8)
          .map((c, i) => {
            const clause =
              c.clause || c.article || c.appendix || "—";
            const src = c.source_file ? truncate(c.source_file, 48) : "—";
            return `${i + 1}. ${escapeHtml(String(clause))} · ${escapeHtml(src)}`;
          })
          .join("<br/>");
        rows.push(`<div class="msg__meta-row"><strong>Citations</strong><br/>${cites}</div>`);
      }
      meta.innerHTML = rows.join("");
    }
  } catch (err) {
    thinking.hidden = true;
    body.classList.remove("is-streaming");
    body.textContent =
      err instanceof Error
        ? `Error: ${err.message}`
        : "An unexpected error occurred.";
  } finally {
    setComposerEnabled(true);
    inputEl.focus();
    scrollToBottom();
  }
});

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function truncate(s, n) {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}
