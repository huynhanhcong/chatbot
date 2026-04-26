const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition || null;

const PRODUCT_SELECTION_TEXT = "Tôi tìm được các sản phẩm sau hãy chọn sản phẩm của bạn.";

const state = {
  busy: false,
  conversationId: null,
  recognition: null,
  listening: false,
  voiceSessionId: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
};

const messagesEl = document.querySelector("#messages");
const micButtonEl = document.querySelector("#micButton");
const statusDotEl = document.querySelector("#statusDot");
const statusTextEl = document.querySelector("#statusText");
const textFallbackEl = document.querySelector("#textFallback");
const sendTextButtonEl = document.querySelector("#sendTextButton");
const newSessionButtonEl = document.querySelector("#newSessionButton");

appendAssistant(
  "Xin chào. Bạn có thể hỏi về gói khám, bác sĩ, thuốc hoặc dịch vụ. Nội dung giọng nói chỉ hỗ trợ thông tin tham khảo, không thay thế tư vấn y tế trực tiếp."
);
setupRecognition();
syncControls();

micButtonEl.addEventListener("click", () => {
  if (!state.recognition) {
    setStatus("Trình duyệt không hỗ trợ nhận giọng nói. Hãy dùng ô nhập văn bản.", "error");
    return;
  }
  if (state.listening) {
    state.recognition.stop();
    return;
  }
  state.recognition.start();
});

sendTextButtonEl.addEventListener("click", sendFallbackText);
textFallbackEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    sendFallbackText();
  }
});

newSessionButtonEl.addEventListener("click", () => {
  window.speechSynthesis.cancel();
  state.conversationId = null;
  state.voiceSessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  messagesEl.replaceChildren();
  appendAssistant("Đã bắt đầu phiên mới.");
  setStatus("Sẵn sàng", "idle");
});

function setupRecognition() {
  if (!SpeechRecognition) {
    setStatus("Trình duyệt không hỗ trợ nhận giọng nói", "error");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "vi-VN";
  recognition.interimResults = true;
  recognition.continuous = false;

  recognition.addEventListener("start", () => {
    state.listening = true;
    setStatus("Đang nghe", "listening");
    syncControls();
  });

  recognition.addEventListener("end", () => {
    state.listening = false;
    if (!state.busy) {
      setStatus("Sẵn sàng", "idle");
    }
    syncControls();
  });

  recognition.addEventListener("error", (event) => {
    state.listening = false;
    setStatus(event.error || "Lỗi microphone", "error");
    syncControls();
  });

  recognition.addEventListener("result", async (event) => {
    let finalTranscript = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      if (result.isFinal) {
        finalTranscript += result[0].transcript;
      }
    }
    const transcript = finalTranscript.trim();
    if (transcript) {
      await sendVoiceTurn(transcript);
    }
  });

  state.recognition = recognition;
}

async function sendFallbackText() {
  const transcript = textFallbackEl.value.trim();
  if (!transcript || state.busy) {
    return;
  }
  textFallbackEl.value = "";
  await sendVoiceTurn(transcript);
}

async function sendVoiceTurn(transcript, options = {}) {
  window.speechSynthesis.cancel();
  appendUser(transcript);
  setBusy(true);
  setStatus("Đang xử lý", "busy");

  const startedAt = performance.now();
  try {
    const payload = { message: transcript };
    if (state.conversationId) {
      payload.conversation_id = state.conversationId;
    }

    const selection = options.selectedIndex || parseSpokenSelection(transcript);
    if (selection) {
      payload.message = String(selection);
      payload.selected_index = selection;
    }
    if (options.selectedSku) {
      payload.selected_sku = options.selectedSku;
    }

    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Voice-Session-Id": state.voiceSessionId,
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    if (data.conversation_id) {
      state.conversationId = data.conversation_id;
    }

    const assistantText = renderAssistantResponse(data);
    safeSpeak(assistantText);
    setStatus(`Đã trả lời trong ${Math.round(performance.now() - startedAt)} ms`, "idle");
  } catch (error) {
    const fallback = "Xin lỗi, tôi chưa xử lý được yêu cầu bằng giọng nói. Bạn vui lòng thử lại.";
    appendAssistant(`${fallback}\n\nChi tiết kỹ thuật: ${error.message || "Không rõ lỗi"}`, true);
    safeSpeak(fallback);
    setStatus(error.message || "Lỗi xử lý", "error");
  } finally {
    setBusy(false);
  }
}

function renderAssistantResponse(data) {
  if (data.status === "need_selection") {
    appendAssistant(PRODUCT_SELECTION_TEXT);
    appendProductOptions(data.options || []);
    return PRODUCT_SELECTION_TEXT;
  }

  const assistantText = data.answer || data.message || "Không có nội dung trả lời.";
  appendAssistant(assistantText);
  return assistantText;
}

function parseSpokenSelection(text) {
  const normalized = text
    .toLowerCase()
    .replace(/\bmột\b|\bmot\b/g, "1")
    .replace(/\bhai\b/g, "2")
    .replace(/\bba\b/g, "3")
    .replace(/\bbốn\b|\bbon\b|\btư\b|\btu\b/g, "4")
    .replace(/\bnăm\b|\bnam\b/g, "5")
    .replace(/\bsáu\b|\bsau\b/g, "6")
    .replace(/\bbảy\b|\bbay\b/g, "7")
    .replace(/\btám\b|\btam\b/g, "8")
    .replace(/\bchín\b|\bchin\b/g, "9");

  if (/^[1-9][0-9]?$/.test(normalized.trim())) {
    return Number(normalized.trim());
  }

  const match = normalized.match(
    /\b(?:chọn|chon|lựa chọn|lua chon|lấy|lay|xem)\s*(?:số|so|option|lựa chọn|lua chon)?\s*([1-9][0-9]?)\b/,
  );
  return match ? Number(match[1]) : null;
}

function safeSpeak(text) {
  try {
    speak(text);
  } catch (error) {
    console.warn("TTS playback failed", error);
  }
}

function speak(text) {
  if (!("speechSynthesis" in window) || !text) {
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "vi-VN";
  utterance.rate = 1;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

function appendUser(text) {
  appendMessage("user", "Bạn", text);
}

function appendAssistant(text, error = false) {
  appendMessage(error ? "assistant error" : "assistant", "Trợ lý", text);
}

function appendMessage(kind, label, text) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${kind}`;
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = label;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.append(meta, bubble);
  messagesEl.append(wrapper);
  scrollToBottom();
}

function appendProductOptions(options) {
  if (!options.length) {
    return;
  }

  const list = document.createElement("div");
  list.className = "option-list";

  options.slice(0, 4).forEach((option) => {
    const item = document.createElement("article");
    item.className = "product-option";
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.setAttribute("aria-label", `Chọn ${option.name || "sản phẩm"}`);

    const media = document.createElement("div");
    media.className = "product-media";
    if (option.image_url) {
      const image = document.createElement("img");
      image.src = option.image_url;
      image.alt = option.name || "Sản phẩm";
      image.loading = "lazy";
      media.append(image);
    } else {
      const fallback = document.createElement("span");
      fallback.textContent = String(option.index || "+");
      media.append(fallback);
    }

    const content = document.createElement("div");
    content.className = "product-content";
    const title = document.createElement("h3");
    title.className = "product-title";
    title.textContent = `${option.index}. ${option.name || "Sản phẩm"}`;
    content.append(title);

    const meta = document.createElement("div");
    meta.className = "product-meta";
    appendMeta(meta, option.brand);
    appendMeta(meta, option.price);
    appendMeta(meta, option.sku);
    content.append(meta);

    item.addEventListener("click", async () => {
      await selectProductOption(list, item, option);
    });
    item.addEventListener("keydown", async (event) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      event.preventDefault();
      await selectProductOption(list, item, option);
    });

    item.append(media, content);
    list.append(item);
  });

  messagesEl.append(list);
  scrollToBottom();
}

function appendMeta(parent, value) {
  if (!value) {
    return;
  }
  const span = document.createElement("span");
  span.textContent = value;
  parent.append(span);
}

async function selectProductOption(list, item, option) {
  if (item.getAttribute("aria-disabled") === "true") {
    return;
  }
  disableProductOptions(list);
  item.classList.add("selected");
  await sendVoiceTurn(String(option.index), {
    selectedIndex: option.index,
    selectedSku: option.sku,
  });
}

function disableProductOptions(container) {
  container.querySelectorAll(".product-option").forEach((item) => {
    item.setAttribute("aria-disabled", "true");
    item.tabIndex = -1;
  });
}

function setBusy(value) {
  state.busy = value;
  syncControls();
}

function syncControls() {
  micButtonEl.disabled = state.busy;
  micButtonEl.setAttribute("aria-pressed", state.listening ? "true" : "false");
  sendTextButtonEl.disabled = state.busy;
  textFallbackEl.disabled = state.busy;
}

function setStatus(text, kind) {
  statusTextEl.textContent = text;
  statusDotEl.className = `status-dot ${kind}`;
}

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}
