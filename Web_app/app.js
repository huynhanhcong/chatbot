const state = {
  conversationId: null,
  busy: false,
};

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const sendButtonEl = document.querySelector("#sendButton");
const newChatButtonEl = document.querySelector("#newChatButton");
const quickPromptEls = document.querySelectorAll("[data-prompt]");

const welcomeText =
  "Xin chào. Bạn có thể hỏi thông tin thuốc hoặc dữ liệu bệnh viện Hạnh Phúc.";

appendAssistantMessage(welcomeText);

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message || state.busy) {
    return;
  }
  inputEl.value = "";
  resizeComposer();
  await sendChat({ message });
});

inputEl.addEventListener("input", resizeComposer);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

newChatButtonEl.addEventListener("click", () => {
  state.conversationId = null;
  messagesEl.replaceChildren();
  appendAssistantMessage(welcomeText);
  inputEl.focus();
});

quickPromptEls.forEach((button) => {
  button.addEventListener("click", async () => {
    if (state.busy) {
      return;
    }
    await sendChat({ message: button.dataset.prompt || "" });
  });
});

async function sendChat({ message, selectedIndex = null, selectedSku = null, displayUser = true }) {
  if (!message.trim()) {
    return;
  }

  if (displayUser) {
    appendUserMessage(message);
  }

  setBusy(true);
  const loadingEl = appendLoadingMessage();
  try {
    const payload = { message };
    if (state.conversationId) {
      payload.conversation_id = state.conversationId;
    }
    if (selectedIndex !== null) {
      payload.selected_index = selectedIndex;
    }
    if (selectedSku) {
      payload.selected_sku = selectedSku;
    }

    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    loadingEl.remove();
    renderAssistantResponse(data);
  } catch (error) {
    loadingEl.remove();
    appendAssistantMessage(error.message || "Không thể xử lý yêu cầu.", { error: true });
  } finally {
    setBusy(false);
  }
}

function renderAssistantResponse(data) {
  if (data.conversation_id) {
    state.conversationId = data.conversation_id;
  }

  if (data.status === "need_selection") {
    appendAssistantMessage(data.message || "Tôi tìm thấy các lựa chọn phù hợp.");
    appendProductOptions(data.options || []);
    return;
  }

  if (data.status === "not_found") {
    appendAssistantMessage(data.message || "Không tìm thấy kết quả phù hợp.");
    return;
  }

  appendAssistantMessage(data.answer || data.message || "Không có nội dung trả lời.");
  appendSources(data.sources || []);
}

function appendUserMessage(text) {
  const wrapper = createMessageShell("user", "Bạn");
  wrapper.querySelector(".bubble").textContent = text;
  messagesEl.append(wrapper);
  scrollToBottom();
}

function appendAssistantMessage(text, options = {}) {
  const wrapper = createMessageShell("assistant", "Trợ lý");
  const bubble = wrapper.querySelector(".bubble");
  if (options.error) {
    bubble.classList.add("error");
    bubble.textContent = text;
  } else {
    renderFormattedText(bubble, text);
  }
  messagesEl.append(wrapper);
  scrollToBottom();
  return wrapper;
}

function appendLoadingMessage() {
  const wrapper = createMessageShell("assistant", "Trợ lý");
  const bubble = wrapper.querySelector(".bubble");
  bubble.replaceChildren(createTypingIndicator());
  messagesEl.append(wrapper);
  scrollToBottom();
  return wrapper;
}

function appendProductOptions(options) {
  const list = document.createElement("div");
  list.className = "option-list";

  options.slice(0, 4).forEach((option) => {
    const item = document.createElement("article");
    item.className = "product-option";

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
      fallback.textContent = "+";
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
    if (option.detail_url) {
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = option.detail_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Pharmacity";
      meta.append(link);
    }
    content.append(meta);

    const button = document.createElement("button");
    button.className = "select-button";
    button.type = "button";
    button.textContent = "Chọn";
    button.addEventListener("click", async () => {
      disableOptionButtons(list);
      await sendChat({
        message: String(option.index),
        selectedIndex: option.index,
        selectedSku: option.sku,
        displayUser: true,
      });
    });

    item.append(media, content, button);
    list.append(item);
  });

  messagesEl.append(list);
  scrollToBottom();
}

function appendSources(sources) {
  if (!sources.length) {
    return;
  }

  const list = document.createElement("div");
  list.className = "source-list";
  sources.forEach((source, index) => {
    const link = document.createElement("a");
    link.href = source.url || source.source_url || "#";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = source.title || `Nguồn ${index + 1}`;
    list.append(link);
  });
  messagesEl.append(list);
  scrollToBottom();
}

function renderFormattedText(container, rawText) {
  const text = normalizeAnswerText(rawText || "");
  container.classList.add("formatted-answer");
  container.replaceChildren();

  let currentList = null;
  const lines = text.split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      currentList = null;
      continue;
    }

    const bulletMatch = trimmed.match(/^[-*•]\s+(.+)$/);
    if (bulletMatch) {
      if (!currentList) {
        currentList = document.createElement("ul");
        container.append(currentList);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, bulletMatch[1]);
      currentList.append(item);
      continue;
    }

    currentList = null;
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, trimmed);
    container.append(paragraph);
  }

  if (!container.childNodes.length) {
    container.textContent = rawText || "";
  }
}

function normalizeAnswerText(rawText) {
  return String(rawText)
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/([^\n])\s+([*•])\s+(?=(?:\*\*)?[\p{L}\d])/gu, "$1\n$2 ")
    .replace(/([^\n])\s+-\s+(?=(?:\*\*)?[\p{L}\d])/gu, "$1\n- ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function appendInlineMarkdown(parent, text) {
  const tokenPattern = /(\*\*[^*]+\*\*|https?:\/\/[^\s)]+)/g;
  let lastIndex = 0;
  let match;

  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parent.append(document.createTextNode(text.slice(lastIndex, match.index)));
    }

    const token = match[0];
    if (token.startsWith("**") && token.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else {
      const link = document.createElement("a");
      link.href = token;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = token;
      parent.append(link);
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parent.append(document.createTextNode(text.slice(lastIndex)));
  }
}

function createMessageShell(kind, label) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${kind}`;

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = label;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  wrapper.append(meta, bubble);
  return wrapper;
}

function createTypingIndicator() {
  const indicator = document.createElement("span");
  indicator.className = "typing";
  indicator.setAttribute("aria-label", "Đang xử lý");
  indicator.append(document.createElement("span"));
  indicator.append(document.createElement("span"));
  indicator.append(document.createElement("span"));
  return indicator;
}

function appendMeta(parent, value) {
  if (!value) {
    return;
  }
  const span = document.createElement("span");
  span.textContent = value;
  parent.append(span);
}

function disableOptionButtons(container) {
  container.querySelectorAll("button").forEach((button) => {
    button.disabled = true;
  });
}

function setBusy(value) {
  state.busy = value;
  sendButtonEl.disabled = value;
  inputEl.disabled = value;
}

function resizeComposer() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
}

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}
