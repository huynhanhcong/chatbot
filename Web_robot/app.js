const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition || null;

const state = {
  activeScreen: "home",
  medicineCode: "",
  qrStream: null,
  qrDetector: null,
  qrAnimationFrame: null,
  qrBusy: false,
  dispenseAbortController: null,
  conversationId: null,
  recognition: null,
  voiceOpen: false,
  voiceBusy: false,
  voiceListening: false,
  voiceSpeaking: false,
  pendingVoiceMedicineSelection: false,
  medicineSelectionOpen: false,
  medicineOptions: [],
};

const screens = {
  home: document.querySelector("#homeScreen"),
  guideList: document.querySelector("#guideListScreen"),
  guideDetail: document.querySelector("#guideDetailScreen"),
  medicine: document.querySelector("#medicineChoiceScreen"),
  manual: document.querySelector("#manualScreen"),
  qr: document.querySelector("#qrScreen"),
  doctor: document.querySelector("#doctorScreen"),
  loading: document.querySelector("#loadingScreen"),
};

const clockTextEl = document.querySelector("#clockText");
const guideButtonEl = document.querySelector("#guideButton");
const doctorButtonEl = document.querySelector("#doctorButton");
const medicineButtonEl = document.querySelector("#medicineButton");
const voiceButtonEl = document.querySelector("#voiceButton");
const manualMedicineButtonEl = document.querySelector("#manualMedicineButton");
const qrMedicineButtonEl = document.querySelector("#qrMedicineButton");
const guideDetailTitleEl = document.querySelector("#guideDetailTitle");
const medicineCodeDisplayEl = document.querySelector("#medicineCodeDisplay");
const medicineErrorEl = document.querySelector("#medicineError");
const manualErrorEl = document.querySelector("#manualError");
const qrVideoEl = document.querySelector("#qrVideo");
const qrCanvasEl = document.querySelector("#qrCanvas");
const qrStatusEl = document.querySelector("#qrStatus");
const qrFallbackButtonEl = document.querySelector("#qrFallbackButton");
const cameraPlaceholderEl = document.querySelector("#cameraPlaceholder");
const voiceOverlayEl = document.querySelector("#voiceOverlay");
const voiceEmojiEl = document.querySelector("#voiceEmoji");
const closeVoiceButtonEl = document.querySelector("#closeVoiceButton");
const voiceStatusEl = document.querySelector("#voiceStatus");
const voiceTranscriptEl = document.querySelector("#voiceTranscript");
const medicineClassGridEl = document.querySelector("#medicineClassGrid");

const GUIDE_TITLES = new Set([
  "Hướng dẫn lấy thuốc",
  "Hướng dẫn đàm thoại với AI",
  "Hướng dẫn sử dụng 1",
  "Hướng dẫn sử dụng 2",
]);

init();

function init() {
  setupOptionalIcons();
  updateClock();
  window.setInterval(updateClock, 1000);

  guideButtonEl.addEventListener("click", () => showScreen("guideList"));
  doctorButtonEl.addEventListener("click", () => showScreen("doctor"));
  medicineButtonEl.addEventListener("click", () => {
    clearMedicineErrors();
    showScreen("medicine");
  });
  voiceButtonEl.addEventListener("click", openVoiceMode);
  closeVoiceButtonEl.addEventListener("click", closeVoiceMode);

  manualMedicineButtonEl.addEventListener("click", () => {
    resetMedicineCode();
    showScreen("manual");
  });
  qrMedicineButtonEl.addEventListener("click", () => {
    clearMedicineErrors();
    showScreen("qr");
    startQrScanner();
  });
  qrFallbackButtonEl.addEventListener("click", () => {
    stopQrScanner();
    resetMedicineCode();
    showScreen("manual");
  });

  document.querySelectorAll("[data-close]").forEach((button) => {
    button.addEventListener("click", returnHome);
  });

  document.querySelectorAll("[data-guide-title]").forEach((button) => {
    button.addEventListener("click", () => {
      const title = button.dataset.guideTitle || "Hướng dẫn lấy thuốc";
      guideDetailTitleEl.textContent = GUIDE_TITLES.has(title) ? title : "Hướng dẫn lấy thuốc";
      showScreen("guideDetail");
    });
  });

  document.querySelectorAll("[data-digit]").forEach((button) => {
    button.addEventListener("click", () => appendMedicineDigit(button.dataset.digit || ""));
  });

  document.querySelector("[data-action='delete']").addEventListener("click", deleteMedicineDigit);
  document.querySelector("[data-action='confirm']").addEventListener("click", confirmManualMedicineCode);
  updateMedicineCodeDisplay();
}

function setupOptionalIcons() {
  document.querySelectorAll("[data-optional-icon]").forEach((image) => {
    const hideMissingIcon = () => {
      image.hidden = true;
    };
    image.addEventListener("error", hideMissingIcon, { once: true });
    if (image.complete && image.naturalWidth === 0) {
      hideMissingIcon();
    }
  });
}

function updateClock() {
  const now = new Date();
  clockTextEl.textContent = new Intl.DateTimeFormat("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(now);
  clockTextEl.setAttribute(
    "aria-label",
    new Intl.DateTimeFormat("vi-VN", {
      weekday: "long",
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(now),
  );
}

function showScreen(name) {
  Object.values(screens).forEach((screen) => screen.classList.remove("is-active"));
  screens[name].classList.add("is-active");
  state.activeScreen = name;
}

function returnHome() {
  stopQrScanner();
  abortDispenseRequest();
  clearMedicineErrors();
  showScreen("home");
}

function clearMedicineErrors() {
  medicineErrorEl.hidden = true;
  medicineErrorEl.textContent = "";
  manualErrorEl.hidden = true;
  manualErrorEl.textContent = "";
}

function showMedicineError(message) {
  medicineErrorEl.textContent = message;
  medicineErrorEl.hidden = false;
  showScreen("medicine");
}

function resetMedicineCode() {
  state.medicineCode = "";
  clearMedicineErrors();
  updateMedicineCodeDisplay();
}

function appendMedicineDigit(digit) {
  if (!/^\d$/.test(digit) || state.medicineCode.length >= 16) {
    return;
  }
  state.medicineCode += digit;
  clearMedicineErrors();
  updateMedicineCodeDisplay();
}

function deleteMedicineDigit() {
  state.medicineCode = state.medicineCode.slice(0, -1);
  updateMedicineCodeDisplay();
}

function updateMedicineCodeDisplay() {
  medicineCodeDisplayEl.textContent = state.medicineCode;
}

async function confirmManualMedicineCode() {
  if (!state.medicineCode) {
    manualErrorEl.textContent = "Vui lòng nhập mã lấy thuốc.";
    manualErrorEl.hidden = false;
    return;
  }
  await submitMedicineCode(state.medicineCode, "manual");
}

async function submitMedicineCode(rawCode, source) {
  const code = String(rawCode || "").trim();
  if (!code) {
    showMedicineError("Không đọc được mã lấy thuốc. Vui lòng thử lại.");
    return;
  }

  stopQrScanner();
  clearMedicineErrors();
  showScreen("loading");

  const abortController = new AbortController();
  state.dispenseAbortController = abortController;

  try {
    const response = await fetch("/robot/api/medicine/dispense", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, source }),
      signal: abortController.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "Không thể gửi yêu cầu lấy thuốc.");
    }
    state.medicineCode = "";
    updateMedicineCodeDisplay();
    showScreen("home");
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    showMedicineError(error.message || "Không thể lấy thuốc. Vui lòng thử lại.");
  } finally {
    if (state.dispenseAbortController === abortController) {
      state.dispenseAbortController = null;
    }
  }
}

function abortDispenseRequest() {
  if (!state.dispenseAbortController) {
    return;
  }
  state.dispenseAbortController.abort();
  state.dispenseAbortController = null;
}

async function startQrScanner() {
  stopQrScanner();
  qrFallbackButtonEl.hidden = true;
  qrStatusEl.textContent = "Đưa mã QR vào giữa khung camera";
  qrStatusEl.style.color = "var(--ink)";
  cameraPlaceholderEl.textContent = "CAMERA";

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showQrFallback("Trình duyệt không hỗ trợ camera.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    state.qrStream = stream;
    qrVideoEl.srcObject = stream;
    await qrVideoEl.play();
    qrVideoEl.parentElement.classList.add("has-video");

    if (!("BarcodeDetector" in window)) {
      showQrFallback("Trình duyệt chưa hỗ trợ nhận QR.");
      return;
    }

    state.qrDetector = new window.BarcodeDetector({ formats: ["qr_code"] });
    state.qrBusy = false;
    scanQrFrame();
  } catch (error) {
    showQrFallback("Không mở được camera.");
  }
}

async function scanQrFrame() {
  if (!state.qrDetector || state.qrBusy || !qrVideoEl.srcObject) {
    return;
  }
  state.qrBusy = true;
  try {
    const codes = await state.qrDetector.detect(qrVideoEl);
    const value = codes[0]?.rawValue?.trim();
    if (value) {
      await submitMedicineCode(value, "qr");
      return;
    }
  } catch (error) {
    const fallbackValue = scanQrWithCanvasFallback();
    if (fallbackValue) {
      await submitMedicineCode(fallbackValue, "qr");
      return;
    }
  } finally {
    state.qrBusy = false;
  }

  if (qrVideoEl.srcObject) {
    state.qrAnimationFrame = window.requestAnimationFrame(scanQrFrame);
  }
}

function scanQrWithCanvasFallback() {
  if (!window.jsQR || !qrVideoEl.videoWidth || !qrVideoEl.videoHeight) {
    return "";
  }
  const context = qrCanvasEl.getContext("2d", { willReadFrequently: true });
  qrCanvasEl.width = qrVideoEl.videoWidth;
  qrCanvasEl.height = qrVideoEl.videoHeight;
  context.drawImage(qrVideoEl, 0, 0, qrCanvasEl.width, qrCanvasEl.height);
  const imageData = context.getImageData(0, 0, qrCanvasEl.width, qrCanvasEl.height);
  const result = window.jsQR(imageData.data, imageData.width, imageData.height);
  return result?.data?.trim() || "";
}

function showQrFallback(message) {
  qrStatusEl.textContent = message;
  qrStatusEl.style.color = "var(--danger)";
  qrFallbackButtonEl.hidden = false;
}

function stopQrScanner() {
  if (state.qrAnimationFrame) {
    window.cancelAnimationFrame(state.qrAnimationFrame);
    state.qrAnimationFrame = null;
  }
  if (state.qrStream) {
    state.qrStream.getTracks().forEach((track) => track.stop());
    state.qrStream = null;
  }
  qrVideoEl.pause();
  qrVideoEl.removeAttribute("src");
  qrVideoEl.srcObject = null;
  qrVideoEl.parentElement.classList.remove("has-video");
  state.qrDetector = null;
  state.qrBusy = false;
}

function openVoiceMode() {
  state.voiceOpen = true;
  voiceOverlayEl.hidden = false;
  clearMedicineSelectionOverlay();
  setVoiceState("idle", "Sẵn sàng nghe");
  voiceTranscriptEl.textContent = "";

  if (!SpeechRecognition) {
    setVoiceState("error", "Trình duyệt không hỗ trợ giọng nói");
    return;
  }

  setupRecognition();
  startListening();
}

function closeVoiceMode() {
  state.voiceOpen = false;
  state.voiceListening = false;
  state.voiceSpeaking = false;
  clearMedicineSelectionOverlay();
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  if (state.recognition) {
    try {
      state.recognition.stop();
    } catch (error) {
      console.warn("Voice recognition stop failed", error);
    }
  }
  voiceOverlayEl.hidden = true;
}

function setupRecognition() {
  if (state.recognition || !SpeechRecognition) {
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "vi-VN";
  recognition.interimResults = true;
  recognition.continuous = false;

  recognition.addEventListener("start", () => {
    state.voiceListening = true;
    setVoiceState("listening", "Đang nghe");
  });

  recognition.addEventListener("end", () => {
    state.voiceListening = false;
    if (state.voiceOpen && !state.voiceBusy && !state.voiceSpeaking) {
      startListening();
    }
  });

  recognition.addEventListener("error", (event) => {
    state.voiceListening = false;
    if (state.voiceOpen) {
      setVoiceState("error", getVoiceErrorMessage(event.error));
    }
  });

  recognition.addEventListener("result", async (event) => {
    let finalTranscript = "";
    let interimTranscript = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      if (result.isFinal) {
        finalTranscript += result[0].transcript;
      } else {
        interimTranscript += result[0].transcript;
      }
    }
    voiceTranscriptEl.textContent = finalTranscript || interimTranscript;
    if (finalTranscript.trim()) {
      await handleVoiceTranscript(finalTranscript.trim());
    }
  });

  state.recognition = recognition;
}

function startListening() {
  if (
    !state.voiceOpen ||
    state.voiceBusy ||
    state.voiceSpeaking ||
    state.voiceListening ||
    state.medicineSelectionOpen ||
    !state.recognition
  ) {
    return;
  }
  try {
    state.recognition.start();
  } catch (error) {
    setVoiceState("error", "Không thể bật microphone");
  }
}

async function handleVoiceTranscript(transcript) {
  state.voiceBusy = true;
  setVoiceState("processing", "Đang xử lý");
  const aiStartedAt = performance.now();
  let aiMetricLogged = false;
  try {
    const payload = { message: transcript };
    if (state.conversationId) {
      payload.conversation_id = state.conversationId;
    }
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    logVoiceMetric("ai_response", performance.now() - aiStartedAt);
    aiMetricLogged = true;
    if (!response.ok) {
      throw new Error(data.detail || "Không thể xử lý câu hỏi.");
    }
    if (data.conversation_id) {
      state.conversationId = data.conversation_id;
    }
    if (data.status === "need_selection" && data.route === "pharmacity") {
      openMedicineSelectionOverlay(data.options || []);
      return;
    }
    const answer = data.answer || data.message || "Tôi đã nhận được yêu cầu.";
    speakAnswer(answer);
  } catch (error) {
    if (!aiMetricLogged) {
      logVoiceMetric("ai_response", performance.now() - aiStartedAt);
    }
    speakAnswer("Xin lỗi, tôi chưa xử lý được yêu cầu. Vui lòng thử lại.");
  } finally {
    state.voiceBusy = false;
  }
}

function openMedicineSelectionOverlay(options) {
  const visibleOptions = Array.isArray(options) ? options.slice(0, 4) : [];
  state.pendingVoiceMedicineSelection = true;
  state.medicineSelectionOpen = true;
  state.medicineOptions = visibleOptions;

  if (state.recognition && state.voiceListening) {
    try {
      state.recognition.stop();
    } catch (error) {
      console.warn("Voice recognition stop failed", error);
    }
  }

  if (!medicineClassGridEl || visibleOptions.length === 0) {
    clearMedicineSelectionOverlay();
    speakAnswer("Tôi chưa tìm được danh sách thuốc phù hợp. Bạn vui lòng thử hỏi lại.");
    return;
  }

  medicineClassGridEl.replaceChildren();
  visibleOptions.forEach((option, optionIndex) => {
    const button = document.createElement("button");
    button.className = "medicine-class-card";
    button.type = "button";
    button.dataset.index = String(option.index || optionIndex + 1);
    button.dataset.sku = option.sku || "";

    const media = document.createElement("span");
    media.className = "medicine-class-media";

    if (option.image_url) {
      const image = document.createElement("img");
      image.src = option.image_url;
      image.alt = "";
      image.loading = "lazy";
      media.appendChild(image);
    } else {
      const placeholder = document.createElement("span");
      placeholder.className = "medicine-class-placeholder";
      placeholder.textContent = "Thuốc";
      media.appendChild(placeholder);
    }

    const name = document.createElement("span");
    name.className = "medicine-class-name";
    name.textContent = option.name || `Thuốc ${option.index || optionIndex + 1}`;

    button.append(media, name);
    button.addEventListener("click", () => selectMedicineClassOption(option, button));
    medicineClassGridEl.appendChild(button);
  });

  medicineClassGridEl.hidden = false;
  setVoiceState("idle", "Chọn thuốc cần hỏi");
}

function clearMedicineSelectionOverlay() {
  state.pendingVoiceMedicineSelection = false;
  state.medicineSelectionOpen = false;
  state.medicineOptions = [];
  if (!medicineClassGridEl) {
    return;
  }
  medicineClassGridEl.hidden = true;
  medicineClassGridEl.replaceChildren();
}

async function selectMedicineClassOption(option, selectedButton) {
  if (!state.voiceOpen || state.voiceBusy || !state.pendingVoiceMedicineSelection) {
    return;
  }

  const selectedIndex = Number(option.index || selectedButton?.dataset.index);
  if (!Number.isFinite(selectedIndex) || selectedIndex < 1) {
    speakAnswer("Tôi chưa chọn được thuốc này. Bạn vui lòng thử lại.");
    return;
  }

  state.voiceBusy = true;
  setVoiceState("processing", "Đang kiểm tra thuốc");
  setMedicineSelectionDisabled(true, selectedButton);
  const aiStartedAt = performance.now();
  let aiMetricLogged = false;

  try {
    const payload = {
      message: String(selectedIndex),
      selected_index: selectedIndex,
    };
    if (option.sku) {
      payload.selected_sku = option.sku;
    }
    if (state.conversationId) {
      payload.conversation_id = state.conversationId;
    }

    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    logVoiceMetric("ai_response", performance.now() - aiStartedAt, "Thời gian phản hồi của AI khi chọn thuốc");
    aiMetricLogged = true;
    if (!response.ok) {
      throw new Error(data.detail || "Không thể lấy thông tin thuốc.");
    }
    if (data.conversation_id) {
      state.conversationId = data.conversation_id;
    }

    clearMedicineSelectionOverlay();
    const answer = data.answer || data.message || "Tôi đã lấy được thông tin thuốc.";
    speakAnswer(answer);
  } catch (error) {
    if (!aiMetricLogged) {
      logVoiceMetric("ai_response", performance.now() - aiStartedAt, "Thời gian phản hồi của AI khi chọn thuốc");
    }
    setMedicineSelectionDisabled(false);
    speakAnswer("Xin lỗi, tôi chưa lấy được thông tin thuốc. Bạn vui lòng thử lại.");
  } finally {
    state.voiceBusy = false;
  }
}

function setMedicineSelectionDisabled(disabled, selectedButton = null) {
  if (!medicineClassGridEl) {
    return;
  }
  medicineClassGridEl.querySelectorAll(".medicine-class-card").forEach((button) => {
    button.disabled = disabled;
    button.classList.toggle("is-selected", button === selectedButton);
  });
}

function logVoiceMetric(metric, elapsedMs, label = "") {
  const payload = {
    metric,
    elapsed_ms: Number(elapsedMs.toFixed(2)),
  };
  if (state.conversationId) {
    payload.conversation_id = state.conversationId;
  }
  if (label) {
    payload.label = label;
  }

  const messageLabel =
    label ||
    {
      ai_response: "Thời gian phản hồi của AI",
      voice_response: "Thời gian phản hồi của voice",
    }[metric] ||
    metric;
  console.info(`${messageLabel}: ${payload.elapsed_ms} ms (${(payload.elapsed_ms / 1000).toFixed(2)} s)`);

  fetch("/robot/api/voice-metrics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    keepalive: true,
  }).catch(() => {});
}

function speakAnswer(text) {
  if (!state.voiceOpen) {
    return;
  }
  if (!("speechSynthesis" in window) || !text) {
    setVoiceState("idle", state.medicineSelectionOpen ? "Chọn thuốc cần hỏi" : "Sẵn sàng nghe");
    startListening();
    return;
  }

  window.speechSynthesis.cancel();
  state.voiceSpeaking = true;
  setVoiceState("speaking", "Đang trả lời");
  const voiceStartedAt = performance.now();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "vi-VN";
  utterance.rate = 0.95;
  utterance.pitch = 1;
  utterance.addEventListener("end", () => {
    logVoiceMetric("voice_response", performance.now() - voiceStartedAt);
    state.voiceSpeaking = false;
    if (state.voiceOpen) {
      setVoiceState("idle", state.medicineSelectionOpen ? "Chọn thuốc cần hỏi" : "Sẵn sàng nghe");
      startListening();
    }
  });
  utterance.addEventListener("error", () => {
    logVoiceMetric("voice_response", performance.now() - voiceStartedAt);
    state.voiceSpeaking = false;
    if (state.voiceOpen) {
      setVoiceState("error", "Không thể phát giọng nói");
    }
  });
  window.speechSynthesis.speak(utterance);
}

function setVoiceState(kind, text) {
  voiceOverlayEl.dataset.state = kind;
  voiceStatusEl.textContent = text;
  setEmojiExpression(
    {
      idle: "happy",
      listening: "happy",
      processing: "neutral",
      speaking: "happy",
      error: "sad",
    }[kind] || "happy",
  );
}

function setEmojiExpression(expression) {
  if (!voiceEmojiEl) {
    return;
  }
  voiceEmojiEl.dataset.expression = expression;
}

function getVoiceErrorMessage(errorCode) {
  const messages = {
    "not-allowed": "Chưa cấp quyền microphone",
    "service-not-allowed": "Chưa cấp quyền giọng nói",
    "no-speech": "Chưa nghe thấy giọng nói",
    "audio-capture": "Không tìm thấy microphone",
    network: "Mất kết nối giọng nói",
    aborted: "Đã dừng nghe",
  };
  return messages[errorCode] || "Lỗi microphone";
}
