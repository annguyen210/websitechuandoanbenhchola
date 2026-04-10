const state = {
  file: null,
  previewUrl: "",
};

const elements = {
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  previewImage: document.getElementById("previewImage"),
  fileName: document.getElementById("fileName"),
  analyzeButton: document.getElementById("analyzeButton"),
  resetButton: document.getElementById("resetButton"),
  resultsCard: document.getElementById("resultsCard"),
  messageBanner: document.getElementById("messageBanner"),
  pipelineList: document.getElementById("pipelineList"),
  serverStatus: document.getElementById("serverStatus"),
  healthYolo: document.getElementById("healthYolo"),
  healthCnn: document.getElementById("healthCnn"),
  healthLabels: document.getElementById("healthLabels"),
  healthOpenAi: document.getElementById("healthOpenAi"),
  processingTime: document.getElementById("processingTime"),
  resultOriginal: document.getElementById("resultOriginal"),
  resultAnnotated: document.getElementById("resultAnnotated"),
  resultCrop: document.getElementById("resultCrop"),
  cnnHeadline: document.getElementById("cnnHeadline"),
  cnnLabel: document.getElementById("cnnLabel"),
  cnnConfidence: document.getElementById("cnnConfidence"),
  cnnWarning: document.getElementById("cnnWarning"),
  predictionList: document.getElementById("predictionList"),
  llmSource: document.getElementById("llmSource"),
  llmHeadline: document.getElementById("llmHeadline"),
  llmSummary: document.getElementById("llmSummary"),
  careSteps: document.getElementById("careSteps"),
  nextSteps: document.getElementById("nextSteps"),
  llmWarning: document.getElementById("llmWarning"),
};

function basePipeline() {
  return [
    {
      title: "YOLO nhận diện lá",
      detail: "Tách vùng lá rõ nhất trước khi đưa sang CNN.",
    },
    {
      title: "CNN phân loại",
      detail: "Đọc ảnh crop và tính xác suất cho từng lớp của model_0.h5.",
    },
    {
      title: "ChatGPT tư vấn",
      detail: "Tóm tắt ngắn gọn, dễ hiểu, có bước chăm sóc tiếp theo.",
    },
  ];
}

async function init() {
  renderPipeline(basePipeline());
  bindEvents();
  await loadHealth();
}

function bindEvents() {
  elements.fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    applyFile(file);
  });

  elements.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("drag-over");
  });

  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.classList.remove("drag-over");
  });

  elements.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("drag-over");
    const [file] = event.dataTransfer.files;
    applyFile(file);
  });

  elements.analyzeButton.addEventListener("click", analyzeImage);
  elements.resetButton.addEventListener("click", resetForm);
}

function applyFile(file) {
  if (!file) {
    return;
  }

  state.file = file;
  elements.fileName.textContent = file.name;
  elements.analyzeButton.disabled = false;

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = state.previewUrl;
  elements.previewImage.classList.remove("is-empty");
}

function resetForm() {
  state.file = null;

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = "";
  elements.fileInput.value = "";
  elements.fileName.textContent = "Chưa chọn ảnh";
  elements.previewImage.removeAttribute("src");
  elements.previewImage.classList.add("is-empty");
  elements.analyzeButton.disabled = true;
  elements.resultsCard.classList.add("hidden");
  hideBanner();
  renderPipeline(basePipeline());
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const dependencies = data.dependencies;

    elements.serverStatus.textContent = "Sẵn sàng";
    elements.serverStatus.className = "status-pill success";
    elements.healthYolo.textContent =
      dependencies.yolo_model_found && dependencies.ultralytics_ready
        ? "Sẵn sàng"
        : dependencies.yolo_model_found
          ? "Thiếu ultralytics"
          : "Thiếu model";
    elements.healthCnn.textContent =
      dependencies.cnn_model_found && dependencies.tensorflow_ready
        ? "Sẵn sàng"
        : dependencies.cnn_model_found
          ? "Thiếu TensorFlow"
          : "Thiếu model";
    elements.healthLabels.textContent = dependencies.cnn_labels_found ? "Có file nhãn" : "Đang dùng nhãn mẫu";
    elements.healthOpenAi.textContent = dependencies.openai_key_configured ? "Đã cấu hình" : "Chưa có API key";

    if (!dependencies.ultralytics_ready || !dependencies.tensorflow_ready) {
      showBanner(
        "Một số thư viện ML chưa có trong môi trường hiện tại. Web vẫn chạy, nhưng có thể dùng chế độ dự phòng thay cho suy luận model thật.",
        "info"
      );
    }
  } catch (error) {
    elements.serverStatus.textContent = "Không kết nối";
    elements.serverStatus.className = "status-pill warning";
    elements.healthYolo.textContent = "Không rõ";
    elements.healthCnn.textContent = "Không rõ";
    elements.healthLabels.textContent = "Không rõ";
    elements.healthOpenAi.textContent = "Không rõ";
  }
}

async function analyzeImage() {
  if (!state.file) {
    return;
  }

  setLoadingState(true);
  hideBanner();
  renderPipeline(basePipeline());

  const formData = new FormData();
  formData.append("image", state.file);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      let message = `Lỗi máy chủ: ${response.status}`;

      try {
        const payload = JSON.parse(text);
        if (payload?.error) {
          message = payload.error;
        }
      } catch {
        if (text) {
          message = text;
        }
      }

      throw new Error(message);
    }

    const payload = await response.json();

    if (!payload.success) {
      throw new Error(payload.error || "Không thể phân tích ảnh.");
    }

    renderResult(payload.result);
  } catch (error) {
    showBanner(error.message, "error");
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  elements.analyzeButton.disabled = isLoading || !state.file;
  elements.analyzeButton.textContent = isLoading ? "Đang phân tích..." : "Phân tích ngay";
  elements.serverStatus.textContent = isLoading ? "Đang xử lý" : "Sẵn sàng";
  elements.serverStatus.className = isLoading ? "status-pill warning" : "status-pill success";
}

function renderPipeline(items) {
  elements.pipelineList.innerHTML = items
    .map((item, index) => {
      const title = item.step || item.title;
      const detail = item.detail || "";
      const durationText = item.duration_ms ? `<br />Thời gian: ${item.duration_ms} ms` : "";
      return `
        <article class="pipeline-item">
          <span class="step-index">${index + 1}</span>
          <div>
            <h3>${escapeHtml(title)}</h3>
            <p>${escapeHtml(detail)}${durationText}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderResult(result) {
  elements.resultsCard.classList.remove("hidden");
  elements.processingTime.textContent = `${result.meta.total_duration_ms} ms`;

  renderPipeline(result.pipeline);
  renderImages(result.images);
  renderClassification(result.classification);
  renderAdvice(result.llm);

  elements.resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderImages(images) {
  elements.resultOriginal.src = images.original || "";
  elements.resultAnnotated.src = images.annotated || "";
  elements.resultCrop.src = images.cropped_leaf || "";
}

function renderClassification(classification) {
  elements.cnnHeadline.textContent = `${classification.input_size.width} x ${classification.input_size.height}`;
  elements.cnnLabel.textContent = classification.display_label;
  elements.cnnConfidence.textContent = `Độ tin cậy: ${(classification.confidence * 100).toFixed(2)}%`;

  if (classification.warning) {
    elements.cnnWarning.textContent = classification.warning;
    elements.cnnWarning.classList.remove("hidden");
  } else {
    elements.cnnWarning.textContent = "";
    elements.cnnWarning.classList.add("hidden");
  }

  elements.predictionList.innerHTML = classification.top_predictions
    .map(
      (item) => `
        <div class="prediction-item">
          <div class="prediction-row">
            <strong>${escapeHtml(item.display_label)}</strong>
            <span>${(item.confidence * 100).toFixed(2)}%</span>
          </div>
          <div class="prediction-bar">
            <span style="width: ${(item.confidence * 100).toFixed(2)}%"></span>
          </div>
        </div>
      `
    )
    .join("");
}

function renderAdvice(llm) {
  elements.llmSource.textContent = `Nguồn: ${llm.source} (${llm.model})`;
  elements.llmHeadline.textContent = llm.headline || "-";
  elements.llmSummary.textContent = llm.summary || "-";
  elements.llmWarning.textContent = llm.warning || "Không có ghi chú thêm.";

  renderList(elements.careSteps, llm.care_steps);
  renderList(elements.nextSteps, llm.next_steps);
}

function renderList(target, items) {
  target.innerHTML = (items || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showBanner(message, type) {
  elements.messageBanner.textContent = message;
  elements.messageBanner.className = `message-banner ${type}`;
}

function hideBanner() {
  elements.messageBanner.textContent = "";
  elements.messageBanner.className = "message-banner hidden";
}

init();
