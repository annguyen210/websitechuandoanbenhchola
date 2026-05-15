const state = {
  file: null,
  previewUrl: "",
  result: null,
  qrDataUrl: "",
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
  healthGemini: document.getElementById("healthGemini"),
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
  healthScoreBar: document.getElementById("healthScoreBar"),
  healthScoreValue: document.getElementById("healthScoreValue"),
  plantType: document.getElementById("plantType"),
  spreadLevel: document.getElementById("spreadLevel"),
  economicImpact: document.getElementById("economicImpact"),
  treatmentSchedule: document.getElementById("treatmentSchedule"),
  diseaseProgression: document.getElementById("diseaseProgression"),
  diagnosisBadge: document.getElementById("diagnosisBadge"),
  finalDiagnosis: document.getElementById("finalDiagnosis"),
  cnnAgreementBadge: document.getElementById("cnnAgreementBadge"),
  diseaseEvidenceSection: document.getElementById("diseaseEvidenceSection"),
  diseaseEvidence: document.getElementById("diseaseEvidence"),
  recommendations: document.getElementById("recommendations"),
  recommendationsSection: document.getElementById("recommendationsSection"),
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
      title: "Tư vấn AI",
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
  if (elements.fileInput) {
    elements.fileInput.addEventListener("change", (event) => {
      const [file] = event.target.files;
      applyFile(file);
    });
  }

  if (elements.dropzone) {
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
  }

  if (elements.analyzeButton) {
    elements.analyzeButton.addEventListener("click", analyzeImage);
  }
  if (elements.resetButton) {
    elements.resetButton.addEventListener("click", resetForm);
  }

  const shareReportBtn = document.getElementById("shareReportBtn");
  if (shareReportBtn) {
    shareReportBtn.addEventListener("click", downloadReport);
  }

  const qrCodeBtn = document.getElementById("qrCodeBtn");
  if (qrCodeBtn) {
    qrCodeBtn.addEventListener("click", openQrModal);
  }

  const qrModalClose = document.getElementById("qrModalClose");
  if (qrModalClose) {
    qrModalClose.addEventListener("click", closeQrModal);
  }

  const qrCloseBtn = document.getElementById("qrCloseBtn");
  if (qrCloseBtn) {
    qrCloseBtn.addEventListener("click", closeQrModal);
  }

  const qrDownloadBtn = document.getElementById("qrDownloadBtn");
  if (qrDownloadBtn) {
    qrDownloadBtn.addEventListener("click", downloadQrImage);
  }

  const qrModal = document.getElementById("qrModal");
  if (qrModal) {
    qrModal.addEventListener("click", (e) => {
      if (e.target === qrModal) closeQrModal();
    });
  }
}

function applyFile(file) {
  if (!file) {
    return;
  }

  state.file = file;
  if (elements.fileName) {
    elements.fileName.textContent = file.name;
  }
  if (elements.analyzeButton) {
    elements.analyzeButton.disabled = false;
  }

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = URL.createObjectURL(file);
  if (elements.previewImage) {
    elements.previewImage.src = state.previewUrl;
    elements.previewImage.classList.remove("is-empty");
  }
}

function resetForm() {
  state.file = null;

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = "";
  if (elements.fileInput) {
    elements.fileInput.value = "";
  }
  if (elements.fileName) {
    elements.fileName.textContent = "Chưa chọn ảnh";
  }
  if (elements.previewImage) {
    elements.previewImage.removeAttribute("src");
    elements.previewImage.classList.add("is-empty");
  }
  if (elements.analyzeButton) {
    elements.analyzeButton.disabled = true;
  }
  if (elements.resultsCard) {
    elements.resultsCard.classList.add("hidden");
  }
  hideBanner();
  renderPipeline(basePipeline());
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const dependencies = data.dependencies;

    if (elements.serverStatus) {
      elements.serverStatus.textContent = "Sẵn sàng";
      elements.serverStatus.className = "status-pill success";
    }
    if (elements.healthYolo) {
      elements.healthYolo.textContent =
        dependencies.yolo_model_found && dependencies.ultralytics_ready
          ? "Sẵn sàng"
          : dependencies.yolo_model_found
            ? "Thiếu ultralytics"
            : "Thiếu model";
    }
    if (elements.healthCnn) {
      elements.healthCnn.textContent =
        dependencies.cnn_model_found && dependencies.tensorflow_ready
          ? "Sẵn sàng"
          : dependencies.cnn_model_found
            ? "Thiếu TensorFlow"
            : "Thiếu model";
    }
    if (elements.healthLabels) {
      elements.healthLabels.textContent = dependencies.cnn_labels_found ? "Có file nhãn" : "Đang dùng nhãn mẫu";
    }
    if (elements.healthGemini) {
      elements.healthGemini.textContent = dependencies.gemini_key_configured ? "Đã cấu hình" : "Chưa có API key";
    }

    if (!dependencies.ultralytics_ready || !dependencies.tensorflow_ready) {
      showBanner(
        "Một số thư viện ML chưa có trong môi trường hiện tại. Web vẫn chạy, nhưng có thể dùng chế độ dự phòng thay cho suy luận model thật.",
        "info"
      );
    }
  } catch (error) {
    if (elements.serverStatus) {
      elements.serverStatus.textContent = "Không kết nối";
      elements.serverStatus.className = "status-pill warning";
    }
    if (elements.healthYolo) {
      elements.healthYolo.textContent = "Không rõ";
    }
    if (elements.healthCnn) {
      elements.healthCnn.textContent = "Không rõ";
    }
    if (elements.healthLabels) {
      elements.healthLabels.textContent = "Không rõ";
    }
    if (elements.healthGemini) {
      elements.healthGemini.textContent = "Không rõ";
    }
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
  if (elements.analyzeButton) {
    elements.analyzeButton.disabled = isLoading || !state.file;
    elements.analyzeButton.textContent = isLoading ? "Đang phân tích..." : "Phân tích ngay";
  }
  if (elements.serverStatus) {
    elements.serverStatus.textContent = isLoading ? "Đang xử lý" : "Sẵn sàng";
    elements.serverStatus.className = isLoading ? "status-pill warning" : "status-pill success";
  }
}

function renderPipeline(items) {
  if (!elements.pipelineList) return;
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
  state.result = result;
  if (elements.resultsCard) {
    elements.resultsCard.classList.remove("hidden");
  }
  if (elements.processingTime) {
    elements.processingTime.textContent = `${result.meta.total_duration_ms} ms`;
  }

  renderPipeline(result.pipeline);
  renderImages(result.images);
  renderClassification(result.classification);
  renderAdvice(result.llm);

  if (elements.resultsCard) {
    elements.resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderImages(images) {
  if (elements.resultOriginal) {
    elements.resultOriginal.src = images.original || "";
  }
  if (elements.resultAnnotated) {
    elements.resultAnnotated.src = images.annotated || "";
  }
  if (elements.resultCrop) {
    elements.resultCrop.src = images.cropped_leaf || "";
  }
}

function renderClassification(classification) {
  if (elements.cnnHeadline) {
    elements.cnnHeadline.textContent = `${classification.input_size.width} x ${classification.input_size.height}`;
  }
  if (elements.cnnLabel) {
    elements.cnnLabel.textContent = classification.display_label;
  }
  if (elements.cnnConfidence) {
    elements.cnnConfidence.textContent = `Độ tin cậy: ${(classification.confidence * 100).toFixed(2)}%`;
  }

  if (elements.cnnWarning) {
    if (classification.warning) {
      elements.cnnWarning.textContent = classification.warning;
      elements.cnnWarning.classList.remove("hidden");
    } else {
      elements.cnnWarning.textContent = "";
      elements.cnnWarning.classList.add("hidden");
    }
  }

  if (elements.predictionList) {
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
}

function renderAdvice(llm) {
  if (elements.llmSource) {
    const imgBadge = llm.image_analyzed
      ? ' <span style="background:#d1fae5;color:#065f46;font-size:.75rem;padding:2px 8px;border-radius:12px;font-weight:600;">📷 Đã phân tích ảnh</span>'
      : '';
    elements.llmSource.innerHTML = `Nguồn: ${llm.source} (${llm.model})${imgBadge}`;
  }
  if (elements.llmHeadline) {
    elements.llmHeadline.textContent = llm.headline || "-";
  }
  if (elements.llmSummary) {
    elements.llmSummary.textContent = llm.summary || "-";
  }
  if (elements.llmWarning) {
    elements.llmWarning.textContent = llm.warning || "Không có ghi chú thêm.";
  }

  renderList(elements.careSteps, llm.care_steps);
  renderList(elements.nextSteps, llm.next_steps);

  // Health Score
  const healthScore = llm.health_score || 50;
  const healthScoreContainer = document.getElementById("healthScoreContainer");
  if (healthScoreContainer) {
    healthScoreContainer.classList.remove("hidden");
  }
  if (elements.healthScoreBar) {
    elements.healthScoreBar.style.width = `${healthScore}%`;
    const scoreColor = healthScore >= 60 ? "#22c55e" : healthScore >= 35 ? "#f59e0b" : "#ef4444";
    elements.healthScoreBar.style.background = scoreColor;
  }
  if (elements.healthScoreValue) {
    elements.healthScoreValue.textContent = healthScore;
  }

  // Additional Info
  if (elements.plantType) {
    elements.plantType.textContent = llm.plant_type || "-";
  }
  if (elements.spreadLevel) {
    elements.spreadLevel.textContent = llm.spread_level || "-";
  }
  if (elements.economicImpact) {
    elements.economicImpact.textContent = llm.economic_impact || "-";
  }
  if (elements.treatmentSchedule) {
    renderList(elements.treatmentSchedule, llm.treatment_schedule?.map((t) => `${t.action} (sau ${t.days_later} ngày)`));
  }

  // Disease Progression
  if (elements.diseaseProgression && llm.disease_progression) {
    elements.diseaseProgression.innerHTML = Object.entries(llm.disease_progression)
      .map(([key, value]) => `<div><strong>${key}:</strong> ${escapeHtml(value)}</div>`)
      .join("");
  }

  // Visual Observations từ Gemini (chỉ hiện khi có dữ liệu)
  const obsSection = document.getElementById("visualObsSection");
  const obsList = document.getElementById("visualObservations");
  if (obsSection && obsList) {
    const obs = llm.visual_observations || [];
    if (obs.length > 0) {
      renderList(obsList, obs);
      obsSection.style.display = "";
    } else {
      obsSection.style.display = "none";
    }
  }

  // Chẩn đoán tổng hợp (final_diagnosis + cnn_agreement)
  if (elements.diagnosisBadge) {
    if (llm.final_diagnosis) {
      const displayName = llm.final_diagnosis.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
      if (elements.finalDiagnosis) elements.finalDiagnosis.textContent = displayName;
      if (elements.cnnAgreementBadge) {
        const agreementMap = {
          agree: { text: "✓ CNN xác nhận", bg: "#d1fae5", color: "#065f46" },
          disagree: { text: "~ Gemini bổ sung", bg: "#fef3c7", color: "#92400e" },
          uncertain: { text: "~ Đang đối chiếu", bg: "#f3f4f6", color: "#374151" },
        };
        const a = agreementMap[llm.cnn_agreement] || { text: llm.cnn_agreement || "-", bg: "#f3f4f6", color: "#374151" };
        elements.cnnAgreementBadge.textContent = a.text;
        elements.cnnAgreementBadge.style.background = a.bg;
        elements.cnnAgreementBadge.style.color = a.color;
      }
      elements.diagnosisBadge.style.display = "flex";
    } else {
      elements.diagnosisBadge.style.display = "none";
    }
  }

  // Bằng chứng hình ảnh từ Gemini
  if (elements.diseaseEvidenceSection && elements.diseaseEvidence) {
    if (llm.disease_evidence) {
      elements.diseaseEvidence.textContent = llm.disease_evidence;
      elements.diseaseEvidenceSection.style.display = "";
    } else {
      elements.diseaseEvidenceSection.style.display = "none";
    }
  }

  // Khuyến nghị thêm
  if (elements.recommendationsSection && elements.recommendations) {
    const recs = llm.recommendations || [];
    if (recs.length > 0) {
      renderList(elements.recommendations, recs);
      elements.recommendationsSection.style.display = "";
    } else {
      elements.recommendationsSection.style.display = "none";
    }
  }
}

function renderList(target, items) {
  if (!target) return;
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
  if (elements.messageBanner) {
    elements.messageBanner.textContent = message;
    elements.messageBanner.className = `message-banner ${type}`;
  }
}

function hideBanner() {
  if (elements.messageBanner) {
    elements.messageBanner.textContent = "";
    elements.messageBanner.className = "message-banner hidden";
  }
}

async function downloadReport() {
  if (!state.result) {
    showBanner("Chưa có kết quả phân tích để xuất báo cáo.", "error");
    return;
  }

  const btn = document.getElementById("shareReportBtn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Đang tạo báo cáo...";
  }

  try {
    const response = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result: state.result }),
    });

    if (!response.ok) {
      throw new Error(`Máy chủ trả về lỗi ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
    link.download = `leafai-report-${ts}.html`;
    link.href = url;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (error) {
    showBanner(`Không thể tải báo cáo: ${error.message}`, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd"/></svg> Tải báo cáo`;
    }
  }
}

async function openQrModal() {
  if (!state.result) {
    showBanner("Chưa có kết quả phân tích để tạo QR Code.", "error");
    return;
  }

  const modal = document.getElementById("qrModal");
  const qrLoading = document.getElementById("qrLoading");
  const qrContent = document.getElementById("qrContent");
  const qrDownloadBtn = document.getElementById("qrDownloadBtn");
  if (!modal) return;

  modal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  if (qrLoading) qrLoading.classList.remove("hidden");
  if (qrContent) qrContent.classList.add("hidden");
  if (qrDownloadBtn) qrDownloadBtn.disabled = true;
  state.qrDataUrl = "";

  try {
    const llm = state.result.llm || {};
    const cls = state.result.classification || {};
    const payload = {
      cnn_label: cls.display_label || "-",
      cnn_conf: (cls.confidence || 0) * 100,
      health_score: llm.health_score || 50,
      plant_type: llm.plant_type || "-",
      summary: llm.summary || "",
    };

    const response = await fetch("/api/qr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!data.success) throw new Error(data.error || "Không thể tạo QR Code");

    state.qrDataUrl = data.qr_data_url;

    const qrImage = document.getElementById("qrImage");
    if (qrImage) qrImage.src = data.qr_data_url;

    const qrTextEl = document.getElementById("qrText");
    if (qrTextEl) qrTextEl.textContent = data.text || "";

    if (qrLoading) qrLoading.classList.add("hidden");
    if (qrContent) qrContent.classList.remove("hidden");
    if (qrDownloadBtn) qrDownloadBtn.disabled = false;
  } catch (error) {
    if (qrLoading) qrLoading.classList.add("hidden");
    if (qrContent) {
      qrContent.innerHTML = `<p style="color:#8b2c1f;text-align:center;">${escapeHtml(error.message)}</p>`;
      qrContent.classList.remove("hidden");
    }
  }
}

function closeQrModal() {
  const modal = document.getElementById("qrModal");
  if (modal) modal.classList.add("hidden");
  document.body.style.overflow = "";
}

function downloadQrImage() {
  if (!state.qrDataUrl) return;
  const link = document.createElement("a");
  const ts = new Date().toISOString().slice(0, 10);
  link.download = `leafai-qr-${ts}.png`;
  link.href = state.qrDataUrl;
  link.click();
}

init();
