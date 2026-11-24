import { Client } from "@gradio/client";
import "./style.css";

const spaceId = import.meta.env.VITE_GRADIO_SPACE || "01mayankk/RetinaScanAI";
const endpointValue = import.meta.env.VITE_GRADIO_API_NAME || "/run_analysis";
const endpoint = endpointValue.startsWith("/") ? endpointValue : `/${endpointValue}`;

const form = document.querySelector("#analysis-form");
const imageInput = document.querySelector("#image-input");
const analyzeButton = document.querySelector("#analyze-button");
const replaceImageButton = document.querySelector("#replace-image");
const analyzeButtonCopy = analyzeButton.querySelector(".button-copy");
const requestStatus = document.querySelector("#request-status");
const preview = document.querySelector("#image-preview");
const uploadBox = document.querySelector("#upload-box");
const selectedImage = document.querySelector("#selected-image");
const fileNameOutput = document.querySelector("#file-name");
const fileDetailsOutput = document.querySelector("#file-details");
const resultsEmpty = document.querySelector("#results-empty");
const resultContent = document.querySelector("#result-content");
const resultsSubtitle = document.querySelector("#results-subtitle");
const diagnosisOutput = document.querySelector("#diagnosis-output");
const riskBadge = document.querySelector("#risk-badge");
const confidenceStatusOutput = document.querySelector("#confidence-status-output");
const confidenceOutput = document.querySelector("#confidence-output");
const confidenceBar = document.querySelector("#confidence-bar");
const runnerUpOutput = document.querySelector("#runner-up-output");
const recommendationOutput = document.querySelector("#recommendation-output");
const reportOutput = document.querySelector("#report-output");
const originalOutput = document.querySelector("#original-output");
const heatmapOutput = document.querySelector("#heatmap-output");
const heatmapUnavailable = document.querySelector("#heatmap-unavailable");
const pdfDownload = document.querySelector("#pdf-download");
const pdfUnavailable = document.querySelector("#pdf-unavailable");
const copySummaryButton = document.querySelector("#copy-summary");
const confidenceTrack = document.querySelector(".progress-track");
const originalTab = document.querySelector("#original-tab");
const heatmapTab = document.querySelector("#heatmap-tab");
const originalPanel = document.querySelector("#original-panel");
const heatmapPanel = document.querySelector("#heatmap-panel");

let clientPromise;
let previewUrl;
let selectedFile;
let currentSummary = "";

function setStatus(message, state = "") {
  requestStatus.textContent = message;
  requestStatus.dataset.state = state;
}

function updateSpaceStatus(status) {
  if (status?.status === "sleeping" || status?.detail === "SLEEPING") {
    setStatus("Waking the analysis service. This can take a few minutes.", "pending");
  } else if (
    status?.status === "building" ||
    status?.status === "starting" ||
    status?.detail === "RUNNING_BUILDING" ||
    status?.detail === "APP_STARTING"
  ) {
    setStatus("The analysis service is starting up. Please wait.", "pending");
  } else if (
    status?.status === "error" ||
    status?.status === "space_error" ||
    status?.status === "paused" ||
    status?.status === "stopped"
  ) {
    setStatus(`The analysis service is unavailable: ${status.message}`, "error");
  }
}

function connectToSpace() {
  if (!clientPromise) {
    clientPromise = Client.connect(spaceId, {
      events: ["data", "status"],
      status_callback: updateSpaceStatus,
    }).catch((error) => {
      clientPromise = undefined;
      throw error;
    });
  }

  return clientPromise;
}

function getFileUrl(value, label) {
  if (typeof value === "string" && value.length > 0) {
    return value;
  }

  if (value && typeof value === "object" && typeof value.url === "string" && value.url.length > 0) {
    return value.url;
  }

  throw new Error(`The Gradio API did not return a usable ${label} URL.`);
}

function updateQueueStatus(status) {
  if (status.stage === "pending") {
    if (Number.isInteger(status.position)) {
      setStatus(`Queued for analysis · position ${status.position + 1}.`, "pending");
    } else {
      setStatus("Queued for analysis. Waiting for the service.", "pending");
    }
  } else if (status.stage === "generating") {
    setStatus("Analyzing the retinal image and generating your report.", "pending");
  } else if (status.stage === "complete") {
    setStatus("Analysis complete. Preparing your results.", "pending");
  } else if (status.stage === "error") {
    throw new Error(status.message || "The Gradio request failed.");
  }
}

function parseReport(report) {
  const fields = ["Prediction", "Confidence", "Second Likely Class", "Risk Level", "Confidence Status", "Medical Recommendation"];
  const parsed = {};

  for (let index = 0; index < fields.length; index += 1) {
    const field = fields[index];
    const nextField = fields[index + 1];
    const start = report.search(new RegExp(`^\\s*${field}:\\s*`, "im"));
    if (start === -1) {
      continue;
    }

    const valueStart = report.indexOf(":", start) + 1;
    let valueEnd = report.length;
    if (nextField) {
      const nextMatch = new RegExp(`^\\s*${nextField}:\\s*`, "im").exec(report.slice(valueStart));
      if (nextMatch) {
        valueEnd = valueStart + nextMatch.index;
      }
    } else {
      const disclaimerIndex = report.slice(valueStart).search(/^\s*DISCLAIMER\s*$/im);
      if (disclaimerIndex !== -1) {
        valueEnd = valueStart + disclaimerIndex;
      }
    }

    parsed[field] = report
      .slice(valueStart, valueEnd)
      .split(/\r?\n/)
      .filter((line) => !/^\s*=+\s*$/.test(line))
      .join(" ")
      .trim();
  }

  if (!parsed.Prediction || !parsed.Confidence || !parsed["Risk Level"]) {
    throw new Error("The analysis service returned a report missing required screening fields.");
  }

  return parsed;
}

function getRiskTone(riskLevel) {
  const risk = riskLevel.toLowerCase();
  if (/critical|high|severe|proliferative/.test(risk)) {
    return "severe";
  }
  if (/moderate|mild/.test(risk)) {
    return "moderate";
  }
  return "low";
}

function formatConfidenceStatus(status) {
  if (/low confidence/i.test(status)) {
    return "Low confidence";
  }
  if (/acceptable/i.test(status)) {
    return "Confidence acceptable";
  }
  return status || "Not provided";
}

function formatConfidence(value) {
  const match = value.match(/[\d.]+/);
  if (!match) {
    throw new Error("The analysis service returned an invalid confidence value.");
  }
  const confidence = Number(match[0]);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 100) {
    throw new Error("The analysis service returned an out-of-range confidence value.");
  }
  return confidence;
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function renderStructuredReport(report) {
  const fields = parseReport(report);
  const confidence = formatConfidence(fields.Confidence);
  const runnerUp = fields["Second Likely Class"] || "Not provided";
  const confidenceStatus = formatConfidenceStatus(fields["Confidence Status"]);
  const riskTone = getRiskTone(fields["Risk Level"]);

  diagnosisOutput.textContent = fields.Prediction;
  riskBadge.textContent = fields["Risk Level"];
  riskBadge.dataset.tone = riskTone;
  confidenceStatusOutput.textContent = confidenceStatus;
  confidenceStatusOutput.parentElement.dataset.tone =
    /low confidence/i.test(confidenceStatus) ? "caution" : "positive";
  confidenceOutput.textContent = `${confidence.toFixed(2)}%`;
  confidenceBar.style.width = "0%";
  confidenceTrack.setAttribute("aria-valuenow", "0");
  runnerUpOutput.textContent = runnerUp;
  recommendationOutput.textContent = fields["Medical Recommendation"] || "No recommendation was provided.";
  reportOutput.textContent = report;
  currentSummary = [
    `Prediction: ${fields.Prediction}`,
    `Risk level: ${fields["Risk Level"]}`,
    `Confidence: ${confidence.toFixed(2)}%`,
    `Second likely class: ${runnerUp}`,
    `Confidence status: ${confidenceStatus}`,
    `Medical recommendation: ${fields["Medical Recommendation"] || "Not provided"}`,
    "For research and triage support only; confirm with a licensed ophthalmologist.",
  ].join("\n");
  return confidence;
}

function renderResults(data) {
  if (!Array.isArray(data) || data.length !== 3) {
    throw new Error("The Gradio API returned an unexpected result. Expected a report, heatmap, and PDF.");
  }

  const [report, heatmap, pdf] = data;
  if (typeof report !== "string") {
    throw new Error("The Gradio API report output was not text.");
  }

  const confidence = renderStructuredReport(report);
  resultsEmpty.hidden = true;
  resultContent.hidden = false;
  resultsSubtitle.textContent = "Review the model output and image attribution below.";
  activateImageTab("original");
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      confidenceBar.style.width = `${confidence}%`;
      confidenceTrack.setAttribute("aria-valuenow", confidence.toFixed(2));
    });
  });

  originalOutput.src = previewUrl;
  originalOutput.alt = selectedFile ? `Original fundus image: ${selectedFile.name}` : "Original uploaded fundus image";

  const missingOutputs = [];
  heatmapOutput.hidden = true;
  heatmapUnavailable.hidden = true;
  pdfDownload.hidden = true;
  pdfUnavailable.hidden = true;

  if (heatmap == null) {
    heatmapUnavailable.hidden = false;
    missingOutputs.push("Grad-CAM heatmap");
  } else {
    heatmapOutput.src = getFileUrl(heatmap, "Grad-CAM heatmap");
    heatmapOutput.hidden = false;
  }

  if (pdf == null) {
    pdfUnavailable.hidden = false;
    missingOutputs.push("PDF report");
  } else {
    pdfDownload.href = getFileUrl(pdf, "PDF report");
    pdfDownload.download =
      typeof pdf === "object" && typeof pdf.orig_name === "string"
        ? pdf.orig_name
        : "RetinaScan-AI-report.pdf";
    pdfDownload.hidden = false;
  }

  return missingOutputs;
}

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files || [];
  const allowedMimeTypes = ["image/jpeg", "image/png"];
  const allowedExtension = /\.(jpe?g|png)$/i.test(file?.name || "");
  const allowedFileType = file && (
    allowedMimeTypes.includes(file.type) ||
    (!file.type && allowedExtension)
  );
  if (file && !allowedFileType) {
    imageInput.value = "";
    setStatus("Choose a JPG or PNG retinal image.", "error");
    return;
  }

  selectedFile = file;
  analyzeButton.disabled = !file;
  resultsEmpty.hidden = false;
  resultContent.hidden = true;
  resultsSubtitle.textContent = "A structured summary will appear after analysis.";
  currentSummary = "";
  confidenceBar.style.width = "0%";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
  }

  if (file) {
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    uploadBox.hidden = true;
    selectedImage.hidden = false;
    selectedImage.classList.remove("is-scanning", "is-dragging");
    fileNameOutput.textContent = file.name;
    fileDetailsOutput.textContent = `${formatFileSize(file.size)} · Loading dimensions…`;
    const dimensionsImage = new Image();
    dimensionsImage.onload = () => {
      if (previewUrl === dimensionsImage.src) {
        fileDetailsOutput.textContent = `${dimensionsImage.naturalWidth} × ${dimensionsImage.naturalHeight} px · ${formatFileSize(file.size)}`;
      }
    };
    dimensionsImage.src = previewUrl;
    setStatus("Image ready. Start the analysis when you are ready.");
  } else {
    preview.removeAttribute("src");
    selectedImage.hidden = true;
    uploadBox.hidden = false;
    selectedImage.classList.remove("is-scanning", "is-dragging");
    if (requestStatus.dataset.state !== "error") {
      setStatus("Select an image to begin.");
    }
  }
});

replaceImageButton.addEventListener("click", () => imageInput.click());

function handleFileDrop(event) {
  event.preventDefault();
  selectedImage.classList.remove("is-dragging");
  uploadBox.classList.remove("is-dragging");

  const [file] = event.dataTransfer?.files || [];
  if (!file) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  imageInput.dispatchEvent(new Event("change", { bubbles: true }));
}

for (const dropTarget of [uploadBox, selectedImage]) {
  dropTarget.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropTarget.classList.add("is-dragging");
  });
  dropTarget.addEventListener("dragleave", (event) => {
    if (!dropTarget.contains(event.relatedTarget)) {
      dropTarget.classList.remove("is-dragging");
    }
  });
  dropTarget.addEventListener("drop", handleFileDrop);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const image = selectedFile;
  if (!image) {
    setStatus("Choose a JPG or PNG retinal image before starting.", "error");
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.classList.add("is-loading");
  analyzeButtonCopy.textContent = "Analyzing Retinal Features...";
  selectedImage.classList.add("is-scanning");
  setStatus("Analyzing Retinal Features...", "pending");

  try {
    const client = await connectToSpace();
    const submission = client.submit(endpoint, { image });
    let responseData;

    for await (const event of submission) {
      if (event.type === "status") {
        updateQueueStatus(event);
      } else if (event.type === "data") {
        responseData = event.data;
      }
    }

    if (!responseData) {
      throw new Error("The Gradio API completed without returning results.");
    }

    const missingOutputs = renderResults(responseData);
    if (missingOutputs.length > 0) {
      setStatus(`Report returned, but unavailable output: ${missingOutputs.join(", ")}.`, "error");
    } else {
      setStatus("Analysis complete.", "success");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Analysis failed: ${message}`, "error");
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.classList.remove("is-loading");
    analyzeButtonCopy.textContent = "Run Retinal Analysis";
    selectedImage.classList.remove("is-scanning");
  }
});

function activateImageTab(view) {
  const showOriginal = view === "original";
  originalTab.classList.toggle("is-active", showOriginal);
  originalTab.setAttribute("aria-selected", String(showOriginal));
  originalPanel.hidden = !showOriginal;
  originalPanel.classList.toggle("is-active", showOriginal);
  heatmapTab.classList.toggle("is-active", !showOriginal);
  heatmapTab.setAttribute("aria-selected", String(!showOriginal));
  heatmapPanel.hidden = showOriginal;
  heatmapPanel.classList.toggle("is-active", !showOriginal);
}

originalTab.addEventListener("click", () => activateImageTab("original"));
heatmapTab.addEventListener("click", () => activateImageTab("heatmap"));

copySummaryButton.addEventListener("click", async () => {
  if (!currentSummary) {
    return;
  }

  try {
    await navigator.clipboard.writeText(currentSummary);
    copySummaryButton.textContent = "Copied";
    window.setTimeout(() => {
      copySummaryButton.textContent = "Copy Summary";
    }, 1800);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Could not copy the screening summary: ${message}`, "error");
  }
});

document.querySelectorAll("[data-dialog]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector(`#${button.dataset.dialog}`).showModal();
  });
});

document.querySelectorAll(".dialog-close").forEach((button) => {
  button.addEventListener("click", () => button.closest("dialog").close());
});

document.querySelectorAll("dialog").forEach((dialog) => {
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      dialog.close();
    }
  });
});
