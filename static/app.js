let currentDocId = null;

// --- Upload ---
const dropzone = document.getElementById("dropzone");
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("border-indigo-500");
});
dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("border-indigo-500"),
);
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("border-indigo-500");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  const status = document.getElementById("uploadStatus");
  status.classList.remove("hidden");
  status.textContent = `Uploading ${file.name}...`;
  status.className = "mt-3 text-sm text-indigo-600";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/ingest", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      status.textContent = `Error: ${data.error || res.statusText}`;
      status.className = "mt-3 text-sm text-red-600";
      return;
    }

    currentDocId = data.doc_id;
    status.className =
      "mt-3 text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-2";
    status.textContent = `✓ ${data.filename} — ${data.flat_chunks} flat chunks, ${data.table_chunks} table chunks, ${data.section_chunks} section chunks`;
  } catch (err) {
    status.textContent = `Upload failed: ${err.message}`;
    status.className = "mt-3 text-sm text-red-600";
  }
}

// --- Query ---
function setQ(btn) {
  document.getElementById("questionInput").value = btn.textContent;
}

async function runQuery() {
  if (!currentDocId) {
    alert("Please upload a document first.");
    return;
  }
  const question = document.getElementById("questionInput").value.trim();
  if (!question) return;

  const btn = document.getElementById("queryBtn");
  const queryStatus = document.getElementById("queryStatus");
  btn.disabled = true;
  queryStatus.classList.remove("hidden");
  document.getElementById("resultsSection").classList.add("hidden");

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: currentDocId, question }),
    });
    const data = await res.json();

    if (!res.ok) {
      alert(`Error: ${data.error || res.statusText}`);
      return;
    }

    renderResults(data);
    document.getElementById("resultsSection").classList.remove("hidden");
  } catch (err) {
    alert(`Query failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    queryStatus.classList.add("hidden");
  }
}

// --- Render ---
function renderResults(data) {
  document.getElementById("flatCol").innerHTML = renderPipeline(
    data.flat_rag,
    "Standard RAG",
    "bg-orange-50 border-orange-200",
  );
  document.getElementById("structCol").innerHTML = renderPipeline(
    data.structured_rag,
    "Structure-Aware RAG",
    "bg-indigo-50 border-indigo-200",
  );
}

function renderPipeline(pipeline, label, headerClass) {
  const conf = pipeline.confidence;
  const pct = Math.round(conf.overall_score * 100);
  const badgeColor =
    pct >= 80
      ? "bg-green-100 text-green-800"
      : pct >= 50
        ? "bg-yellow-100 text-yellow-800"
        : "bg-red-100 text-red-800";

  const dimensionsHtml = conf.dimensions
    .map(
      (d) => `
    <div class="mb-2">
      <div class="flex justify-between text-xs mb-1">
        <span class="text-gray-600 font-medium">${d.name}</span>
        <span class="font-semibold">${Math.round(d.score * 100)}%</span>
      </div>
      <div class="w-full bg-gray-200 rounded-full h-2" title="${escHtml(d.explanation)}">
        <div class="h-2 rounded-full ${scoreColor(d.score)}" style="width:${Math.round(d.score * 100)}%"></div>
      </div>
      <p class="text-xs text-gray-400 mt-0.5">${escHtml(d.explanation)}</p>
    </div>
  `,
    )
    .join("");

  const sourcesHtml = pipeline.retrieved_chunks
    .map(
      (c, i) => `
    <div class="text-xs text-gray-600 border rounded p-2 mb-2 bg-white">
      <span class="inline-block px-1.5 py-0.5 rounded text-xs font-medium mr-1 ${chunkTypeBadge(c.chunk_type)}">${c.chunk_type}</span>
      <span class="text-gray-400">p.${c.page_number}</span>
      <p class="mt-1 text-gray-700 line-clamp-3">${escHtml(c.text.substring(0, 300))}${c.text.length > 300 ? "…" : ""}</p>
    </div>
  `,
    )
    .join("");

  const sourceCount = pipeline.retrieved_chunks.length;
  const toggleId = `sources-${pipeline.pipeline}`;

  return `
    <div class="bg-white rounded-xl border overflow-hidden">
      <!-- Header -->
      <div class="px-5 py-3 border-b ${headerClass} flex items-center justify-between">
        <span class="font-semibold text-gray-800">${label}</span>
        <span class="text-sm font-bold px-2 py-1 rounded-full ${badgeColor}">${pct}% confidence</span>
      </div>

      <!-- Answer -->
      <div class="px-5 py-4">
        <p class="text-sm text-gray-800 leading-relaxed">${escHtml(pipeline.answer)}</p>
      </div>

      <!-- Sources toggle -->
      <div class="px-5 pb-3">
        <button onclick="toggleSources('${toggleId}')"
          class="text-xs text-indigo-600 hover:underline">
          Show ${sourceCount} source chunk${sourceCount !== 1 ? "s" : ""}
        </button>
        <div id="${toggleId}" class="hidden mt-2">
          ${sourcesHtml}
        </div>
      </div>

      <!-- Confidence scorecard -->
      <div class="px-5 pb-5 border-t pt-4">
        <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Confidence Scorecard</p>
        ${dimensionsHtml}
        ${
          conf.recommendation
            ? `
          <div class="mt-3 text-xs bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-amber-800">
            <span class="font-semibold">Recommendation:</span> ${escHtml(conf.recommendation)}
          </div>
        `
            : ""
        }
      </div>
    </div>
  `;
}

function toggleSources(id) {
  const el = document.getElementById(id);
  const btn = el.previousElementSibling;
  if (el.classList.contains("hidden")) {
    el.classList.remove("hidden");
    btn.textContent = btn.textContent.replace("Show", "Hide");
  } else {
    el.classList.add("hidden");
    btn.textContent = btn.textContent.replace("Hide", "Show");
  }
}

function scoreColor(score) {
  if (score >= 0.8) return "bg-green-500";
  if (score >= 0.5) return "bg-yellow-400";
  return "bg-red-400";
}

function chunkTypeBadge(type) {
  if (type === "table") return "bg-purple-100 text-purple-700";
  if (type === "section") return "bg-blue-100 text-blue-700";
  return "bg-gray-100 text-gray-600";
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
