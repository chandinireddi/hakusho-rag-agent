"use strict";

const EXAMPLES = [
  "COP15第二部はどこで開催されましたか？",
  "ひきこもり状態の期間はどのくらい長期化していますか？",
  "Where is Rapidus building its next-generation chip plant?",
];

const STATUS = {
  answered:              { label: "出典で裏付け済み", cls: "ok" },
  partially_supported:   { label: "一部未検証の記述あり", cls: "warn" },
  insufficient_evidence: { label: "根拠不足のため回答不可", cls: "none" },
};

const $ = (id) => document.getElementById(id);
const form = $("ask-form"), input = $("question"), btn = $("ask-btn");
let busy = false;
let threadId = null; // set after the first answer → follow-up questions share the conversation

/* ---------- helpers ---------- */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function showError(message) {
  const box = $("form-error");
  box.textContent = message;
  box.hidden = !message;
}

/** Answer text → DOM, turning "[2]" into a link that scrolls to source #2 (no innerHTML). */
function renderAnswer(text, refs) {
  const p = el("p", { class: "answer" });
  for (const part of text.split(/(\[\d+\])/g)) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m && refs.has(Number(m[1]))) {
      p.append(el("a", { class: "cite", href: `#src-${m[1]}`, "aria-label": `出典${m[1]}へ` }, m[1]));
    } else {
      p.append(part);
    }
  }
  return p;
}

/** Answer inspection sheet (検品表): one row per check, pass/fail is shown as text, not colour only. */
function renderChecks(checks) {
  const passed = checks.filter((c) => c.passed).length;
  return el("section", { class: "inspection", "aria-label": "回答の検品結果" },
    el("h3", {}, `回答の検品結果 ${passed} / ${checks.length} 項目合格`),
    el("ul", { class: "checks" }, checks.map((c) =>
      el("li", { class: c.passed ? "pass" : "fail" },
        el("span", { class: "check-icon", "aria-hidden": "true" }, c.passed ? "✓" : "!"),
        el("span", { class: "check-name" }, c.name),
        el("span", { class: "check-result" }, c.passed ? "合格" : "要確認"),
        el("span", { class: "check-detail" }, c.detail)))));
}

/* ---------- rendering ---------- */
function addTraceStep(step) {
  $("trace").hidden = false;
  $("trace-list").append(
    el("li", {}, el("span", { class: "step-label" }, step.label),
         el("span", { class: "step-detail" }, step.detail),
         el("span", { class: "step-ms" }, `${step.ms} ms`)),
  );
}

function renderResult(data) {
  const status = STATUS[data.status];
  const refs = new Set(data.citations.map((c) => c.ref));
  const copyBtn = el("button", { type: "button", class: "ghost", onclick: async () => {
    try {
      await navigator.clipboard.writeText(data.answer);
      copyBtn.textContent = "コピーしました";
    } catch {
      copyBtn.textContent = "コピーできませんでした";
    }
    setTimeout(() => (copyBtn.textContent = "回答をコピー"), 1500);
  } }, "回答をコピー");

  const card = el("article", { class: "card" },
    el("div", { class: "card-head" },
      el("span", { class: `badge ${status.cls}` }, status.label),
      el("span", { class: "meta" },
        `${(data.latency_ms / 1000).toFixed(1)} 秒 · ${data.total_tokens.toLocaleString()} tokens · ${data.model}`)),
  );
  if (data.standalone_question && data.standalone_question !== data.question) {
    card.append(el("p", { class: "queries" }, `前の会話を踏まえて「${data.standalone_question}」と解釈しました`));
  }
  card.append(renderAnswer(data.answer, refs));
  if (data.checks.length) card.append(renderChecks(data.checks));

  if (data.unsupported_claims.length) {
    card.append(el("div", { class: "msg warn" },
      el("strong", {}, "出典で確認できなかった記述："),
      el("ul", {}, data.unsupported_claims.map((c) => el("li", {}, c)))));
  }

  if (data.numeric_issues.length) {
    card.append(el("div", { class: "msg warn" },
      el("strong", {}, "数値照合で出典と一致しなかった記述（PDFでご確認ください）："),
      el("ul", {}, data.numeric_issues.map((c) => el("li", {}, c)))));
  }

  if (data.citations.length) {
    card.append(el("h3", {}, "出典"), el("ol", { class: "sources" }, data.citations.map((c) =>
      el("li", { id: `src-${c.ref}` },
        el("div", { class: "src-head" },
          el("span", { class: "src-ref" }, c.ref),
          el("span", { class: "src-title" }, c.title),
          el("span", { class: "src-page" }, `${c.ministry} · p.${c.page}`),
          el("a", { href: c.url, target: "_blank", rel: "noopener" }, "PDFを開く ↗")),
        el("blockquote", {}, c.snippet)))));
  }

  if (data.search_queries.length) {
    card.append(el("p", { class: "queries" }, `検索クエリ：${data.search_queries.join(" ／ ")}`));
  }
  if (data.status !== "insufficient_evidence") card.append(copyBtn);

  threadId = data.thread_id;
  $("new-thread").hidden = false;

  const result = $("result");
  result.replaceChildren(card);
  result.hidden = false;
}

/* ---------- networking (POST + Server-Sent Events) ---------- */
async function ask(question) {
  const docIds = [...document.querySelectorAll("#doc-filters input:checked")].map((i) => i.value);
  const res = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, doc_ids: docIds.length ? docIds : null, thread_id: threadId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(res.status === 422 ? "質問は2〜500文字で入力してください。" : err.detail || "サーバーエラーが発生しました。");
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] ?? "{}");
      if (event === "step") addTraceStep(data);
      else if (event === "result") renderResult(data);
      else if (event === "error") throw new Error(data.detail);
    }
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  const question = input.value.trim();
  if (question.length < 2) return showError("質問を2文字以上で入力してください。");

  busy = true;
  btn.disabled = true;
  btn.textContent = "調査中…";
  showError("");
  $("trace-list").replaceChildren();
  $("result").hidden = true;
  try {
    await ask(question);
  } catch (err) {
    showError(err.message);
  } finally {
    busy = false;
    btn.disabled = false;
    btn.textContent = "質問する";
  }
});

$("new-thread").addEventListener("click", () => {
  threadId = null;
  $("new-thread").hidden = true;
  $("trace").hidden = true;
  $("result").hidden = true;
  input.value = "";
  input.dispatchEvent(new Event("input"));
  input.focus();
});

input.addEventListener("input", () => ($("char-count").textContent = `${input.value.length} / 500`));
input.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") form.requestSubmit();
});

/* ---------- init ---------- */
$("examples").append(...EXAMPLES.map((q) =>
  el("button", { type: "button", class: "example", onclick: () => {
    input.value = q;
    input.dispatchEvent(new Event("input"));
    form.requestSubmit();
  } }, q)));

fetch("/api/documents").then((r) => r.json()).then((docs) => {
  $("doc-filters").append(...docs.map((d) =>
    el("label", { class: "chip", title: d.title },
      el("input", { type: "checkbox", value: d.doc_id }), `${d.ministry}`)));
}).catch(() => {});

fetch("/api/health").then((r) => r.json()).then((h) => {
  $("health").textContent =
    `${h.documents} 文書 · ${h.chunks} チャンク · ${h.dense_retrieval ? "Hybrid" : "BM25"} · v${h.version}`;
}).catch(() => ($("health").textContent = "サーバーに接続できません"));