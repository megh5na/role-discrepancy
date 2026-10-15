/* Role-discrepancy demo. No framework. Numbers come from live scoring (server) or the saved results in
   service/demo_cases.json and service/thresholds.json. Scoring runs automatically after a short pause. */
const ROLES = ["system", "user", "document", "tool", "reasoning"];
const CAT = {
  benign_email_text: ["Benign email text", false],
  injected_original: ["Attack, original wording", true],
  destyled_attack: ["Attack, softened", true],
  user_pastes_document: ["Benign: user pastes a document", false],
  document_quotes_request: ["Benign: document quotes a request", false],
  control_user: ["Benign user text", false],
};
const UNRELIABLE = new Set(["tool", "reasoning"]);       // per-channel thresholds for these are degenerate
const $ = (id) => document.getElementById(id);
const S = {cond: "naive", policy: "per_channel", declared: "document", carrier: null, onlyWrong: false, attack: 0, cases: [], thr: null, seq: 0, timer: null};
const f2 = (x) => x.toFixed(2);
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

function threshold(cond, policy, declared) {
  return policy === "pooled" ? S.thr.thresholds[cond] : S.thr.thresholds_per_channel[cond][declared];
}
function verdictFor(score, cond, policy, declared) {            // true / false, or null when no reliable threshold exists
  if (policy === "per_channel" && UNRELIABLE.has(declared)) return null;
  return score >= threshold(cond, policy, declared);
}

/* ---------- options ---------- */
function tabs(host, items, getter, setter, labeler, after) {
  host.textContent = "";
  after = after || (() => { renderControls(); schedule(0); renderLists(); });
  items.forEach((v) => {
    const b = el("button", getter() === v ? "on" : "", labeler ? labeler(v) : cap(v));
    b.onclick = () => { setter(v); after(); };
    host.append(b);
  });
}
function renderControls() {
  tabs($("declared"), ROLES, () => S.declared, (v) => (S.declared = v));
  tabs($("opt-cond"), ["naive", "transplant"], () => S.cond, (v) => (S.cond = v));
  tabs($("opt-policy"), ["per_channel", "pooled"], () => S.policy, (v) => (S.policy = v), (v) => (v === "pooled" ? "Pooled" : "Per channel"));
  tabs($("opt-carrier"), [null, ...ROLES], () => S.carrier, (v) => (S.carrier = v), (v) => (v ? cap(v) : "None"));
}

/* ---------- scoring ---------- */
function schedule(ms) { clearTimeout(S.timer); S.timer = setTimeout(score, ms); }
async function health() { try { return await (await fetch("/api/health")).json(); } catch { return {}; } }
function stateText(h) { $("state").textContent = h.loading ? "Loading detector" : h.loaded ? cap(h.loaded) + " detector loaded" : "No detector loaded"; }

async function score() {
  const text = $("text").value.trim(), mine = ++S.seq;
  $("error").hidden = true;
  if (!text) { clearResult(); return; }
  const h = await health(); stateText(h);
  $("result").classList.add("busy");
  if (h.loaded !== S.cond) {                                      // do not leave the previous detector's verdict on screen
    $("result").dataset.v = "empty"; $("verdict").textContent = ""; $("verdict").className = "verdict"; $("rolebar").textContent = ""; $("legend").textContent = ""; $("figures").textContent = ""; $("wrapped").hidden = true;
    $("why").textContent = `Loading the ${S.cond} detector. This takes about 16 seconds once, then about a second per text.`;
  }
  try {
    const resp = await fetch("/api/score", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, declared: S.declared, condition: S.cond, policy: S.policy, carrier: S.carrier})});
    const data = await resp.json();
    if (mine !== S.seq) return;                                   // a newer request superseded this one
    if (!resp.ok) throw new Error(data.error || "Scoring failed");
    renderResult(data);
  } catch (e) { if (mine === S.seq) { $("error").hidden = false; $("error").textContent = e.message; } }
  finally { if (mine === S.seq) { $("result").classList.remove("busy"); stateText(await health()); } }
}

function clearResult() {
  ["verdict", "why", "legend", "figures"].forEach((i) => ($(i).textContent = ""));
  $("rolebar").textContent = ""; $("wrapped").hidden = true; $("verdict").className = "verdict";
  $("result").dataset.v = "empty";
}
function renderResult(res) {
  const r = res.bare, v = verdictFor(r.score, S.cond, S.policy, r.declared_role);
  const top = r.perceived_role, topPct = Math.round(r.role_probabilities[top] * 100);
  const ve = $("verdict");
  ve.className = "verdict";
  $("result").dataset.v = v === null ? "none" : v ? "flag" : "clear";
  ve.textContent = v === null ? "No verdict" : v ? "Flagged" : "Clear";
  $("why").textContent = top === r.declared_role
    ? `Reads as ${top} (${topPct}%), as declared.`
    : `Reads as ${top} (${topPct}%), but declared ${r.declared_role}.` + (v === null ? " This channel has no reliable threshold." : "");
  const bar = $("rolebar"); bar.textContent = ""; const legend = $("legend"); legend.textContent = "";
  ROLES.forEach((ro) => {
    const p = r.role_probabilities[ro];
    const seg = el("div", ro === top ? "top" : "", p >= 0.14 ? ro : ""); seg.style.flexBasis = Math.max(p * 100, 0.6) + "%"; seg.style.flexGrow = 0; bar.append(seg);
    legend.append(el("span", ro === r.declared_role ? "decl" : "", `${ro} ${Math.round(p * 100)}%` + (ro === r.declared_role ? " (declared)" : "")));
  });
  const where = S.policy === "pooled" ? "pooled threshold" : r.declared_role + " threshold";
  $("figures").textContent = `score ${f2(r.score)} · ${where} ${f2(r.threshold)} · ${S.cond} detector`;
  const w = $("wrapped");
  if (res.wrapped) {
    const wv = verdictFor(res.wrapped.score, S.cond, S.policy, r.declared_role), same = res.wrapped.perceived_role === top;
    w.hidden = false; w.textContent = "";
    w.append(el("div", "", `In a ${S.carrier} carrier: score ${f2(res.wrapped.score)}, ${same ? "still reads as" : "now reads as"} ${res.wrapped.perceived_role}` +
      (wv === null ? "." : wv ? ", flagged." : ", not flagged.")), el("code", "", res.wrapped_text));
  } else w.hidden = true;
  const notes = [];
  if (r.text_changed_by_delimiter_stripping) notes.push("Chat-template delimiters were removed before scoring.");
  if (res.truncation_note) notes.push("Only the first ~64 tokens are read.");
  if (notes.length) $("figures").textContent += " · " + notes.join(" ");
}

/* ---------- examples ---------- */
function outcome(c) {
  const sc = c.scores[S.cond], flagged = S.policy === "pooled" ? sc.flagged_pooled : sc.flagged_per_channel;
  if (S.policy === "per_channel" && !sc.per_channel_threshold_reliable) return ["No verdict", false];
  const attack = CAT[c.category][1];
  if (attack) return flagged ? ["Caught", false] : ["Missed", true];
  return flagged ? ["False alarm", true] : ["Correct", false];
}
function load(c) { $("text").value = c.text; S.declared = c.declared_role; autosize(); renderControls(); window.scrollTo({top: 0, behavior: "smooth"}); schedule(0); }
function renderLists() {
  if (!S.cases.length) return;
  const list = $("list"); list.textContent = "";
  const cs = S.cases.filter((c) => c.category !== "destyled_attack");
  let wrong = 0;
  cs.forEach((c) => {
    const [label, bad] = outcome(c); if (bad) wrong++;
    if (S.onlyWrong && !bad) return;
    const li = el("li", bad ? "bad" : ""), b = el("button");
    const t = el("span", "t");
    t.append(el("span", "tx", c.text.replace(/\s+/g, " ")), el("span", "c", CAT[c.category][0] + " · declared " + c.declared_role));
    b.append(t, el("span", "n", f2(c.scores[S.cond].score)), el("span", "o" + (bad ? " wrong" : ""), label));
    b.onclick = () => load(c); li.append(b); list.append(li);
  });
  $("only-wrong").textContent = S.onlyWrong ? "Show all" : `Show only the wrong ones (${wrong} of ${cs.length})`;
  renderDestyle();
}
function renderDestyle() {
  const d = S.cases.filter((c) => c.category === "destyled_attack"), groups = [];
  for (let i = 0; i < d.length; i += 3) groups.push(d.slice(i, i + 3));
  if (!groups.length) return;
  const kinds = groups.map((g, i) => `Attack ${i + 1} (${(g[0].source.match(/\((\w+),/) || [, "attack"])[1].replace("_", " ")})`);
  tabs($("attack-pick"), groups.map((_, i) => i), () => S.attack, (v) => (S.attack = v), (i) => kinds[i], () => renderDestyle());
  const host = $("levels"); host.textContent = "";
  groups[S.attack].forEach((c, k) => {
    const sc = c.scores[S.cond], v = verdictFor(sc.score, S.cond, S.policy, "document"), box = el("div", "level");
    box.append(el("div", "k", ["As written", "Softened 0.5", "Softened 1.0"][k]), el("div", "num", f2(sc.score)),
      el("div", "m" + (v ? "" : " slip"), v ? "Still flagged" : "Slips under the threshold"), el("div", "txt", c.text));
    host.append(box);
  });
}

/* ---------- init ---------- */
function autosize() { const t = $("text"); t.style.height = "auto"; t.style.height = Math.max(150, t.scrollHeight) + "px"; }
async function init() {
  renderControls();
  $("text").addEventListener("input", () => { autosize(); schedule(800); });
  $("text").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") schedule(0); });
  $("only-wrong").onclick = () => { S.onlyWrong = !S.onlyWrong; renderLists(); };
  const [cases, thr] = await Promise.all([fetch("/api/cases").then((r) => r.json()), fetch("/api/thresholds").then((r) => r.json())]);
  S.cases = cases.cases; S.thr = thr;
  renderLists(); stateText(await health());
  setInterval(async () => stateText(await health()), 5000);
}
init();
