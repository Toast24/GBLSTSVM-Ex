// Midsem progress deck. Run: node build_deck.js out.pptx  (needs SKILL_DIR and pptxgenjs on NODE_PATH)
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");
const FIG = path.join(__dirname, "figures");

const THEME = {
  name: "GBLSTSVM Midsem",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "14324A", lt2: "EEF3F6",
    accent1: "1F6F8B", accent2: "E07A3F", accent3: "5C9E74", accent4: "B84A4A",
    accent5: "7A8B99", accent6: "C9A227", hlink: "1F6F8B", folHlink: "5C6B78",
  },
};
const HEX = { teal: "1F6F8B", orange: "E07A3F", green: "5C9E74", red: "B84A4A", gray: "7A8B99", navy: "14324A", ink: "1B2430", grid: "E3E9EE", muted: "5C6B78" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "GBLSTSVM Extensions: Midsem Progress Report";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.6, fontSize: 40, bold: true, color: C.background1, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 2.2, fontSize: 18, color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 1.05, w: 12.1, h: 0.4, fontSize: 14, italic: true, color: C.accent1, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5 },
});

let section = "";
function sec(t, sub) {
  section = t; pres.addSection({ title: t });
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
  s.addText(t, { placeholder: "title" });
  if (sub) s.addText(sub, { placeholder: "body" });
  return s;
}
function content(title, kicker) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (kicker) s.addText(kicker, { placeholder: "kicker" });
  return s;
}
const T = (o) => Object.assign({ isTextBox: true, margin: 0, color: C.text1, valign: "top" }, o);
function bullets(items, size) {
  return items.map((t, i) => {
    const o = { bullet: true, breakLine: i < items.length - 1 };
    if (size) o.fontSize = size;
    if (Array.isArray(t)) {
      const tail = { breakLine: o.breakLine }; if (size) tail.fontSize = size;
      return [{ text: t[0], options: Object.assign({}, o, { bold: true, breakLine: false }) }, { text: t[1], options: tail }];
    }
    return [{ text: t, options: o }];
  }).flat();
}
function card(s, x, y, w, h, head, body, color, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.background2 }, line: { color: opts.fill || C.background2 }, objectName: "card " + head });
  s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.24, w: 0.22, h: 0.22, fill: { color }, line: { color } });
  s.addText(head, T({ x: x + 0.55, y: y + 0.12, w: w - 0.7, h: 0.46, fontSize: opts.hs || 16, bold: true, color: C.text2, valign: "middle" }));
  s.addText(body, T({ x: x + 0.22, y: y + 0.66, w: w - 0.44, h: h - 0.8, fontSize: opts.bs || 14, paraSpaceAfter: 4 }));
}
function stat(s, x, y, w, num, label, color, size = 40) {
  s.addText(num, T({ x, y, w, h: 0.8, fontSize: size, bold: true, color, fontFace: THEME.headFontFace }));
  s.addText(label, T({ x, y: y + 0.82, w, h: 0.9, fontSize: 13 }));
}
function verdict(s, x, y, w, text, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.62, rectRadius: 0.1, fill: { color }, line: { color }, objectName: "verdict" });
  s.addText(text, T({ x: x + 0.2, y, w: w - 0.4, h: 0.62, fontSize: 15, bold: true, color: C.background1, valign: "middle" }));
}
function box(s, x, y, w, h, text, fill, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: "box " + String(text).slice(0, 20) });
  s.addText(text, T({ x: x + 0.1, y, w: w - 0.2, h, fontSize: opts.fs || 13, color: opts.color || C.background1, bold: opts.bold !== false, align: "center", valign: "middle" }));
}
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, flipV: y2 < y1, line: { color: HEX.gray, width: 1.75, endArrowType: "triangle" } });
}
const hdr = { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13 };
function table(s, rows, x, y, w, colW, hl = [], fs = 13) {
  const data = rows.map((r, i) => r.map((c) => ({ text: String(c), options: i === 0 ? Object.assign({}, hdr, { fontSize: fs }) : { fontSize: fs, color: C.text1, fill: { color: hl.includes(i) ? "FBE9DD" : (i % 2 ? "FFFFFF" : "EEF3F6") } } })));
  s.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: "D5DEE5" }, margin: 0.05 });
}
function chartOpts(title, extra) {
  return Object.assign({
    showTitle: true, title, titleFontSize: 14, titleColor: HEX.navy, titleFontFace: "+mn-lt",
    showValue: true, dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.ink,
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, legendFontFace: "+mn-lt", legendFontSize: 11,
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  }, extra);
}
function note(s, text) { s.addText(text, T({ x: 0.6, y: 6.75, w: 11.5, h: 0.35, fontSize: 11, color: C.accent5, italic: true })); }

// =====================================================================
// INTRO
// =====================================================================
section = "Introduction"; pres.addSection({ title: section });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
s.addText("Extending Granular-Ball LSTSVM", { placeholder: "title" });
s.addText("Midsem progress report\nFrom heuristic fixes to a sufficient-statistic account of granular balls\nBharatSOP research project · October 2026", { placeholder: "body" });

s = content("The story in one slide", "Eight sessions, each a hypothesis that was tested and either kept or dropped");
const story = [
  ["1–2", "Reweight the balls", "Class, reliability, margin and overlap weights. Only class weighting helped, and only a little.", HEX.orange],
  ["3", "Use the ball graph", "Topology weights help on dense graphs and fail badly on sparse ones.", HEX.orange],
  ["4", "Look upstream", "Granulation deletes minority samples; a split-geometry signal predicts it.", HEX.teal],
  ["5–7", "Fix granulation", "Gates, entropy schedulers and compression. Gains came from compression, not ball structure.", HEX.teal],
  ["A3–A7", "Ask what balls throw away", "LSTSVM needs only moments, so balls that store them are exact.", HEX.green],
  ["8", "Check and quantify", "Center-only error is directional and measurable; deletion was hiding in earlier numbers.", HEX.green],
];
story.forEach((r, i) => {
  const y = 1.6 + i * 0.85;
  box(s, 0.6, y, 1.1, 0.68, r[0], r[3], { fs: 15 });
  s.addText([{ text: r[1] + "   ", options: { bold: true, color: C.text2 } }, { text: r[2] }], T({ x: 1.9, y, w: 10.8, h: 0.68, fontSize: 15, valign: "middle" }));
});

s = content("Agenda");
const agenda = ["Background: SVM → twin SVM → LSTSVM → granular balls", "How we run experiments", "Sessions 1–2: weighting extensions on real data", "Session 3: topology-supported weights", "Session 4: granulation diagnostics", "Sessions 5–7: fixing granulation and compression", "Bridge: re-baselining and the sufficient-statistic view (A3–A7)", "Session 8: angle gate, clean geometry, held-out and kernel results", "Where we stand and the plan for the second half"];
agenda.forEach((t, i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 1.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.85, h: 1.5, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "agenda " + i });
  s.addText(String(i + 1).padStart(2, "0"), T({ x: x + 0.2, y: y + 0.12, w: 1, h: 0.5, fontSize: 22, bold: true, color: C.accent1, fontFace: THEME.headFontFace }));
  s.addText(t, T({ x: x + 0.2, y: y + 0.62, w: 3.45, h: 0.8, fontSize: 14 }));
});

// =====================================================================
// BACKGROUND
// =====================================================================
sec("Background", "What GBLSTSVM is and why it was worth extending");

s = content("From SVM to the least-squares twin SVM", "LSTSVM replaces one large QP with two small linear systems");
// diagram: two planes
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.6, w: 5.8, h: 4.9, fill: { color: "F7FAFB" }, line: { color: "E3E9EE" }, objectName: "twin plane canvas" });
const ptsA = [[1.2, 2.3], [1.6, 2.9], [1.0, 3.4], [2.0, 2.4], [1.5, 3.9], [2.3, 3.2], [1.9, 4.4], [1.1, 4.6]];
const ptsB = [[4.3, 3.6], [4.9, 4.1], [4.6, 4.8], [5.4, 3.9], [4.1, 5.3], [5.2, 5.1], [5.7, 4.6], [4.7, 5.8]];
ptsA.forEach((p) => s.addShape(pres.shapes.OVAL, { x: p[0], y: p[1], w: 0.16, h: 0.16, fill: { color: HEX.teal }, line: { color: HEX.teal } }));
ptsB.forEach((p) => s.addShape(pres.shapes.OVAL, { x: p[0], y: p[1], w: 0.16, h: 0.16, fill: { color: HEX.orange }, line: { color: HEX.orange } }));
s.addShape(pres.shapes.LINE, { x: 0.9, y: 1.9, w: 1.7, h: 4.3, line: { color: HEX.teal, width: 2.5 } });
s.addShape(pres.shapes.LINE, { x: 3.9, y: 2.9, w: 1.9, h: 3.4, line: { color: HEX.orange, width: 2.5 } });
s.addText("plane 1: close to class A,\nfar from class B", T({ x: 2.4, y: 1.75, w: 2.6, h: 0.6, fontSize: 12, color: HEX.teal }));
s.addText("plane 2: close to B, far from A", T({ x: 3.2, y: 6.05, w: 3.1, h: 0.35, fontSize: 12, color: HEX.orange }));
s.addText([
  { text: "SVM", options: { bold: true, breakLine: true } },
  { text: "One separating hyperplane with maximum margin, found by a quadratic program over all n samples.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Twin SVM (TSVM)", options: { bold: true, breakLine: true } },
  { text: "Two non-parallel planes, one per class. Each plane hugs its own class and stays at least unit distance from the other. A test point gets the class of the nearer plane.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Least-squares TSVM (LSTSVM)", options: { bold: true, breakLine: true } },
  { text: "Replaces the inequality constraints with squared errors. Each plane is then the solution of a single (d+1)×(d+1) linear system:", options: { breakLine: true } },
  { text: "[w₁; b₁] = −(GᵀG + HᵀH/c₁)⁻¹ Gᵀe", options: { italic: true, color: C.accent1 } },
], T({ x: 6.8, y: 1.6, w: 5.9, h: 4.9, fontSize: 15 }));

s = content("Granular balls: training on summaries instead of samples", "Granular-ball computing replaces many points with a few balls, each with a center, radius and label");
s.addImage({ path: path.join(FIG, "granulation.png"), x: 0.6, y: 1.6, w: 7.4, h: 3.1, altText: "Raw samples next to the granular balls built from them, with deleted singletons marked" });
s.addText(bullets([
  ["Build: ", "start with one ball holding all data. While a ball's purity is below ρ, split it with KMeans(k = 2) and recurse."],
  ["Summarise: ", "each ball keeps center c, radius r (mean distance to c), majority label and purity."],
  ["Filter: ", "del_ball drops balls with fewer than 2 samples (the red crosses)."],
  ["Promise: ", "fewer training units, so it is faster and more robust to label noise."],
]), T({ x: 8.3, y: 1.6, w: 4.4, h: 4.9, fontSize: 15, paraSpaceAfter: 8 }));
s.addText("Illustration on synthetic 2-D data with a 3:1 imbalance. The deleted singletons sit exactly where the classes meet.", T({ x: 0.6, y: 4.9, w: 7.4, h: 0.7, fontSize: 13, italic: true, color: C.accent5 }));

s = content("GBLSTSVM, the model we extend", "Tanveer, Sharma, Quadir & Sajid (2025), Pattern Recognition 112021");
const pipe = [["Granulate", "Recursive KMeans(2) until purity ≥ ρ"], ["Filter", "del_ball removes n < 2"], ["Weight", "(our extensions plug in here)"], ["Solve", "LSTSVM on ball centers; radius enters the targets"], ["Predict", "Nearer plane wins"]];
pipe.forEach((p, i) => {
  const x = 0.6 + i * 2.5;
  box(s, x, 1.65, 2.2, 0.75, p[0], i === 2 ? HEX.orange : HEX.teal, { fs: 16 });
  s.addText(p[1], T({ x, y: 2.5, w: 2.2, h: 0.9, fontSize: 13, align: "center" }));
  if (i < 4) arrow(s, x + 2.22, 2.02, x + 2.48, 2.02);
});
card(s, 0.6, 3.7, 6.0, 2.8, "Why it was a good starting point", bullets(["Published, with reference code we could reproduce exactly", "Linear and cheap, so we could run thousands of controlled fits", "Plenty of room on imbalanced data: minority recall was often poor"]), HEX.green);
card(s, 6.9, 3.7, 5.8, 2.8, "Original research question", "Can we make ball construction and ball weighting more reliable under class imbalance, sparse minority support and unstable topology, without redesigning the classifier?", HEX.orange, { bs: 15 });

s = content("How we run experiments", "Rules we set before Session 1 and kept throughout");
const rules = [
  ["Freeze the baseline", "Reference code reproduced bit-for-bit first; the frozen modules are never edited during an experiment."],
  ["Pre-register the test", "Each session states hypotheses and what would falsify them before any run."],
  ["Training data only", "Scaling, balls, class weights and tuning use the training split; test is touched once."],
  ["Paired statistics", "17 UCI/KEEL datasets × 10 seeds; dataset is the unit; Wilcoxon, bootstrap CIs, Holm correction."],
  ["Diagnose mechanisms", "Log weights, balls and hyperplanes, not just F1, to tell 'inactive' apart from 'active but useless'."],
  ["Stop when falsified", "One fix per failure; no tuning an extension until it wins."],
];
rules.forEach((r, i) => card(s, 0.6 + (i % 3) * 4.1, 1.6 + Math.floor(i / 3) * 2.55, 3.85, 2.3, r[0], r[1], [HEX.teal, HEX.orange, HEX.green, HEX.red, HEX.gray, HEX.navy][i]));

s = content("Timeline of the work", "Aug → Oct 2026");
const tl = [["S1", "Reproduce + CW/R/CR"], ["S2", "Real data, BMRU, OBRU"], ["S3", "Topo, SA-Topo"], ["S4", "Granulation diagnostics"], ["S5", "Split gates, entropy"], ["S6", "Compression K"], ["S7", "Ablation → stop"], ["A3–A7", "Re-baseline, ladder"], ["S8", "Angle gate, paper plan"]];
s.addShape(pres.shapes.LINE, { x: 0.8, y: 3.3, w: 11.8, h: 0, line: { color: HEX.gray, width: 2 } });
tl.forEach((t, i) => {
  const x = 0.6 + i * 1.36, col = i < 3 ? HEX.orange : i < 7 ? HEX.teal : HEX.green;
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 2.95, w: 0.7, h: 0.7, fill: { color: col }, line: { color: C.background1, width: 2 } });
  s.addText(t[0], T({ x: x + 0.05, y: 2.95, w: 1.2, h: 0.7, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle" }));
  s.addText(t[1], T({ x: x - 0.02, y: i % 2 ? 1.85 : 3.85, w: 1.34, h: 0.9, fontSize: 13, align: "center", valign: i % 2 ? "bottom" : "top" }));
});
[["Phase A: weighting", HEX.orange, 0.6, 4.0], ["Phase B: granulation", HEX.teal, 4.7, 5.4], ["Phase C: theory", HEX.green, 10.15, 2.55]].forEach((p) => box(s, p[2], 5.4, p[3], 0.6, p[0], p[1], { fs: 14 }));

// =====================================================================
// SESSION 1
// =====================================================================
sec("Session 1: reproduction and weighting", "Phases 1–5: rebuild the baseline exactly, then add class and reliability weights");

s = content("Phase 1: reproducing the published baseline", "A clean reimplementation matched the reference code to machine precision");
stat(s, 0.6, 1.6, 2.8, "18 = 18", "balls from clean vs reference code (n=100, d=4, ρ=0.985)", HEX.green, 36);
stat(s, 3.6, 1.6, 2.8, "0.0", "max difference in any center or radius", HEX.green, 36);
stat(s, 6.6, 1.6, 2.8, "100%", "prediction agreement with the reference rule", HEX.green, 36);
card(s, 0.6, 3.6, 12.1, 3.0, "The first surprise: the paper and the code predict differently", [
  { text: "Paper: argmin |fᵢ(x)| / ‖wᵢ‖      Code: sign(|f₂(x)| − |f₁(x)|)", options: { bold: true, color: C.accent1, breakLine: true } },
  { text: "They agree on only 71% of points in the synthetic check. There ‖w₁‖ = 0.643 and ‖w₂‖ = 0.00057, a norm ratio of about 1,118, so dividing by ‖w‖ flips many decisions.", options: { breakLine: true } },
  { text: "Decision: keep both rules, use the repository rule as the primary baseline, and record the rule in every experiment. Session 8 later explains this gap.", options: {} },
], HEX.orange, { bs: 15 });

s = content("Phases 2–4: class, reliability and combined weights", "Three controlled extensions that each reduce exactly to the baseline");
table(s, [["Model", "Ball weight", "Idea"], ["CW", "κ_y = n / (2 n_y)", "Upweight the minority class"], ["R", "a = ε + (1−ε) pᵞ exp(−λ r̃ (1−p))", "Trust pure, compact balls more"], ["CR", "κ_y × a", "Class emphasis × local trust"]], 0.6, 1.6, 7.3, [1.0, 3.6, 2.7]);
s.addText(bullets(["Weights enter the optimisation on the opposite class only, never the prediction", "Unit tests prove the reductions: CR→CW when a=1, CR→R when balanced, both off → baseline", "59 tests passing at the end of Session 1"]), T({ x: 0.6, y: 3.6, w: 7.3, h: 2.5, fontSize: 15, paraSpaceAfter: 6 }));
card(s, 8.2, 1.6, 4.5, 4.9, "Result: CR = CW", bullets([
  "At ρ = 0.95 nearly every ball is pure, so a ≈ 1 and CR collapses to CW",
  "Lowering ρ to 0.80–0.85 restores variation (std of a ≈ 0.07–0.12)",
  "Predictions still barely change: mean Δ(CR − CW) ≈ 0 across thresholds",
  "One +0.014 and one −0.031 macro-F1 case: noise, not a trend",
]), HEX.red);

s = content("Session 1 inference", "Reliability did what its formula said, but it did not change predictions");
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.6, w: 12.1, h: 1.3, rectRadius: 0.08, fill: { color: "FBE9DD" }, line: { color: "FBE9DD" }, objectName: "s1 key" });
s.addText("Purity-driven granulation makes almost every ball pure. A reliability score built from purity and radius then has nothing to say.", T({ x: 0.9, y: 1.6, w: 11.5, h: 1.3, fontSize: 18, bold: true, color: C.text2, valign: "middle" }));
card(s, 0.6, 3.2, 3.85, 3.3, "Established", bullets(["Exact reproduction", "CW is a valid extension", "R behaves as designed"]), HEX.green);
card(s, 4.7, 3.2, 3.85, 3.3, "Not established", bullets(["Any gain from R beyond CW", "Behaviour on real data"]), HEX.red);
card(s, 8.8, 3.2, 3.9, 3.3, "Handed to Session 2", bullets(["Run on real imbalanced data before tuning anything", "Log whether weights move hyperplanes and labels"]), HEX.teal);

// =====================================================================
// SESSION 2
// =====================================================================
sec("Session 2: real data, BMRU and OBRU", "Phases 6–8 plus two new granular-ball signals");

s = content("Phases 6–7: the first real datasets", "Frozen protocol: seed 42, 70/15/15 split, no scaling, no tuning");
table(s, [["Dataset", "Samples", "Imbalance", "Balls", "Macro F1", "Min. recall"], ["breast_cancer", "286", "2.4 : 1", "–", "0.411", "0.000"], ["diabetes", "768", "1.87 : 1", "85", "0.393", "0.000"], ["ecoli2", "336", "5.46 : 1", "13", "0.805", "0.625"], ["vehicle0", "846", "3.25 : 1", "59", "0.692", "0.400"]], 0.6, 1.6, 7.4, [1.8, 1.1, 1.2, 0.9, 1.2, 1.2], [1, 2]);
s.addText(bullets([
  "breast_cancer and diabetes collapsed: every model predicted the majority class only",
  "On breast_cancer del_ball dropped 77 of 200 training samples; only 20 of 59 minority points made it into balls",
  "Only vehicle0 improved: CW beat the baseline by +0.014 macro F1 and +0.067 minority recall",
  "CR never beat CW, even where reliability was active",
]), T({ x: 0.6, y: 3.65, w: 7.4, h: 2.6, fontSize: 15, paraSpaceAfter: 6 }));
card(s, 8.3, 1.6, 4.4, 4.9, "Inference", "The failure was stack-wide: majority collapse plus saturated reliability. It was not specific to CR.\n\nThe question became whether this was the protocol (no scaling, default c₁, c₂) or the mechanisms.", HEX.orange, { bs: 15 });

s = content("Phase 8: is the null result a protocol artifact?", "Six ablations: purity, scaling, c₁×c₂ grid, final protocol, six seeds, mechanism diagnostics");
s.addChart(pres.charts.BAR, [
  { name: "CW − Clean", labels: ["diabetes", "ecoli2", "vehicle0"], values: [0.003, -0.025, 0.011] },
  { name: "CR − CW", labels: ["diabetes", "ecoli2", "vehicle0"], values: [-0.007, 0, 0] },
], Object.assign(chartOpts("Tuned test Δ macro-F1", { barDir: "col", chartColors: [HEX.teal, HEX.orange], showLegend: true, legendPos: "b", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000" }), { x: 0.6, y: 1.6, w: 6.2, h: 4.9 }));
s.addText(bullets([
  ["Scaling fixed the collapse. ", "MinMax scaling plus tuned c₁/c₂ took diabetes validation F1 from ~0.39 to ~0.76."],
  ["Defaults were poor. ", "c₁ = 100, c₂ = 0.01 is far from optimal, and tuning changes conclusions."],
  ["Six seeds: ", "mean (CW − Clean) is negative on all three datasets; mean (CR − CW) ≤ +0.01."],
  ["Mechanism: ", "R moves the hyperplanes (‖w_CR − w_CW‖ > 0) but test labels rarely change."],
]), T({ x: 7.1, y: 1.6, w: 5.6, h: 4.9, fontSize: 14, paraSpaceAfter: 8 }));

s = content("BMRU and OBRU: two new ball signals, both falsified", "Each was tested against a diagnostic it had to pass, not just F1");
card(s, 0.6, 1.6, 6.0, 4.3, "BMRU: margin uncertainty", [
  { text: "u = 2r / |s|, μ = exp(−u): downweight balls wide relative to their margin.", options: { breakLine: true, italic: true } },
  ...bullets(["vehicle0: CW+BMRU lost 0.261 macro F1; minority recall fell 0.40 → 0.00", "Falsification: u ≥ 1 on 98.7% of diabetes balls, yet 0% straddle the boundary", "Large u reflects ball size, not boundary position"], 14),
], HEX.red);
card(s, 6.9, 1.6, 5.8, 4.3, "OBRU: inter-class overlap", [
  { text: "O = overlap of opposite-class balls; minority μ = 1 + O, majority μ = 1 − O.", options: { breakLine: true, italic: true } },
  ...bullets(["Overlap was zero on 93% (diabetes), 100% (ecoli2) and 61% (vehicle0) of balls", "corr(CW×μ, CW) ≈ 0.998: a tiny perturbation of CW", "0 of 3 datasets improved on any metric"], 14),
], HEX.red);
verdict(s, 0.6, 6.1, 12.1, "Session 2 verdict: class weighting is the only mechanism with any positive signal, and it is small", HEX.navy);

s = content("Between sessions: fuzzy-membership redesign", "Why the failures failed, and what a better ball signal would need");
table(s, [["Mechanism", "Signal", "Outcome"], ["R", "purity, radius (inside the ball)", "Redundant with CW"], ["BMRU", "radius / margin (self-referential)", "Harmful"], ["OBRU", "literal ball intersection", "Never fires"], ["CA", "soft k-NN class competition among centers", "Best so far (vehicle0 gain)"], ["IPA, SP, CBCM", "isolation, same-class connectivity", "Failed criteria"]], 0.6, 1.6, 7.6, [1.6, 3.6, 2.4], [4]);
card(s, 8.5, 1.6, 4.2, 4.9, "Design lesson", "A useful signal must describe where the other class sits relative to a ball, across the neighbourhood of ball centers. It must also tell real minority regions from isolated noise.\n\nCompetition ambiguity (CA) became the frozen baseline for Session 3.", HEX.teal, { bs: 14 });

// =====================================================================
// SESSION 3
// =====================================================================
sec("Session 3: topology-supported weights", "Does reciprocal opposite-class support on the ball graph filter noisy competition?");

s = content("Topo-GBLSTSVM: the mechanism", "Build a k-NN graph on ball centers; trust a ball's competition only if the opposite class 'points back'");
// graph diagram
const nodes = [[1.3, 2.4, "A"], [2.6, 1.9, "A"], [2.1, 3.4, "A"], [3.9, 2.6, "B"], [4.6, 3.7, "B"], [3.4, 4.6, "A"], [5.2, 2.2, "B"], [1.4, 4.8, "A"]];
const edges = [[0, 1], [0, 2], [1, 3], [3, 1], [2, 5], [3, 4], [4, 5], [3, 6], [6, 4], [5, 7], [2, 7]];
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.6, w: 5.6, h: 4.0, fill: { color: "F7FAFB" }, line: { color: "E3E9EE" }, objectName: "graph canvas" });
edges.forEach((e) => { const a = nodes[e[0]], b = nodes[e[1]]; s.addShape(pres.shapes.LINE, { x: Math.min(a[0], b[0]) + 0.25, y: Math.min(a[1], b[1]) + 0.25, w: Math.abs(a[0] - b[0]) || 0.01, h: Math.abs(a[1] - b[1]) || 0.01, flipV: (b[0] - a[0]) * (b[1] - a[1]) < 0, line: { color: (a[2] !== b[2]) ? HEX.red : "B9C6CF", width: (a[2] !== b[2]) ? 2.25 : 1.25 } }); });
nodes.forEach((n) => { s.addShape(pres.shapes.OVAL, { x: n[0], y: n[1], w: 0.5, h: 0.5, fill: { color: n[2] === "A" ? HEX.teal : HEX.orange }, line: { color: C.background1, width: 1.5 } }); });
s.addText("red edges: cross-class neighbours. R_opp asks how many of them are reciprocal.", T({ x: 0.6, y: 5.7, w: 5.6, h: 0.6, fontSize: 12, italic: true, color: C.accent5 }));
s.addText([
  { text: "Competition ambiguity", options: { bold: true, breakLine: true } },
  { text: "A = 2·min(comp, 1 − comp), where comp is the opposite-class share of a ball's k-NN.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Reciprocal opposite support", options: { bold: true, breakLine: true } },
  { text: "R_opp = (reciprocal opposite neighbours) / (opposite neighbours).", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Topo weight", options: { bold: true, breakLine: true } },
  { text: "μ = A × R_opp, w = CW(y) × μ, with k = max(5, ⌈0.05 N⌉).", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Novelty check: reciprocal opposite-class symmetry on a ball graph is not a documented FSVM membership, so it was worth screening.", options: { italic: true, color: C.accent1 } },
], T({ x: 6.6, y: 1.6, w: 6.1, h: 5.0, fontSize: 15 }));

s = content("Broad screen: Topo is bimodal", "17 datasets × 3 seeds = 51 runs, frozen hyperparameters, compared against CA");
s.addChart(pres.charts.BAR, [{ name: "Topo − CA", labels: ["chronic_kidney", "spambase", "credit_approval", "vehicle0", "ecoli2", "ionosphere"], values: [0.189, 0.112, 0.023, -0.028, -0.037, -0.085] }], Object.assign(chartOpts("Dataset-mean Δ macro-F1, Topo − CA", { barDir: "bar", chartColors: [HEX.teal], dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000;-0.000", catAxisOrientation: "maxMin", catAxisLabelPos: "low" }), { x: 0.6, y: 1.6, w: 6.4, h: 4.9 }));
table(s, [["Run category", "n", "Mean Δ F1", "Topo wins"], ["informative R_opp", "29", "+0.059", "55%"], ["adversarial R_opp", "19", "−0.092", "0%"], ["redundant", "3", "0.000", "0%"]], 7.3, 1.7, 5.4, [2.2, 0.7, 1.3, 1.2], [2]);
s.addText(bullets(["Overall mean Δ vs CA ≈ −0.001: no average gain", "When R_opp reshapes weights it either clearly helps or always hurts", "Mean corr(Topo, CA weights) = 0.74"]), T({ x: 7.3, y: 3.6, w: 5.4, h: 2.8, fontSize: 14, paraSpaceAfter: 6 }));

s = content("Forensics: why Topo fails", "Ball-level comparison of three successful and three adversarial runs");
table(s, [["Metric", "Successful", "Adversarial"], ["Total balls (3 runs)", "685", "91"], ["Mean n_opp of downweighted balls", "10.1", "2.3"], ["High-competition minority balls with R_opp = 0", "2.2%", "31.6%"], ["Mean |Topo − CA| on those balls", "0.37", "0.65"]], 0.6, 1.6, 7.0, [3.8, 1.6, 1.6], [3]);
const fl = [["Sparse ball graph (N ≈ k)", HEX.gray], ["Few opposite neighbours", HEX.gray], ["R_opp = 0 on a contested minority ball", HEX.orange], ["Its CA weight is wiped out", HEX.red], ["Minority recall collapses", HEX.red]];
fl.forEach((f, i) => { box(s, 8.2, 1.6 + i * 1.0, 4.5, 0.72, f[0], f[1], { fs: 14 }); if (i < 4) arrow(s, 10.45, 2.33 + i * 1.0, 10.45, 2.58 + i * 1.0); });
s.addText("Purity and radius do not explain the failures. ecoli2 has only 13 balls with k = 5; ionosphere's mean R_opp is 0.08.", T({ x: 0.6, y: 4.6, w: 7.0, h: 1.2, fontSize: 15 }));

s = content("SupportAware-Topo: one principled fix, tested once", "Shrink R_opp toward 1 when there is little evidence:  R_opp^S = c·R_opp + (1 − c),  c = n_opp / (n_opp + 1)");
s.addChart(pres.charts.BAR, [
  { name: "Topo", labels: ["ecoli2_s42", "ionosphere_s0", "vehicle0_s1", "chronic_kidney_s42", "spambase_s42"], values: [-0.215, -0.193, -0.167, 0.336, 0.128] },
  { name: "SupportAware-Topo", labels: ["ecoli2_s42", "ionosphere_s0", "vehicle0_s1", "chronic_kidney_s42", "spambase_s42"], values: [0.024, 0, 0.005, 0, 0.177] },
], Object.assign(chartOpts("Δ macro-F1 vs CA on focus runs", { barDir: "col", chartColors: [HEX.red, HEX.green], showLegend: true, legendPos: "b", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00" }), { x: 0.6, y: 1.6, w: 7.4, h: 4.9 }));
card(s, 8.3, 1.6, 4.4, 4.9, "Inference", bullets(["Removes all three catastrophes with no tuning", "Keeps and improves the dense-graph win (spambase)", "Erases the biggest win (chronic_kidney, mean n_opp 1.6)", "n_opp alone cannot separate trusted from untrusted sparse graphs"]), HEX.orange);

s = content("Session 3 inference", "The signal is real but only where the ball graph is dense enough to trust it");
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.6, w: 12.1, h: 1.4, rectRadius: 0.08, fill: { color: "FBE9DD" }, line: { color: "FBE9DD" }, objectName: "s3 key" });
s.addText("Every downstream weight depends on how many balls survive granulation, and how many of them are minority. That moved our attention upstream.", T({ x: 0.9, y: 1.6, w: 11.5, h: 1.4, fontSize: 18, bold: true, color: C.text2, valign: "middle" }));
card(s, 0.6, 3.3, 6.0, 3.2, "Decisions", bullets(["Do not promote Topo or SA-Topo to cross-validation", "No further tweaks (one test per fix)"]), HEX.navy);
card(s, 6.9, 3.3, 5.8, 3.2, "Open question for Session 4", "Why are minority balls so sparse to begin with? Is the damage done during granulation?", HEX.teal, { bs: 15 });

// =====================================================================
// SESSION 4
// =====================================================================
sec("Session 4: granulation diagnostics", "Read-only replays of 51 runs and 13,654 KMeans splits to find where minority information is lost");

s = content("Nine candidate mechanisms screened", "Each was a read-only diagnostic with a pre-set pass gate (AUROC ≥ 0.80 where applicable)");
table(s, [["Mechanism", "Key number", "Verdict"],
  ["hd1/hd2 singleton outlier rule", "75% of singletons have hd2 undefined", "Rejected"],
  ["Nested heterogeneous balls", "9 pairs total, 0 in Topo failures", "Rejected"],
  ["Gaussian membership", "corr with CA 0.05; hurts 2 of 3 datasets", "Rejected"],
  ["Gaussian overlap (Bhattacharyya)", "AUROC 0.33", "Rejected"],
  ["Minority isolation chain", "Reproducible; corr with ΔF1 0.05", "Partial"],
  ["B1 minority repair", "survival 0.58→0.77 but geometry worsens", "Rejected"],
  ["Parent-local singleton repair", "best AUROC ≈ 0.58", "Rejected"],
  ["Topo harm_risk score", "AUROC 0.525", "Rejected"],
  ["Split utility geometry", "center-radius separation AUROC 0.864", "Supported"]], 0.6, 1.6, 12.1, [4.2, 5.6, 2.3], [9], 13);

s = content("The minority isolation chain", "The most consistent pathway by which minority samples disappear");
const chain = [["Small impure parent", "median parent size 3"], ["KMeans(2) split", "always makes two children"], ["Minority peeled off", "25% of splits isolate minority"], ["Singleton child", "9,174 singleton events"], ["del_ball removes it", "100% of singletons deleted"], ["Minority survival drops", "0.583 vs 0.750 majority"]];
chain.forEach((c, i) => {
  const x = 0.6 + i * 2.05;
  box(s, x, 1.8, 1.8, 1.1, c[0], i < 3 ? HEX.teal : HEX.red, { fs: 14 });
  s.addText(c[1], T({ x, y: 3.0, w: 1.8, h: 0.8, fontSize: 13, align: "center" }));
  if (i < 5) arrow(s, x + 1.82, 2.35, x + 2.03, 2.35);
});
stat(s, 0.6, 4.2, 3.0, "−0.167", "minority minus majority survival; minority is lower in 92% of runs", HEX.red, 34);
stat(s, 3.9, 4.2, 3.0, "80%", "of isolating splits end up hit by del_ball", HEX.red, 34);
stat(s, 7.2, 4.2, 2.6, "−0.52", "corr(isolating-split rate, minority survival)", HEX.teal, 34);
stat(s, 10.1, 4.2, 2.6, "0.05", "corr(isolating rate, Δ F1): no direct classifier link", HEX.gray, 34);

s = content("Split utility: a geometry signal that predicts harmful splits", "Harmful minority peels (class A, 44% of splits) vs useful splits where both children survive (class C, 33%)");
s.addChart(pres.charts.BAR, [{ name: "AUROC", labels: ["parent_n", "center-radius separation", "radius + separation", "separation / compactness", "r₁", "purity gain", "split depth", "size ratio", "radius reduction"], values: [0.930, 0.864, 0.862, 0.824, 0.800, 0.716, 0.708, 0.593, 0.532] }], Object.assign(chartOpts("AUROC for separating C from A", { barDir: "bar", chartColors: [HEX.teal], dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", catAxisOrientation: "maxMin", valAxisMinVal: 0.5, valAxisMaxVal: 1.0 }), { x: 0.6, y: 1.6, w: 7.2, h: 5.0 }));
table(s, [["Median", "Class A (harmful)", "Class C (useful)"], ["parent size", "3", "16"], ["center-radius sep.", "2.87", "0.92"], ["split depth", "10", "7"]], 8.1, 1.7, 4.6, [1.8, 1.4, 1.4]);
s.addText(bullets(["‖c₁ − c₂‖ / (r₁ + r₂) is the first non-tautological geometry measure to clear 0.80", "Harmful peels look like clean KMeans splits on tiny parents, then lose their small child", "Caveat: it predicts deletion; it does not prove the split is wrong by itself"]), T({ x: 8.1, y: 3.4, w: 4.6, h: 3.2, fontSize: 14, paraSpaceAfter: 6 }));

s = content("Session 4 inference", "The damage happens upstream, in granulation, before any weight is computed");
card(s, 0.6, 1.6, 3.85, 4.9, "Know", bullets(["Granulation deletes minority samples systematically", "Weight fixes and singleton repair cannot recover it", "Split geometry predicts the harmful splits"]), HEX.green);
card(s, 4.7, 1.6, 3.85, 4.9, "Think", bullets(["Purity-driven KMeans keeps splitting small mixed parents into doomed peels", "A split-acceptance gate is the best place to intervene"]), HEX.orange);
card(s, 8.8, 1.6, 3.9, 4.9, "Don't know", bullets(["Whether a split gate improves the classifier at all", "Whether the AUROC holds out of sample"]), HEX.red);

// =====================================================================
// SESSIONS 5-7
// =====================================================================
sec("Sessions 5–7: fixing granulation", "Split gates, entropy schedulers, compression and a fair ablation");

s = content("Session 5: split gates and entropy scheduling", "Translate the split-utility signal into interventions without touching the classifier");
card(s, 0.6, 1.6, 4.0, 4.9, "What predicts F1?", [
  { text: "Across 306 runs (17 × 3 × 6 representations):", options: { breakLine: true } },
  ...bullets(["minority survival r = +0.57", "fraction mixed balls r = +0.35", "mean purity r = +0.04", "mean radius r = −0.13"], 14).map((r, i, a) => i === a.length - 1 ? { text: r.text, options: Object.assign({}, r.options, { breakLine: true }) } : r),
  { text: "Survival matters, purity does not.", options: { bold: true } },
], HEX.teal);
s.addChart(pres.charts.BAR, [
  { name: "Macro F1", labels: ["baseline ρ=0.9", "entropy v2"], values: [0.597, 0.613] },
  { name: "Collapse rate", labels: ["baseline ρ=0.9", "entropy v2"], values: [0.059, 0.518] },
], Object.assign(chartOpts("Online entropy vs baseline", { barDir: "col", chartColors: [HEX.teal, HEX.red], showLegend: true, legendPos: "b", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00" }), { x: 4.9, y: 1.6, w: 4.1, h: 4.9 }));
card(s, 9.3, 1.6, 3.4, 4.9, "Why entropy fails", bullets(["Stops after 3.6 splits vs 240", "94% of runs run out of candidates", "Guards cut collapse only to ~49%", "Split gates (SU1–SU3): no classifier win"], 14), HEX.red);

s = content("Session 6: compression helps, ranking does not", "Keep only K balls from the ρ = 0.9 pool (17 datasets × 10 seeds × 13 values of K)");
s.addChart(pres.charts.BAR, [{ name: "Macro F1", labels: ["full ρ=0.9 (77 balls)", "random K=2", "entropy K=2", "sample-stratified K=2", "validation-selected K"], values: [0.584, 0.647, 0.647, 0.647, 0.623] }], Object.assign(chartOpts("Mean test macro-F1", { barDir: "col", chartColors: [HEX.teal], dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.5, valAxisMaxVal: 0.68 }), { x: 0.6, y: 1.6, w: 7.2, h: 4.9 }));
s.addText(bullets([
  ["Compression: ", "keeping very few balls beat the full set by about +0.06 F1."],
  ["Ranking: ", "entropy, boundary and random selection tie (Δ < 0.01)."],
  ["Validation-selected K: ", "+0.039 over full ρ = 0.9 (p = 0.002), median K ≈ 12."],
  ["Red flag: ", "it also beat a 'matched' sample control by +0.192, which was suspiciously large."],
]), T({ x: 8.1, y: 1.6, w: 4.6, h: 4.9, fontSize: 14, paraSpaceAfter: 8 }));
note(s, "Validation-selected value is 0.584 + 0.039.");

s = content("Session 7: reconcile the controls, then ablate", "The two 'sample controls' differed in sampling, representation and class weights at once");
table(s, [["", "sample_stratified_k", "sample_stratified_matched"], ["Sampling RNG", "per-K", "per-target size"], ["Representation", "re-granulate ρ=0.9", "one ball per sample"], ["Class weights", "full y_train", "subset only"], ["Macro F1", "0.617", "0.474"]], 0.6, 1.6, 6.3, [1.9, 2.2, 2.2], [4]);
s.addText("Then: fix one identical sample subset (hash-verified across every arm) and change only the representation.", T({ x: 0.6, y: 4.3, w: 6.3, h: 1.0, fontSize: 15, bold: true, color: C.text2 }));
s.addChart(pres.charts.BAR, [{ name: "Macro F1", labels: ["singleton", "regran ρ=0.80", "regran ρ=0.90", "regran ρ=0.98", "subsample + ρ=0.9"], values: [0.4775, 0.6211, 0.6179, 0.6038, 0.6179] }], Object.assign(chartOpts("Same samples, different representation", { barDir: "col", chartColors: [HEX.teal], dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.4, valAxisMaxVal: 0.66 }), { x: 7.2, y: 1.6, w: 5.5, h: 4.4 }));
verdict(s, 0.6, 6.1, 12.1, "CASE B: re-granulating is no better than plain subsampling (0.6179 = 0.6179). Stop the granular-structure branch.", HEX.red);

s = content("What Sessions 5–7 taught us", "Every apparent granular gain had a simpler explanation");
const lessons = [["Gains came from compression", "Fewer training units regularised the solver; which balls were kept barely mattered."], ["…and from weight semantics", "Class weights from the full training set vs the subset explained the 0.14 gap."], ["Granular structure added nothing", "Re-granulation matched subsampling exactly once samples were held fixed."], ["So we changed the question", "Instead of 'how do we build better balls', ask 'what information does a ball throw away?'"]];
lessons.forEach((l, i) => card(s, 0.6 + (i % 2) * 6.15, 1.6 + Math.floor(i / 2) * 2.55, 5.95, 2.3, l[0], l[1], [HEX.teal, HEX.teal, HEX.red, HEX.green][i], { bs: 15 }));

// =====================================================================
// BRIDGE A3-A7
// =====================================================================
sec("Bridge: re-baselining and the estimator view", "Experiments A1–A7, between Sessions 7 and 8");

s = content("Re-baseline: protocol choices dwarf mechanisms", "2 × 2 × 2 factorial on the published GBLSTSVM: scaling × prediction rule × tuning c (17 datasets × 10 seeds)");
s.addChart(pres.charts.BAR, [
  { name: "Default c", labels: ["code rule, unscaled", "paper rule, unscaled", "code rule, scaled", "paper rule, scaled"], values: [0.584, 0.651, 0.648, 0.668] },
  { name: "Tuned c", labels: ["code rule, unscaled", "paper rule, unscaled", "code rule, scaled", "paper rule, scaled"], values: [0.741, 0.748, 0.773, 0.774] },
], Object.assign(chartOpts("Mean test macro-F1", { barDir: "col", chartColors: [HEX.gray, HEX.teal], showLegend: true, legendPos: "b", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.5, valAxisMaxVal: 0.82 }), { x: 0.6, y: 1.6, w: 7.4, h: 4.9 }));
card(s, 8.3, 1.6, 4.4, 4.9, "Inference", bullets(["Tuning c alone adds +0.16 F1, more than any extension we built", "The rule and scaling each move F1 by several points", "Every earlier comparison must state these three choices", "Remaining gaps to published accuracy: 1–10 points (e.g. ecoli2 −5)"]), HEX.orange);

s = content("The key observation: LSTSVM only needs moments", "If balls store count, mean and scatter, the full-data solution is recovered exactly");
s.addText([
  { text: "Each LSTSVM plane solves  (GᵀG + HᵀH / c) z = −Gᵀe.", options: { breakLine: true } },
  { text: "The data enter only through sums of outer products.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Within a ball j with nⱼ points, mean μⱼ and scatter Sⱼ:", options: { breakLine: true } },
  { text: "Σᵢ xᵢxᵢᵀ = nⱼ μⱼμⱼᵀ + Sⱼ", options: { bold: true, color: C.accent1, breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "So a ball that keeps (nⱼ, μⱼ, Sⱼ) loses nothing, for any partition and any number of balls. A ball that keeps only center and radius drops Sⱼ and approximates.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "We found no granular-ball paper that states this; checking that is part of the literature review.", options: { italic: true, color: C.accent5 } },
], T({ x: 0.6, y: 1.6, w: 6.2, h: 5.0, fontSize: 16 }));
s.addText("Information ladder of ball summaries", T({ x: 7.2, y: 1.6, w: 5.5, h: 0.45, fontSize: 16, bold: true, color: C.text2 }));
[["L0", "center only, unit weight"], ["L1", "center × population nⱼ"], ["ISO", "+ isotropic scatter (tr Sⱼ/d)·I"], ["DIAG", "+ diagonal of Sⱼ"], ["FULL = L3", "+ full Sⱼ  ≡ RAW (exact)"]].forEach((l, i) => {
  const y = 6.0 - i * 0.85, w = 2.8 + i * 0.68;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y, w, h: 0.7, rectRadius: 0.06, fill: { color: i === 4 ? HEX.green : HEX.teal }, line: { color: C.background1 }, objectName: "ladder " + l[0] });
  s.addText([{ text: l[0] + "  ", options: { bold: true } }, { text: l[1] }], T({ x: 7.35, y, w: w - 0.2, h: 0.7, fontSize: 13, color: C.background1, valign: "middle" }));
});

s = content("A3–A7: first tests of the ladder", "17 datasets × 10 seeds, ρ = 0.9, code rule, c tuned per arm");
table(s, [["Exp.", "Question", "Finding"],
  ["A3", "Do the two prediction rules disagree on real data?", "14% of test points; paper rule +0.065 F1 (p = 0.08); fewer collapses (11% vs 20%)"],
  ["A4", "Does each rung add information?", "L1 − L0 = +0.012; L2 − L1 = +0.031 (p = 0.004, 14/17 wins); L3 = RAW"],
  ["A5", "Is L3 = RAW for any number of balls?", "Yes on a complete partition; L2 − L1 depends on the rule"],
  ["A6", "Is the scatter gain directional or just extra ridge?", "Labelled 'no clear scatter advantage'; ISO ≡ trace-matched ridge"],
  ["A7", "How far is each arm's plane from RAW?", "FULL 23°, L1 38°, ISO 52°, which made no sense if FULL is exact"]], 0.6, 1.6, 12.1, [0.8, 4.6, 6.7], [5], 13);
verdict(s, 0.6, 5.8, 12.1, "If FULL is exact, why is it 23° from RAW? Session 8 was built to answer this before claiming anything.", HEX.navy);

// =====================================================================
// SESSION 8
// =====================================================================
sec("Session 8: angle gate, clean geometry and paper strategy", "29 Sep 2026: gate first, then every claim that depends on it");

s = content("Session 8 plan", "Each step decides whether a later claim is allowed");
const plan8 = [["E0", "Gate", "Is θ(FULL, RAW) really 0? What made A7's 23°?"], ["E1", "Clean geometry", "θ for each rung at matched c; does θ track discarded variance?"], ["E2", "Held-out ladder", "Do the classification results survive leak-free tuning?"], ["E3", "Norm ratio", "Does ‖w₁‖/‖w₂‖ explain the rule disagreement?"], ["E4/E5", "Kernel pilot + scaling", "Does the ladder hold in feature space? Are balls worth their cost?"]];
plan8.forEach((p, i) => {
  const y = 1.6 + i * 1.0;
  box(s, 0.6, y, 1.2, 0.8, p[0], i === 0 ? HEX.red : HEX.green, { fs: 16 });
  s.addText([{ text: p[1] + "   ", options: { bold: true, color: C.text2 } }, { text: p[2] }], T({ x: 2.0, y, w: 10.7, h: 0.8, fontSize: 16, valign: "middle" }));
  if (i < 4) arrow(s, 1.2, y + 0.82, 1.2, y + 0.98);
});
note(s, "Protocol: 17 datasets, seeds 0–9 (kernel 0–4), 70/15/15 split, dataset-level Wilcoxon + bootstrap, Holm within each table.");

s = content("E0 gate passed: the 23° was deletion, not geometry", "Two ways to build balls: the published deletion path, and a partition path that keeps every sample");
// path diagram
box(s, 0.6, 1.7, 2.2, 0.8, "Training set", HEX.navy, { fs: 14 });
box(s, 3.4, 1.4, 3.0, 0.7, "Deletion path: drop n < 2", HEX.red, { fs: 13 });
box(s, 3.4, 2.3, 3.0, 0.7, "Partition path: keep all", HEX.green, { fs: 13 });
arrow(s, 2.82, 2.0, 3.38, 1.75); arrow(s, 2.82, 2.2, 3.38, 2.65);
table(s, [["FULL arm", "θ vs RAW", "θ vs RAW on kept samples"], ["Partition path, shared c", "0.000°", "16.4°"], ["Partition path, held-out c", "0.000°", "15.0°"], ["Deletion path, shared c", "16.8°", "0.000°"], ["Deletion path, held-out c", "15.5°", "0.000°"]], 0.6, 3.4, 6.6, [3.0, 1.4, 2.2], [1, 2]);
stat(s, 7.7, 1.5, 5.0, "26%", "of training samples silently dropped by del_ball (up to 49% on heart, 47% messidor, 46% german)", HEX.red, 44);
s.addText(bullets(["~16° of A7's 23° is sample deletion; the rest is each arm tuning its own c", "Parameters reproduce A6 to ≤ 1.3e-10 on all 170 RAW runs and to machine precision on 152 of 170 FULL runs (the rest are KMeans ties on one-hot data)", "So every earlier 'vs RAW' comparison mixed compression with deletion"]), T({ x: 7.7, y: 3.4, w: 5.0, h: 3.2, fontSize: 14, paraSpaceAfter: 6 }));

s = content("E1: clean geometry of the ladder", "Partition path, c matched to RAW: angle between each arm's plane and the exact RAW plane");
s.addChart(pres.charts.BAR, [
  { name: "unscaled", labels: ["FULL", "DIAG", "L1", "L0", "ISO"], values: [0, 18.3, 23.6, 28.8, 40.6] },
  { name: "standardized", labels: ["FULL", "DIAG", "L1", "L0", "ISO"], values: [0, 28.8, 38.5, 46.2, 36.8] },
], Object.assign(chartOpts("Mean θ to RAW (degrees)", { barDir: "col", chartColors: [HEX.teal, HEX.orange], showLegend: true, legendPos: "b", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0" }), { x: 0.6, y: 1.6, w: 6.8, h: 4.9 }));
table(s, [["Contrast", "Unscaled", "Standardized", "Robust?"], ["L0 − L1 (population helps)", "+5.2° (p .019)", "+7.7° (p .047)", "Yes"], ["DIAG − ISO (where mass goes)", "−22.3° (p .001)", "−8.0° (p .0006)", "Yes"], ["L1 − DIAG", "+5.3° (n.s.)", "+9.7° (n.s.)", "Direction only"], ["L1 − ISO", "−17.0° (n.s.)", "+1.7° (n.s.)", "No, drop it"]], 7.7, 1.7, 5.0, [1.9, 1.05, 1.15, 0.9], [4], 12);
s.addText("ISO hurts because it penalises low-variance directions, exactly the directions a twin-plane normal wants. θ_ISO tracks scatter anisotropy (Spearman 0.75, p = 0.0005).", T({ x: 7.7, y: 4.75, w: 5.0, h: 1.8, fontSize: 14 }));

s = content("E1: the error grows with discarded variance", "η = share of class variance that center-only balls throw away, varied through ρ ∈ {0.6 … 1.0}");
s.addChart(pres.charts.LINE, [
  { name: "θ L1", labels: ["0.6", "0.7", "0.8", "0.9", "0.95", "1.0"], values: [62.1, 42.0, 33.2, 23.6, 21.6, 16.9] },
  { name: "θ L0", labels: ["0.6", "0.7", "0.8", "0.9", "0.95", "1.0"], values: [63.0, 42.1, 35.6, 29.1, 26.3, 22.9] },
  { name: "θ DIAG", labels: ["0.6", "0.7", "0.8", "0.9", "0.95", "1.0"], values: [33.8, 29.9, 24.3, 18.3, 17.3, 15.1] },
], Object.assign(chartOpts("Mean θ to RAW vs granulation threshold ρ", { chartColors: [HEX.teal, HEX.gray, HEX.green], showLegend: true, legendPos: "b", showValue: false, lineSize: 2.5, lineDataSymbolSize: 7 }), { x: 0.6, y: 1.6, w: 5.6, h: 4.9 }));
s.addImage({ path: path.join(FIG, "theta_vs_eta.png"), x: 6.4, y: 1.55, w: 4.0, h: 3.0, altText: "Scatter of L1 angle against discarded variance, coloured by rho" });
s.addText(bullets(["Mean η falls 0.81 → 0.12 as ρ rises; θ_L1 falls 62° → 17°", "Spearman(θ_L1, η) = 0.75 (p = 2e-19)", "Positive within every one of the 17 datasets (median 0.94)"]), T({ x: 10.6, y: 1.6, w: 2.1, h: 4.9, fontSize: 13, paraSpaceAfter: 6 }));
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.4, y: 4.75, w: 4.0, h: 1.75, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "closed form" });
s.addText([{ text: "Exact closed form", options: { bold: true, breakLine: true } }, { text: "z_L1 = (I + M_L1⁻¹ ΔM) z_RAW", options: { color: C.accent1, bold: true, breakLine: true } }, { text: "ΔM = the dropped scatter blocks. Center-only is RAW pushed through a known linear map.", options: { fontSize: 12 } }], T({ x: 6.6, y: 4.8, w: 3.6, h: 1.65, fontSize: 14 }));
note(s, "Source: experiments/output/a8_angle_gate/summary_sweep.csv and sweep/runs.csv (partition path, unscaled).");

s = content("E2: held-out classification ladder", "A4/A6 had tuned c on rows inside the fitted balls. Now each arm is re-granulated on the selection rows only.");
s.addChart(pres.charts.BAR, [{ name: "Test macro-F1", labels: ["RAW", "D_FULL", "P_L1", "D_L1", "P_L0", "Published"], values: [0.785, 0.775, 0.757, 0.727, 0.725, 0.714] }], Object.assign(chartOpts("Test macro-F1, code rule (17 × 10)", { barDir: "bar", chartColors: [HEX.teal], dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", catAxisOrientation: "maxMin", valAxisMinVal: 0.6, valAxisMaxVal: 0.82 }), { x: 0.6, y: 1.6, w: 6.4, h: 4.9 }));
table(s, [["Contrast", "Δ F1", "Holm p"], ["D_FULL − D_L1 (scatter helps)", "+0.048", "0.018"], ["P_L1 − D_L1 (deletion costs)", "+0.031", "0.019"], ["P_L1 − P_L0 (population helps)", "+0.032", "0.028"], ["Published − RAW", "−0.071", "0.03"]], 7.3, 1.7, 5.4, [3.0, 1.1, 1.3], [1]);
s.addText(bullets(["D = deletion path, P = partition path", "Under the paper rule nothing separates (0.67–0.70, all p ≥ 0.44)", "The rule effect on RAW (0.785 vs 0.685) is bigger than any representation effect", "Granulation ≈ 180 ms per fit vs 0.26 ms for the exact RAW solve: ~700×"]), T({ x: 7.3, y: 4.0, w: 5.4, h: 2.6, fontSize: 13, paraSpaceAfter: 4 }));

s = content("E3: why the two prediction rules disagree", "The paper rule divides by ‖wᵢ‖; the code rule does not");
s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.6, w: 5.4, h: 4.0, fill: { color: "F7FAFB" }, line: { color: "E3E9EE" }, objectName: "wedge canvas" });
s.addShape(pres.shapes.LINE, { x: 1.0, y: 1.9, w: 4.6, h: 3.4, line: { color: HEX.teal, width: 2.5 } });
s.addShape(pres.shapes.LINE, { x: 1.0, y: 4.2, w: 4.6, h: 0.9, flipV: true, line: { color: HEX.orange, width: 2.5 } });
s.addShape(pres.shapes.OVAL, { x: 3.9, y: 3.45, w: 0.25, h: 0.25, fill: { color: HEX.red }, line: { color: HEX.red } });
s.addText("plane 1 (‖w₁‖ large)", T({ x: 1.2, y: 1.75, w: 2.4, h: 0.35, fontSize: 12, color: HEX.teal }));
s.addText("plane 2 (‖w₂‖ small)", T({ x: 1.1, y: 4.75, w: 2.3, h: 0.35, fontSize: 12, color: HEX.orange }));
s.addText("A point in the wedge 1 < |f₁|/|f₂| < ‖w₁‖/‖w₂‖ (red) is classified differently by the two rules.", T({ x: 0.6, y: 5.75, w: 5.4, h: 0.8, fontSize: 13, italic: true, color: C.accent5 }));
stat(s, 6.4, 1.6, 3.0, "0.58", "Spearman of disagreement with |log ‖w₁‖/‖w₂‖|, unscaled (n = 2,040 runs)", HEX.teal, 40);
stat(s, 9.7, 1.6, 3.0, "0.76", "same, standardized features", HEX.teal, 40);
s.addText(bullets(["Earlier dataset-level test (ρ = 0.27, n.s.) was underpowered: averaging left only 17 points", "Larger imbalance also shifts F1 toward the code rule (ρ ≈ −0.2 to −0.3, p < 0.02)", "This explains Session 1's 29% disagreement at a norm ratio of 1,118"]), T({ x: 6.4, y: 3.6, w: 6.3, h: 2.9, fontSize: 15, paraSpaceAfter: 8 }));

s = content("E4/E5: kernel pilot and compute cost", "RBF kernel, standardized, held-out tuned; 17 datasets × 5 seeds");
s.addChart(pres.charts.BAR, [{ name: "Test F1", labels: ["K_RAW", "GB basis, exact", "k-means basis, exact", "random basis, exact", "GB mean embedding", "GB L1 (standard)", "GB L0"], values: [0.843, 0.839, 0.839, 0.831, 0.821, 0.822, 0.797] }], Object.assign(chartOpts("Kernel test macro-F1", { barDir: "bar", chartColors: [HEX.teal], dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", catAxisOrientation: "maxMin", valAxisMinVal: 0.75, valAxisMaxVal: 0.86 }), { x: 0.6, y: 1.6, w: 6.6, h: 4.9 }));
card(s, 7.5, 1.6, 5.2, 2.35, "The ladder holds in feature space", bullets(["L0 < L1 ≈ MEAN < EXACT ≈ RAW", "EXACT − L1 = +0.017 F1 (Holm p = 0.03)", "Second-order structure matters, not just the mean"], 13), HEX.green);
card(s, 7.5, 4.15, 5.2, 2.35, "Balls add nothing as the basis", bullets(["k-means landmarks at the same k tie (Δ = +0.0005)", "At n ≤ 3.2k, granulation costs more than the O(n³) RAW fit", "On hard data GB is slower until n ≈ 8,000 (then saves ~10%)"], 13), HEX.red);

s = content("Session 8: what changed in our thesis", "Several earlier claims were revised because the gate caught them");
card(s, 0.6, 1.6, 6.0, 4.9, "Holds or got stronger", bullets([
  "Moment-storing balls are exact: θ = 0.000°",
  "Center-only error is directional, has a closed form and scales with η (17/17 datasets)",
  "Where restored mass goes matters: DIAG/FULL ≫ ISO in both scalings",
  "With leak-free tuning, scatter restoration improves F1",
  "The rule gap is driven by the norm ratio",
  "New: del_ball discards ~26% of data and costs ~3 F1 points",
]), HEX.green);
card(s, 6.9, 1.6, 5.8, 4.9, "Revised or dropped", bullets([
  "A7's 23°/38°/52°: mixed with deletion and c mismatch",
  "'ISO is worse than restoring nothing': depends on coordinates",
  "A6's 'no scatter advantage': an in-sample tuning artifact",
  "All A4/A6 classification tables: to be re-run held-out",
  "ISO and trace-matched ridge counted as two arms: they are identical",
]), HEX.red);

// =====================================================================
// STATUS
// =====================================================================
sec("Where we stand", "Inferences across all sessions and the plan for the second half");

s = content("Inferences across the whole project", "What eight sessions of falsification add up to");
const inf = [
  ["Weights cannot fix lost data", "Sessions 1–3: every weighting scheme was limited by what granulation had already removed."],
  ["Granulation loses the minority", "Session 4: purity-driven KMeans plus del_ball drops minority samples systematically."],
  ["Ball structure is not the source of gains", "Sessions 5–7: gains came from compression and class-weight semantics."],
  ["The loss is in the summary, not the partition", "A3–A8: keep moments and any partition is exact; drop scatter and the error is measurable."],
  ["Protocol matters more than mechanism", "Tuning, scaling and the prediction rule each outweigh every extension."],
  ["Balls do not pay for themselves", "Linear: ~700× slower than exact. Kernel: no better than k-means landmarks."],
];
inf.forEach((r, i) => card(s, 0.6 + (i % 2) * 6.15, 1.6 + Math.floor(i / 2) * 1.68, 5.95, 1.5, r[0], r[1], [HEX.orange, HEX.teal, HEX.teal, HEX.green, HEX.gray, HEX.red][i], { bs: 13, hs: 15 }));

s = content("Engineering behind the results");
const eng = [
  ["Reproducible pipeline", "Frozen clean reimplementation, 17-dataset registry, fixed seeds and splits, paired statistics with Holm correction."],
  ["Models and modules", "CW, R, CR, BMRU, OBRU, CA, Topo, SA-Topo, ladder_gblstsvm, margin_multigranularity, svm_aware_granulation."],
  ["Tests", "184+ unit tests, including reduction identities, sample-hash checks and moment identities."],
  ["Bugs found", "A pandas 3 dtype check silently dropped 5 of 17 datasets; an in-sample validation leak in A4/A6 tuning."],
  ["Documentation", "Eight session write-ups with hypotheses, verdicts and handoffs, plus a research proposal."],
  ["Side probes", "SVM-aware and margin multi-granularity generators: no advantage over purity-1.0 balls."],
];
eng.forEach((e, i) => card(s, 0.6 + (i % 3) * 4.1, 1.5 + Math.floor(i / 3) * 2.6, 3.85, 2.4, e[0], e[1], [HEX.teal, HEX.green, HEX.navy, HEX.red, HEX.orange, HEX.gray][i], { bs: 13 }));

s = content("Plan for the second half", "Write the analytical paper; the kernel pilot becomes a section");
const next = [
  "Re-run A4/A6 with the held-out protocol for the paper tables",
  "Report angles in standardized coordinates; unscaled as robustness check",
  "Plot θ_L1 against η across ρ for the paper",
  "Predict θ from ‖M_L1⁻¹ΔM‖ before fitting, to test the closed form",
  "Literature review: confirm no GB paper states the moment identity",
  "Optional: one large kernel dataset (n ≈ 20k) to settle compute",
];
next.forEach((t, i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 1.6;
  s.addShape(pres.shapes.OVAL, { x, y: y + 0.15, w: 0.7, h: 0.7, fill: { color: i === 4 ? HEX.orange : HEX.teal }, line: { color: C.background1 } });
  s.addText(String(i + 1), T({ x, y: y + 0.15, w: 0.7, h: 0.7, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle" }));
  s.addText(t, T({ x: x + 0.95, y, w: 4.9, h: 1.0, fontSize: 16, valign: "middle" }));
});
note(s, "A method paper would need something GB-specific that beats Nyström / reduced-kernel LSTSVM with k-means landmarks; nothing so far suggests that exists.");

s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
s.addText("Recommendation: an analytical paper", { placeholder: "title" });
s.addText("In the linear case GBLSTSVM is a lossy, deletion-affected approximation of an exact estimator that is ~700× cheaper. The loss is directional, has a closed form and scales with discarded variance. In the kernel case the same ladder holds, but granular balls offer no edge over k-means landmarks.", { placeholder: "body" });

(async () => {
  const out = process.argv[2];
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
})();
