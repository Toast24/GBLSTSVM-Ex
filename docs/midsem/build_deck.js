const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

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

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "GBLSTSVM Extensions: Midsem Progress Report";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6, fontSize: 40, bold: true, color: C.background1, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 11.7, h: 1.6, fontSize: 18, color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5 },
});

let section = "";
function sec(t) { section = t; pres.addSection({ title: t }); }
function content(title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function card(s, x, y, w, h, head, body, color, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "card " + head });
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.28, w: 0.22, h: 0.22, fill: { color }, line: { color } });
  s.addText(head, { x: x + 0.6, y: y + 0.15, w: w - 0.8, h: 0.5, fontSize: opts.hs || 17, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
  s.addText(body, { x: x + 0.25, y: y + 0.7, w: w - 0.5, h: h - 0.85, fontSize: opts.bs || 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 4 });
}
function bullets(items) { return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })); }
function stat(s, x, y, w, num, label, color) {
  s.addText(num, { x, y, w, h: 0.9, fontSize: 44, bold: true, color, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  s.addText(label, { x, y: y + 0.9, w, h: 0.8, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}
const hdr = { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13 };
function table(s, rows, x, y, w, colW, hl = []) {
  const data = rows.map((r, i) => r.map((c) => ({ text: String(c), options: i === 0 ? hdr : { fontSize: 13, color: C.text1, fill: { color: hl.includes(i) ? "FBE9DD" : (i % 2 ? "FFFFFF" : "EEF3F6") } } })));
  s.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: "D5DEE5" }, margin: 0.06 });
}

// ---------- Title
sec("Introduction");
let s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
s.addText("Extending Granular-Ball LSTSVM", { placeholder: "title" });
s.addText("Midsem progress report: from heuristic fixes to a sufficient-statistic account of granular balls\nBharatSOP research project · October 2026", { placeholder: "body" });

// ---------- Problem
s = content("What we set out to improve");
s.addText([
  { text: "GBLSTSVM", options: { bold: true, breakLine: true } },
  { text: "Replaces training samples with granular balls (center + radius), then solves a least-squares twin SVM on the balls.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "Original question", options: { bold: true, breakLine: true } },
  { text: "Can we make ball construction and ball weighting more reliable under class imbalance, sparse minority support and unstable topology?" },
], { x: 0.6, y: 1.5, w: 5.6, h: 4.5, fontSize: 16, color: C.text1, valign: "top", isTextBox: true });
const steps = ["Granulate\nKMeans(k=2) splits until purity ≥ ρ", "Filter\ndel_ball drops balls with n < 2", "Weight\nCA / Topo / reliability weights", "Classify\nTwin LSTSVM on ball centers"];
steps.forEach((t, i) => {
  const y = 1.5 + i * 1.3;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y, w: 5.7, h: 1.05, rectRadius: 0.08, fill: { color: i === 1 ? "FBE9DD" : C.background2 }, line: { color: C.background2 }, objectName: "pipeline " + i });
  s.addShape(pres.shapes.OVAL, { x: 7.2, y: y + 0.27, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  s.addText(String(i + 1), { x: 7.2, y: y + 0.27, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  const [a, b] = t.split("\n");
  s.addText([{ text: a, options: { bold: true, breakLine: true } }, { text: b }], { x: 7.9, y, w: 4.7, h: 1.05, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
});
s.addNotes("The baseline pipeline has four stages. Step 2, ball deletion, later turned out to matter a lot.");

// ---------- Timeline
s = content("Eight research sessions, three phases");
const phases = [
  ["Phase A · Sessions 1–3", "Weighting extensions", "Reproduced GBLSTSVM; built CW, R, CR, CA, BMRU, OBRU, Topo and SupportAware-Topo weighting.", C.accent2],
  ["Phase B · Sessions 4–7", "Granulation diagnostics", "Traced failures upstream to granulation; split-utility gates, entropy schedulers, compression and re-granulation ablations.", C.accent1],
  ["Phase C · Session 8 + proposal", "Sufficient-statistic theory", "Showed that center-only balls are a lossy, quantifiable approximation of an exact, much cheaper estimator.", C.accent3],
];
phases.forEach((p, i) => card(s, 0.6 + i * 4.1, 1.6, 3.85, 4.6, p[0], [{ text: p[1], options: { bold: true, breakLine: true, fontSize: 16 } }, { text: " ", options: { breakLine: true } }, { text: p[2] }], p[3], { hs: 15 }));
s.addText("Each phase closed with a falsification test, and each verdict set the next question.", { x: 0.6, y: 6.4, w: 12, h: 0.4, fontSize: 14, italic: true, color: C.accent5, isTextBox: true, margin: 0 });

// ---------- Phase A
sec("Phase A: Weighting");
s = content("Phase A: weighting schemes gave weak, inconsistent gains");
card(s, 0.6, 1.5, 5.9, 2.4, "CW / R / CR (Sessions 1–2)", bullets(["CR weights differed numerically from CW but rarely changed predictions", "Under the frozen protocol, diabetes collapsed to majority-only predictions (minority recall 0)", "CR added nothing beyond class weighting"]), C.accent4);
card(s, 0.6, 4.1, 5.9, 2.4, "Topo / SupportAware-Topo (Session 3)", bullets(["Topo helped when the ball graph was dense (vehicle0: +0.05 minority F1)", "It failed when opposite-class support was sparse (ecoli2: −0.22 minority F1)", "Shrinking R_opp fixed the failures but erased the wins"]), C.accent2);
table(s, [["Dataset", "Δ G-mean", "Δ minority F1", "Verdict"], ["diabetes", "0.000", "0.000", "Neutral"], ["vehicle0", "+0.030", "+0.052", "Improved"], ["ecoli2", "−0.115", "−0.220", "Failed"]], 6.9, 1.6, 5.8, [1.6, 1.3, 1.6, 1.3], [3]);
s.addText("Topo vs CA on the three Phase-7 datasets (test)", { x: 6.9, y: 3.4, w: 5.8, h: 0.35, fontSize: 11, color: C.accent5, isTextBox: true, margin: 0 });
s.addText("Lesson: the weights could not fix what granulation had already removed.", { x: 6.9, y: 4.3, w: 5.8, h: 1.2, fontSize: 20, bold: true, color: C.text2, fontFace: THEME.headFontFace, isTextBox: true, margin: 0 });

// ---------- Phase B
sec("Phase B: Granulation");
s = content("Phase B: the failure starts inside granulation");
stat(s, 0.6, 1.6, 2.8, "25%", "of KMeans splits isolate minority mass (17 datasets, 13,654 splits)", C.accent4);
stat(s, 3.7, 1.6, 2.8, "80%", "of isolated children become singletons that del_ball removes", C.accent4);
stat(s, 6.8, 1.6, 2.8, "0.864", "AUROC of center-radius separation for flagging harmful splits", C.accent1);
stat(s, 9.9, 1.6, 2.8, "0.58", "best AUROC for repairing singletons from parent-local features", C.accent5);
card(s, 0.6, 4.0, 12.1, 2.6, "What we tried next (Sessions 5–7)", bullets([
  "Split-utility gates and entropy schedulers: online entropy rejected (>50% collapse)",
  "Compression at small K raised macro F1 (+0.039 vs full ρ=0.9, p=0.002), but ball ranking added little",
  "Re-granulation vs singletons at a fixed sample budget: the gap came from class-weight semantics, not ball structure",
  "Verdict (Session 7, CASE B): stop the granular-heuristic branch",
]), C.accent1);

// ---------- Phase C: idea
sec("Phase C: Theory");
s = content("Phase C: balls that store moments are exact");
s.addText([
  { text: "LSTSVM depends on the data only through second-moment sums, HᵀH and GᵀG.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "If each ball stores its count, mean and scatter matrix, the solution equals the full-data solution exactly, for any granulation.", options: { bold: true, breakLine: true } },
  { text: " ", options: { breakLine: true } },
  { text: "The published GBLSTSVM keeps only center and radius, so it discards within-ball scatter." },
], { x: 0.6, y: 1.5, w: 5.8, h: 4.8, fontSize: 17, color: C.text1, valign: "top", isTextBox: true });
const ladder = [["L0", "center only"], ["L1", "center + population (published-style)"], ["ISO", "isotropic scatter"], ["DIAG", "diagonal scatter"], ["FULL / L3", "full scatter ≡ RAW"]];
s.addText("Information ladder of ball summaries", { x: 7.0, y: 1.5, w: 5.7, h: 0.45, fontSize: 16, bold: true, color: C.text2, isTextBox: true, margin: 0 });
ladder.forEach((l, i) => {
  const y = 6.0 - i * 0.85, w = 2.6 + i * 0.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y, w, h: 0.7, rectRadius: 0.06, fill: { color: i === 4 ? C.accent3 : C.accent1 }, line: { color: C.background1 }, objectName: "ladder " + l[0] });
  s.addText([{ text: l[0] + "  ", options: { bold: true } }, { text: l[1] }], { x: 7.15, y, w: w - 0.2, h: 0.7, fontSize: 13, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
});

// ---------- Gate
s = content("Session 8 gate: the 23° error was deletion, not geometry");
table(s, [["FULL arm (A7 angle definition)", "θ vs RAW", "θ vs RAW on retained"], ["Partition path, shared c", "0.000°", "16.4°"], ["Partition path, RAW held-out c", "0.000°", "15.0°"], ["Deletion path, shared c", "16.8°", "0.000°"], ["Deletion path, RAW held-out c", "15.5°", "0.000°"]], 0.6, 1.6, 7.2, [3.6, 1.6, 2.0], [1, 2]);
stat(s, 8.4, 1.5, 4.3, "26%", "of training samples are silently dropped by the n<2 deletion step (up to 49% on heart)", C.accent4);
stat(s, 8.4, 3.6, 4.3, "≈3 F1", "points lost to that deletion alone", C.accent4);
s.addText("Every earlier comparison against RAW mixed compression with deletion and c mismatch. The partition path reproduces RAW to machine precision.", { x: 0.6, y: 4.6, w: 7.2, h: 1.5, fontSize: 15, color: C.text1, isTextBox: true, margin: 0 });

// ---------- Dose-response chart
s = content("Center-only error grows with discarded variance");
s.addChart(pres.charts.BAR, [
  { name: "Mean θ unscaled", labels: ["FULL", "DIAG", "L1", "L0", "ISO"], values: [0, 18.3, 23.6, 28.8, 40.6] },
  { name: "Mean θ standardized", labels: ["FULL", "DIAG", "L1", "L0", "ISO"], values: [0, 28.8, 38.5, 46.2, 36.8] },
], {
  x: 0.6, y: 1.5, w: 7.0, h: 5.0, barDir: "col", chartColors: ["1F6F8B", "E07A3F"],
  showTitle: true, title: "Angle to RAW solution (degrees), partition path", titleFontSize: 14, titleColor: "14324A", titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelColor: "1B2430",
  showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 11,
  catAxisLabelColor: "5C6B78", valAxisLabelColor: "5C6B78", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  valGridLine: { color: "E3E9EE", size: 0.5 }, catGridLine: { style: "none" },
});
s.addText([
  { text: "η = discarded within-ball variance / class variance", options: { bold: true, breakLine: true } },
  { text: "As ρ goes 0.6 → 1.0, η falls 0.81 → 0.12 and θ_L1 falls 62° → 17°.", options: { bullet: true, breakLine: true } },
  { text: "Spearman(θ_L1, η) = 0.75 (p = 2e-19); positive in 17/17 datasets.", options: { bullet: true, breakLine: true } },
  { text: "Closed form: z_L1 = (I + M_L1⁻¹ΔM) z_RAW.", options: { bullet: true, breakLine: true } },
  { text: "DIAG ≪ ISO in both scalings: where restored mass goes matters.", options: { bullet: true } },
], { x: 8.0, y: 1.6, w: 4.7, h: 4.8, fontSize: 15, color: C.text1, valign: "top", isTextBox: true, paraSpaceAfter: 8 });

// ---------- Held-out classification
s = content("With leak-free tuning, scatter restoration helps");
s.addChart(pres.charts.BAR, [{ name: "Test macro-F1", labels: ["RAW", "D_FULL", "P_L1", "D_L1", "P_L0", "Published"], values: [0.785, 0.775, 0.757, 0.727, 0.725, 0.714] }], {
  x: 0.6, y: 1.5, w: 7.0, h: 5.0, barDir: "bar", chartColors: ["1F6F8B"],
  showTitle: true, title: "Test macro-F1, 17 datasets × 10 seeds", titleFontSize: 14, titleColor: "14324A", titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelColor: "1B2430",
  showLegend: false, valAxisMinVal: 0.6, valAxisMaxVal: 0.82, catAxisOrientation: "maxMin",
  catAxisLabelColor: "5C6B78", valAxisLabelColor: "5C6B78", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  valGridLine: { color: "E3E9EE", size: 0.5 }, catGridLine: { style: "none" },
});
s.addText(bullets([
  "Earlier tuning scored validation rows that had been inside the fitted balls; we now re-granulate per fold",
  "D_FULL − D_L1 = +0.048 F1 (Holm p = 0.018, 14/17 wins)",
  "Published GBLSTSVM is 0.071 below RAW (Holm p = 0.03)",
  "Granulation costs ~180 ms per fit vs 0.26 ms for the exact RAW solve (~700×)",
]), { x: 8.0, y: 1.6, w: 4.7, h: 4.8, fontSize: 15, color: C.text1, valign: "top", isTextBox: true, paraSpaceAfter: 8 });

// ---------- Kernel + rule
s = content("Kernel pilot and the prediction-rule gap");
card(s, 0.6, 1.5, 6.0, 5.0, "Kernel pilot (RBF, 17 × 5 seeds)", bullets([
  "The ladder replicates in feature space: L0 < L1 ≈ MEAN < EXACT ≈ RAW",
  "EXACT − L1 = +0.017 F1 (Holm p = 0.03)",
  "k-means landmarks with the same k match granular balls (Δ F1 = +0.0005)",
  "At n ≤ 3.2k, granulation alone costs more than the full O(n³) RAW kernel fit",
]), C.accent3);
card(s, 6.9, 1.5, 5.8, 5.0, "Paper rule vs reference code", bullets([
  "The published decision rule and the repository rule disagree",
  "Disagreement tracks |log ‖w₁‖/‖w₂‖| (Spearman 0.58 unscaled, 0.76 standardized)",
  "Rule choice moves RAW F1 by ≈0.10 (0.785 vs 0.685), more than any representation effect",
]), C.accent2);

// ---------- Engineering
s = content("Engineering work behind the results");
const eng = [
  ["Reproducible pipeline", "Frozen protocols, 17-dataset registry, seeds 0–9, Wilcoxon + bootstrap + Holm correction"],
  ["New modules", "ladder_gblstsvm, margin_multigranularity, svm_aware_granulation; unit tests for each"],
  ["Bug fixes", "pandas 3 dtype check silently dropped 5 of 17 datasets; in-sample validation leak removed"],
  ["Documentation", "Eight session write-ups with hypotheses, verdicts and handoffs, plus the research proposal"],
];
eng.forEach((e, i) => card(s, 0.6 + (i % 2) * 6.15, 1.5 + Math.floor(i / 2) * 2.55, 5.95, 2.3, e[0], e[1], [C.accent1, C.accent3, C.accent4, C.accent2][i]));

// ---------- Status
sec("Status and next steps");
s = content("Where the thesis stands");
card(s, 0.6, 1.5, 6.0, 5.0, "Holds", bullets([
  "Count, mean and scatter per ball give the exact LSTSVM solution (θ = 0.000°)",
  "Center-only error is directional, has a closed form and scales with η",
  "DIAG/FULL ≫ ISO, explained by scatter anisotropy",
  "Scatter restoration improves held-out F1",
  "The rule discrepancy is driven by the norm ratio",
]), C.accent3);
card(s, 6.9, 1.5, 5.8, 5.0, "Revised or dropped", bullets([
  "A7 angle numbers (23°/38°/52°): contaminated by deletion and c mismatch",
  "\"ISO is worse than restoring nothing\": depends on coordinates",
  "\"No classification advantage for scatter\": an in-sample tuning artifact",
  "All A4/A6 classification tables: to be re-run held-out",
]), C.accent4);

s = content("Plan for the second half");
const next = [
  "Re-run A4/A6 with the held-out protocol for the paper tables",
  "Report angles in standardized coordinates, unscaled as robustness check",
  "Plot θ_L1 against η across ρ",
  "Predict θ from ‖M_L1⁻¹ΔM‖ before fitting",
  "Optional: one large kernel dataset (n ≈ 20k) to settle compute cost",
  "Write the analytical paper: linear core plus kernel section",
];
next.forEach((t, i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 1.6;
  s.addShape(pres.shapes.OVAL, { x, y: y + 0.15, w: 0.7, h: 0.7, fill: { color: i === 5 ? C.accent2 : C.accent1 }, line: { color: C.background1 } });
  s.addText(String(i + 1), { x, y: y + 0.15, w: 0.7, h: 0.7, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(t, { x: x + 0.95, y, w: 4.9, h: 1.0, fontSize: 16, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
});

s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
s.addText("Recommendation: an analytical paper", { placeholder: "title" });
s.addText("In the linear case GBLSTSVM is a lossy, deletion-affected approximation of an exact estimator that is ~700× cheaper. In the kernel case the same ladder holds, but granular balls offer no edge over k-means landmarks.", { placeholder: "body" });

(async () => {
  const out = process.argv[2];
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
})();
