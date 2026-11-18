// Session 9 results and recommendations deck. Run: node build_reco_deck.js out.pptx
// (needs SKILL_DIR and pptxgenjs on NODE_PATH; same theme as docs/midsem/build_deck.js)
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
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "GBLSTSVM: Session 9 results and recommendations";
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

const SECTION = "Session 9";
pres.addSection({ title: SECTION });
function content(title, kicker) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: SECTION });
  s.addText(title, { placeholder: "title" });
  if (kicker) s.addText(kicker, { placeholder: "kicker" });
  return s;
}
const T = (o) => Object.assign({ isTextBox: true, margin: 0, color: C.text1, valign: "top" }, o);
function bullets(items, size) {
  return items.map((t, i) => {
    const o = { bullet: true, breakLine: i < items.length - 1 };
    if (size) o.fontSize = size;
    return { text: t, options: o };
  });
}
function card(s, x, y, w, h, head, body, color, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "card " + head });
  s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.24, w: 0.22, h: 0.22, fill: { color }, line: { color } });
  s.addText(head, T({ x: x + 0.55, y: y + 0.12, w: w - 0.7, h: 0.46, fontSize: 16, bold: true, color: C.text2, valign: "middle" }));
  s.addText(body, T({ x: x + 0.22, y: y + 0.66, w: w - 0.44, h: h - 0.8, fontSize: opts.bs || 14, paraSpaceAfter: 4 }));
}
function stat(s, x, y, w, num, label, color, size = 40) {
  s.addText(num, T({ x, y, w, h: 0.8, fontSize: size, bold: true, color, fontFace: THEME.headFontFace }));
  s.addText(label, T({ x, y: y + 0.82, w, h: 0.9, fontSize: 13 }));
}
function box(s, x, y, w, h, text, fill, fs = 13) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: "box " + text.slice(0, 20) });
  s.addText(text, T({ x: x + 0.1, y, w: w - 0.2, h, fontSize: fs, color: C.background1, bold: true, align: "center", valign: "middle" }));
}
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, line: { color: HEX.gray, width: 1.75, endArrowType: "triangle" } });
}
const hdr = { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13 };
function table(s, rows, x, y, w, colW, hl = [], fs = 13) {
  const data = rows.map((r, i) => r.map((c) => ({ text: String(c), options: i === 0 ? Object.assign({}, hdr, { fontSize: fs }) : { fontSize: fs, color: C.text1, fill: { color: hl.includes(i) ? "E3F0E8" : (i % 2 ? "FFFFFF" : "EEF3F6") } } })));
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

// 1. Title
let s = pres.addSlide({ masterName: "DARK", sectionTitle: SECTION });
s.addText("Session 9: paper evidence and next steps", { placeholder: "title" });
s.addText("Held-out ladder, closed-form test, deletion cost, and a reproducibility bug\nGBLSTSVM project · November 2026", { placeholder: "body" });

// 2. What we ran + bottom line
s = content("What we ran, and the bottom line", "The coding steps from Session 8's plan; 17 datasets × 10 seeds throughout");
table(s, [["Step", "What", "Outcome"],
  ["1", "Full ladder with held-out tuning, both scalings, 3 split draws", "Direction robust; significance only in standardized"],
  ["2", "Does ‖M⁻¹ΔM‖ predict the angle before solving?", "Yes: Spearman 0.78–0.84, beats η"],
  ["3", "Per-dataset cost of del_ball", "26% of data dropped; costs 1–2 F1 points"],
  ["4", "θ–η and θ–q figures for the paper", "Done, both scalings"],
  ["+", "Found while checking step 1", "Split seed was not deterministic: fixed"]], 0.6, 1.6, 12.1, [0.7, 6.0, 5.4], [5], 14);
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.6, w: 12.1, h: 1.6, rectRadius: 0.08, fill: { color: "FBE9DD" }, line: { color: "FBE9DD" }, objectName: "bottom line" });
s.addText("The analytical-paper thesis holds. Its classification claims need restating: use standardized coordinates, and report split-draw stability next to every p-value.", T({ x: 0.9, y: 4.6, w: 11.5, h: 1.6, fontSize: 18, bold: true, color: C.text2, valign: "middle" }));

// 3. Bug
s = content("Bug: the 'deterministic' split changed on every run", "Found because RAW, which has no randomness, picked a different c in 114 of 170 runs");
const chain = [["seed = hash(\"dataset|seed|…\")", HEX.navy], ["Python salts str hashing per process", HEX.red], ["New 80/20 selection/validation split each run", HEX.red], ["Different tuned c, different F1", HEX.red]];
chain.forEach((c, i) => { box(s, 0.6 + i * 3.1, 1.7, 2.8, 1.1, c[0], c[1], 14); if (i < 3) arrow(s, 3.42 + i * 3.1, 2.25, 3.68 + i * 3.1, 2.25); });
card(s, 0.6, 3.2, 6.0, 3.3, "Impact", bullets(["Every held-out tuning result since Session 6 is one random split draw", "Validation-selected K, the principled K rule and Session 8's E2 cannot be reproduced exactly", "Single-draw Holm p-values can be split luck (next slides)"]), HEX.red);
card(s, 6.9, 3.2, 5.8, 3.3, "Fix", bullets(["deterministic_rng_seed now uses sha256", "Seeds verified identical across processes", "Pre-fix runs archived as extra split draws", "Two old one-off scripts still use hash(); flagged, not fixed"]), HEX.green);

// 4. Held-out ladder
s = content("Held-out ladder: the direction holds in both scalings", "Test macro-F1, code rule, held-out c per arm; P = partition path (no deletion), D = deletion path");
const arms = ["RAW = P_FULL", "D_FULL", "P_DIAG", "P_ISO", "P_L1", "D_L1", "P_L0", "Published"];
s.addChart(pres.charts.BAR, [
  { name: "Unscaled", labels: arms, values: [0.786, 0.775, 0.765, 0.740, 0.755, 0.746, 0.732, 0.721] },
  { name: "Standardized", labels: arms, values: [0.829, 0.813, 0.808, 0.808, 0.777, 0.746, 0.753, 0.780] },
], Object.assign(chartOpts("Mean test macro-F1 (17 datasets × 10 seeds)", { barDir: "col", chartColors: [HEX.gray, HEX.teal], showLegend: true, legendPos: "b", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 9, valAxisMinVal: 0.65, valAxisMaxVal: 0.86 }), { x: 0.6, y: 1.6, w: 8.0, h: 5.0 }));
s.addText(bullets(["Exact moments (P_FULL) reproduce RAW to the last digit", "Each contrast keeps its sign in every split draw", "Standardizing lifts every arm except D_L1 and widens the gaps", "Published GBLSTSVM sits 5–6 points below RAW"], 15), T({ x: 8.9, y: 1.7, w: 3.8, h: 4.8, paraSpaceAfter: 10 }));

// 5. Stability
s = content("Which classification claims survive split re-draws", "Δ F1 on the standard draw (Holm p); 'significant draws' = Holm p < 0.05 across independent split draws");
table(s, [["Contrast", "Standardized Δ (Holm p)", "Significant draws", "Unscaled Δ", "Significant draws"],
  ["Scatter on deletion path (D_FULL − D_L1)", "+0.066 (0.005)", "4 of 4", "+0.029", "1 of 5"],
  ["RAW vs published", "+0.049 (0.003)", "4 of 4", "+0.065", "1 of 5"],
  ["Exact cost of deletion (RAW − D_FULL)", "+0.016 (0.026)", "2 of 4", "+0.011", "0 of 5"],
  ["Population weights (P_L1 − P_L0)", "+0.024 (0.28)", "1 of 4", "+0.024", "1 of 5"],
  ["Diagonal scatter (P_DIAG − P_L1)", "+0.031 (0.29)", "0 of 4", "+0.009", "0 of 5"],
  ["Off-diagonal scatter (P_FULL − P_DIAG)", "+0.021 (0.29)", "0 of 4", "+0.021", "0 of 5"],
  ["DIAG vs ISO", "0.000 (0.89)", "0 of 4", "+0.025", "1 of 5"]], 0.6, 1.6, 12.1, [4.4, 2.3, 1.8, 1.7, 1.9], [1, 2, 3], 13);
s.addText("Session 8's \"+0.048, Holm 0.018\" was the one lucky unscaled draw. Every sign is the same in every draw (except DIAG vs ISO standardized); significance is the fragile part.", T({ x: 0.6, y: 4.95, w: 12.1, h: 1.0, fontSize: 15, bold: true, color: C.text2 }));

// 6. Closed form
s = content("Closed form: exact, bounded, and predictive before solving", "z_arm = (I + M_arm⁻¹ΔM) z_RAW, so ‖z_arm − z_RAW‖ / ‖z_RAW‖ ≤ q = ‖M_arm⁻¹ΔM‖₂, computed from ball moments only");
s.addImage({ path: path.join(FIG, "theta_vs_q_std.png"), x: 0.6, y: 1.6, w: 8.6, h: 2.78, altText: "Angle to RAW against log q for L1, ISO and DIAG, standardized" });
stat(s, 9.5, 1.55, 3.2, "2e-8", "max identity residual over 6,120 rows", HEX.green, 36);
stat(s, 9.5, 3.15, 3.2, "100%", "of rows satisfy the bound", HEX.green, 36);
s.addText(bullets(["θ rises with q inside 15–17 of 17 datasets for every arm (median within-dataset Spearman 0.94–1.0)", "The bound is loose (median error / q = 0.002–0.34), so q ranks settings rather than predicting the angle value", "q does not order arms within one setting (49% agreement standardized): don't claim that"], 14), T({ x: 0.6, y: 4.6, w: 12.1, h: 2.0, paraSpaceAfter: 6 }));
note(s, "Standardized coordinates; colour = ρ ∈ {0.6 … 1.0}. Unscaled figure in experiments/output/a9_paper_evidence/figures.");

// 7. q vs eta
s = content("q beats η as a predictor in 5 of 6 cells", "Spearman correlation with θ to RAW, over dataset × ρ cells");
s.addChart(pres.charts.BAR, [
  { name: "q = ‖M⁻¹ΔM‖ (pre-fit)", labels: ["L1 std", "ISO std", "DIAG std", "L1 unscaled", "ISO unscaled", "DIAG unscaled"], values: [0.78, 0.84, 0.80, 0.82, 0.69, 0.39] },
  { name: "η (discarded variance)", labels: ["L1 std", "ISO std", "DIAG std", "L1 unscaled", "ISO unscaled", "DIAG unscaled"], values: [0.65, 0.44, 0.39, 0.76, 0.35, 0.67] },
], Object.assign(chartOpts("Spearman with θ", { barDir: "col", chartColors: [HEX.teal, HEX.gray], showLegend: true, legendPos: "b", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1 }), { x: 0.6, y: 1.6, w: 7.6, h: 5.0 }));
card(s, 8.5, 1.6, 4.2, 5.0, "Why it matters", bullets(["η only describes center-only balls; q covers every rung, including ISO and DIAG", "q comes straight from the algebra, so the paper gets an identity, a bound and a predictor", "Exception: DIAG unscaled, where η still wins"], 14), HEX.green);

// 8. Deletion
s = content("del_ball discards a quarter of the training set", "Mean fraction of training samples removed by the n < 2 filter, per dataset (ρ = 0.9)");
const dl = [["heart", 0.491], ["messidor", 0.468], ["german_credit", 0.462], ["credit_approval", 0.442], ["australian", 0.423], ["breast_cancer", 0.387], ["wholesale", 0.355], ["diabetes", 0.349], ["chronic_kidney", 0.317], ["spambase", 0.219], ["tic_tac_toe", 0.213], ["vehicle0", 0.106], ["ionosphere", 0.102], ["bc_wisc_diag", 0.079], ["voting", 0.041], ["ecoli2", 0.038], ["bc_wisc_orig", 0.0]];
s.addChart(pres.charts.BAR, [{ name: "Dropped", labels: dl.map((d) => d[0]), values: dl.map((d) => d[1]) }], Object.assign(chartOpts("Share of training samples dropped", { barDir: "bar", chartColors: [HEX.red], dataLabelPosition: "outEnd", dataLabelFormatCode: "0%", catAxisOrientation: "maxMin", valAxisMaxVal: 0.6, valAxisLabelFormatCode: "0%", dataLabelFontSize: 9, catAxisLabelFontSize: 10 }), { x: 0.6, y: 1.5, w: 7.0, h: 5.2 }));
stat(s, 8.0, 1.6, 4.7, "26%", "dropped on average (minority 30%, majority 28%)", HEX.red, 40);
stat(s, 8.0, 3.3, 4.7, "+0.016 F1", "exact cost of deletion, standardized (Holm 0.026; up to +0.08 on heart)", HEX.red, 32);
s.addText("Correction: Session 8's \"~3 F1 points\" was one draw of the center-only contrast. The robust figure is 1–2 points exactly.", T({ x: 8.0, y: 5.2, w: 4.7, h: 1.3, fontSize: 14, italic: true, color: C.text2 }));

// 9. Recommendations
s = content("Recommendations", "What the paper should say, and what to run before submitting");
const rec = [
  ["Standardized is primary", "Use it for angles and classification; unscaled becomes the robustness check."],
  ["Report split stability", "Sign agreement and significant draws next to every contrast; 17 datasets is too few for single-draw p-values."],
  ["Lead with three robust results", "Scatter on the deletion path, the exact cost of deletion, RAW vs published."],
  ["Promote the closed form", "State it as a proposition: identity, bound, and q as a pre-fit ranking predictor."],
  ["Re-run Session 6–7 K-selection", "With the fixed seed, before citing any of those p-values."],
  ["Drop coordinate-dependent claims", "ISO vs DIAG and ISO vs L1 flip with scaling, for both angles and F1."],
];
rec.forEach((r, i) => card(s, 0.6 + (i % 2) * 6.15, 1.55 + Math.floor(i / 2) * 1.7, 5.95, 1.55, `${i + 1}. ${r[0]}`, r[1], [HEX.teal, HEX.teal, HEX.green, HEX.green, HEX.orange, HEX.red][i], { bs: 13 }));

// 10. Next steps
s = content("What's left before writing", "Coding is done; the remaining work is reading and proving");
const left = [
  ["Literature review", "Does any granular-ball paper state the moment identity? This decides the novelty claim.", HEX.orange],
  ["Positioning", "Compare against Nyström, reduced-kernel LSTSVM and coreset methods; the kernel section depends on it.", HEX.teal],
  ["Proof", "Exactness for the regularized, weighted variant the code actually solves (ridge, radius targets, class weights).", HEX.teal],
  ["Optional compute run", "One large kernel dataset (n ≈ 20k) to settle where granular balls stop costing more than they save.", HEX.gray],
];
left.forEach((l, i) => {
  const y = 1.6 + i * 1.25;
  box(s, 0.6, y, 2.8, 1.0, l[0], l[2], 15);
  s.addText(l[1], T({ x: 3.7, y, w: 9.0, h: 1.0, fontSize: 16, valign: "middle" }));
});
note(s, "Full write-up: docs/Sessions/Session_9_Paper_Evidence_and_Split_Bug.md · Data: experiments/output/a9_paper_evidence/");

(async () => {
  const out = process.argv[2];
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
})();
