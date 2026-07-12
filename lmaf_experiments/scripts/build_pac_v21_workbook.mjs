import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");

const args = parseArgs(process.argv.slice(2));
const reportDir = path.resolve(ROOT, args.reportDir || "results/reports/pac_v21_all_summary");
const payloadPath = path.join(reportDir, "workbook_payload.json");
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Dashboard");
dashboard.showGridLines = false;
await writeDashboard(dashboard, payload, reportDir);

for (const sheetSpec of payload.sheets || []) {
  const sheet = workbook.worksheets.add(safeSheetName(sheetSpec.name));
  sheet.showGridLines = false;
  writeRows(sheet, sheetSpec.rows || []);
}

const inspect = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 4000,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const previewDir = path.join(reportDir, "workbook_previews");
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["Dashboard", "ModelOverall", "SubsetModel", "A_Position", "B_Interference", "C_Binding", "D_Multihop"]) {
  try {
    const preview = await workbook.render({
      sheetName,
      autoCrop: "all",
      scale: 1,
      format: "png",
    });
    await fs.writeFile(
      path.join(previewDir, `${sheetName}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  } catch (err) {
    console.warn(`Could not render ${sheetName}: ${err.message}`);
  }
}

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(reportDir, "pac_v21_summary_tables.xlsx");
await output.save(outputPath);
console.log(`Wrote workbook to ${outputPath}`);

async function writeDashboard(sheet, payload, reportDir) {
  const rows = [
    ["PAC v2.1 Result Summary", ""],
    ["Generated at", payload.generated_at || ""],
    ["Scope", "Current PAC v2.1 queue results with coverage-aware metrics"],
    ["Scoring", "accuracy_success excludes API errors; accuracy_all_conservative counts API errors as zero"],
    ["", ""],
    ["Figure", "Relative path"],
  ];
  for (const [key, relPath] of Object.entries(payload.figures || {})) rows.push([key, relPath]);
  writeRows(sheet, rows);
  sheet.getRange("A1:B1").format = {
    fill: "#1F2937",
    font: { bold: true, color: "#FFFFFF", size: 14 },
  };
  sheet.getRange("A6:B6").format = {
    fill: "#E5E7EB",
    font: { bold: true, color: "#111827" },
  };
  sheet.getRange("D1:J1").values = [["Key Figures", null, null, null, null, null, null]];
  sheet.getRange("D1:J1").format = {
    fill: "#1F2937",
    font: { bold: true, color: "#FFFFFF", size: 14 },
  };

  const embed = [
    "fig_subset_accuracy",
    "fig_subset_coverage",
    "fig_pac_b_interference",
    "fig_pac_c_binding",
    "fig_pac_d_multihop",
  ];
  let row = 2;
  for (const key of embed) {
    const relPath = payload.figures?.[key];
    if (!relPath) continue;
    sheet.getRangeByIndexes(row - 1, 3, 1, 1).values = [[key]];
    await addImage(sheet, path.join(reportDir, relPath), row, 4, 720, 430).catch((err) => {
      console.warn(`Could not embed ${key}: ${err.message}`);
    });
    row += 23;
  }
  sheet.freezePanes.freezeRows(1);
}

function writeRows(sheet, rows) {
  if (!rows.length) {
    sheet.getRange("A1").values = [["No data"]];
    return;
  }
  const width = Math.max(...rows.map((r) => r.length));
  const normalized = rows.map((r) => {
    const next = [...r];
    while (next.length < width) next.push(null);
    return next.map((value) => normalizeCell(value));
  });
  const range = sheet.getRangeByIndexes(0, 0, normalized.length, width);
  range.values = normalized;
  const header = sheet.getRangeByIndexes(0, 0, 1, width);
  header.format = {
    fill: "#2563EB",
    font: { bold: true, color: "#FFFFFF" },
  };
  range.format.borders = { preset: "inside", style: "thin", color: "#E5E7EB" };
  range.format.autofitColumns();
  sheet.freezePanes.freezeRows(1);
}

async function addImage(sheet, imagePath, row, col, widthPx, heightPx) {
  const bytes = await fs.readFile(imagePath);
  const dataUrl = `data:image/png;base64,${Buffer.from(bytes).toString("base64")}`;
  sheet.images.add({
    dataUrl,
    anchor: {
      from: { row: row - 1, col: col - 1 },
      extent: { widthPx, heightPx },
    },
  });
}

function normalizeCell(value) {
  if (value === undefined) return null;
  if (typeof value === "number" && Number.isNaN(value)) return null;
  return value;
}

function safeSheetName(name) {
  return String(name).replace(/[\[\]:*?/\\]/g, "_").slice(0, 31) || "Sheet";
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--report-dir") out.reportDir = argv[++i];
  }
  return out;
}
