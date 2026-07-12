import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");

const args = parseArgs(process.argv.slice(2));
const reportDir = path.resolve(ROOT, args.reportDir || "results/reports/public_benchmarks_summary");
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

const preview = await workbook.render({
  sheetName: "Dashboard",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(reportDir, "workbook_dashboard_preview.png"),
  new Uint8Array(await preview.arrayBuffer()),
);

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(reportDir, "summary_tables.xlsx");
await output.save(outputPath);
console.log(`Wrote workbook to ${outputPath}`);

async function writeDashboard(sheet, payload, reportDir) {
  const rows = [
    ["Public Benchmark Summary", ""],
    ["Generated at", payload.generated_at || ""],
    ["Scope", "LongBench, NIAH, and RULER public benchmark results"],
    ["Scoring note", "Main scores use successful API calls; coverage/error_rate tracks API failures separately."],
    ["", ""],
    ["Figure", "Relative path"],
  ];
  const figures = Object.entries(payload.figures || {});
  for (const [key, relPath] of figures) rows.push([key, relPath]);
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
    "fig_longbench_task_heatmap",
    "fig_niah_subtask_heatmap",
    "fig_ruler_task_heatmap",
  ];
  let row = 2;
  let col = 4;
  for (const key of embed) {
    const relPath = payload.figures?.[key];
    if (!relPath) continue;
    sheet.getRangeByIndexes(row - 1, col - 1, 1, 1).values = [[key]];
    await addImage(sheet, path.join(reportDir, relPath), row, col, 650, 360).catch((err) => {
      console.warn(`Could not embed ${key}: ${err.message}`);
    });
    row += 19;
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
  sheet.freezePanes.freezeRows(1);
  try {
    range.format.autofitColumns();
    range.format.autofitRows();
  } catch {
    // Autofit is best-effort; the workbook remains valid without it.
  }
  for (let c = 0; c < width; c++) {
    const name = String(normalized[0][c] || "").toLowerCase();
    if (name.includes("score") || name.includes("rate") || name.includes("coverage") || name.includes("accuracy")) {
      const colRange = sheet.getRangeByIndexes(1, c, Math.max(1, normalized.length - 1), 1);
      colRange.format.numberFormat = "0.000";
    }
  }
}

async function addImage(sheet, filePath, row, col, widthPx, heightPx) {
  const bytes = await fs.readFile(filePath);
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
  sheet.images.add({
    dataUrl,
    anchor: {
      from: { row, col },
      extent: { widthPx, heightPx },
    },
  });
}

function normalizeCell(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "boolean") return value;
  return String(value);
}

function safeSheetName(name) {
  return String(name).replace(/[\\/?*:[\]]/g, "_").slice(0, 31) || "Sheet";
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--report-dir") out.reportDir = argv[++i];
  }
  return out;
}
