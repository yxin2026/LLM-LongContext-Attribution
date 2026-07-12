import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = process.cwd();
const REPORT_DIR = path.join(ROOT, "results", "reports", "longbench_pac");
const TABLE_DIR = path.join(REPORT_DIR, "tables");
const FIGURE_DIR = path.join(REPORT_DIR, "figures");
const OUTPUT_XLSX = path.join(REPORT_DIR, "LongBench_PAC_Report.xlsx");
const PREVIEW_DIR = path.join(REPORT_DIR, "previews");

const TABLE_SHEETS = [
  ["longbench_by_model.csv", "LongBench Models", "LongBenchModels"],
  ["longbench_by_task_model.csv", "LongBench Tasks", "LongBenchTasks"],
  ["longbench_by_category_model.csv", "LongBench Categories", "LongBenchCategories"],
  ["pac_by_subset_model.csv", "PAC Subsets", "PacSubsets"],
  ["pac_A_position_details.csv", "PAC A Position", "PacAPosition"],
  ["pac_B_interference_details.csv", "PAC B Interference", "PacBInterference"],
  ["pac_C_overlap_details.csv", "PAC C Entity", "PacCEntity"],
  ["pac_D_multihop_details.csv", "PAC D Multihop", "PacDMultihop"],
];

const CHART_SHEETS = [
  [
    "LongBench Charts",
    [
      ["longbench_task_heatmap.png", "LongBench score by task and model", 1, 0, 1080, 640],
      ["longbench_category_heatmap.png", "LongBench score by task category", 35, 0, 900, 520],
      ["longbench_model_ranking.png", "LongBench model ranking", 35, 9, 760, 520],
    ],
  ],
  [
    "PAC Overview Charts",
    [
      ["pac_subset_heatmap.png", "PAC accuracy by subset and model", 1, 0, 960, 620],
      ["pac_C_model_ranking.png", "PAC-C model ranking", 34, 0, 760, 480],
    ],
  ],
  [
    "PAC A Charts",
    [
      ["pac_A_position_heatmap.png", "PAC-A position sensitivity", 1, 0, 900, 540],
      ["pac_A_length_heatmap.png", "PAC-A length sensitivity", 1, 9, 900, 540],
    ],
  ],
  [
    "PAC B Charts",
    [
      ["pac_B_density_in_domain.png", "PAC-B in-domain interference", 1, 0, 780, 500],
      ["pac_B_density_out_domain.png", "PAC-B out-domain interference", 1, 8, 780, 500],
      ["pac_B_density_random_noise.png", "PAC-B random-noise interference", 29, 0, 780, 500],
    ],
  ],
  [
    "PAC C-D Charts",
    [
      ["pac_C_condition_heatmap.png", "PAC-C entity-overlap conditions", 1, 0, 820, 520],
      ["pac_D_hops_heatmap.png", "PAC-D hop count", 1, 8, 820, 520],
      ["pac_D_distance_heatmap.png", "PAC-D hop distance", 30, 0, 900, 540],
    ],
  ],
];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];

    if (quoted) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(coerceValue(field));
      field = "";
    } else if (ch === "\n") {
      row.push(coerceValue(field.replace(/\r$/, "")));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(coerceValue(field.replace(/\r$/, "")));
    rows.push(row);
  }
  return rows.filter((items) => items.some((item) => item !== null && item !== ""));
}

function coerceValue(value) {
  const trimmed = value.trim();
  if (trimmed === "") return null;
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
  return trimmed;
}

function columnName(index) {
  let n = index + 1;
  let name = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    name = String.fromCharCode(65 + r) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}

function rangeAddress(startRow, startCol, rowCount, colCount) {
  const first = `${columnName(startCol)}${startRow + 1}`;
  const last = `${columnName(startCol + colCount - 1)}${startRow + rowCount}`;
  return `${first}:${last}`;
}

async function loadCsv(fileName) {
  const text = await fs.readFile(path.join(TABLE_DIR, fileName), "utf8");
  return parseCsv(text.replace(/^\uFEFF/, ""));
}

function styleTitle(range) {
  range.format = {
    fill: "#17324D",
    font: { bold: true, color: "#FFFFFF", size: 15 },
  };
}

function styleHeader(range) {
  range.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
}

function styleTable(range) {
  range.format.borders = { preset: "all", style: "thin", color: "#D7DEE8" };
}

function applyNumberFormats(sheet, rows) {
  const headers = rows[0].map((value) => String(value));
  const rowCount = rows.length;
  for (let col = 0; col < headers.length; col += 1) {
    const header = headers[col];
    const colRange = sheet.getRangeByIndexes(1, col, Math.max(rowCount - 1, 1), 1);
    if (header.startsWith("n_")) {
      colRange.format.numberFormat = "0";
    } else if (
      header.includes("score") ||
      header.includes("coverage") ||
      header.includes("error_rate") ||
      header.includes("f1") ||
      header.includes("rouge")
    ) {
      colRange.format.numberFormat = "0.000";
    } else if (header.includes("latency")) {
      colRange.format.numberFormat = "0.00";
    }
  }
}

function fitColumns(sheet, rows) {
  const headers = rows[0].map((value) => String(value));
  const rowCount = rows.length;
  for (let col = 0; col < headers.length; col += 1) {
    const header = headers[col];
    let width = 96;
    if (["model", "task", "category", "subset", "subtask", "interference_type", "condition"].includes(header)) {
      width = 160;
    } else if (header.includes("latency")) {
      width = 105;
    } else if (header.startsWith("n_")) {
      width = 80;
    } else if (header.includes("score") || header.includes("coverage") || header.includes("error")) {
      width = 92;
    }
    sheet.getRangeByIndexes(0, col, rowCount, 1).format.columnWidthPx = width;
  }
}

function setColumnWidths(sheet, widths, rowCount = 80) {
  for (let col = 0; col < widths.length; col += 1) {
    sheet.getRangeByIndexes(0, col, rowCount, 1).format.columnWidthPx = widths[col];
  }
}

async function addTableSheet(workbook, fileName, sheetName, tableName) {
  const rows = await loadCsv(fileName);
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  const address = rangeAddress(0, 0, rows.length, rows[0].length);
  const range = sheet.getRange(address);
  range.values = rows;
  styleHeader(sheet.getRangeByIndexes(0, 0, 1, rows[0].length));
  styleTable(range);
  fitColumns(sheet, rows);
  applyNumberFormats(sheet, rows);
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add(address, true, tableName);
  table.showFilterButton = true;
  table.showBandedRows = true;
  return { sheet, rows };
}

async function imageDataUrl(fileName) {
  const bytes = await fs.readFile(path.join(FIGURE_DIR, fileName));
  return `data:image/png;base64,${bytes.toString("base64")}`;
}

async function addImage(sheet, fileName, title, row, col, widthPx, heightPx) {
  sheet.getRangeByIndexes(row, col, 1, 4).values = [[title, null, null, null]];
  styleTitle(sheet.getRangeByIndexes(row, col, 1, 4));
  sheet.images.add({
    dataUrl: await imageDataUrl(fileName),
    anchor: {
      from: { row: row + 1, col },
      extent: { widthPx, heightPx },
    },
  });
}

async function addChartSheet(workbook, sheetName, charts) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  for (const chart of charts) {
    await addImage(sheet, ...chart);
  }
}

function summarizeModelRows(longbenchRows, pacRows) {
  const lb = longbenchRows.slice(1);
  const pac = pacRows.slice(1);
  const scoreByModel = new Map(lb.map((row) => [row[0], row[6]]));
  const pacByModel = new Map();
  for (const row of pac) {
    const model = row[0];
    const count = Number(row[3] ?? 0);
    const weighted = Number(row[8] ?? 0) * count;
    const current = pacByModel.get(model) ?? { total: 0, weighted: 0 };
    current.total += count;
    current.weighted += weighted;
    pacByModel.set(model, current);
  }

  return [...scoreByModel.keys()].sort().map((model) => {
    const pacItem = pacByModel.get(model) ?? { total: 0, weighted: 0 };
    return [
      model,
      300,
      Number(scoreByModel.get(model) ?? 0),
      600,
      pacItem.total ? pacItem.weighted / pacItem.total : null,
    ];
  });
}

async function addOverview(workbook) {
  const sheet = workbook.worksheets.add("Overview");
  sheet.showGridLines = false;
  sheet.getRange("A1:F1").merge();
  sheet.getRange("A1:F1").values = [["LongBench / PAC Experiment Report"]];
  styleTitle(sheet.getRange("A1:F1"));
  sheet.getRange("A3:F6").values = [
    ["Item", "Value", "Models", "Per-model target", "Source", "Notes"],
    ["LongBench", 2700, 9, 300, "official_budget_topup_main", "Grouped by task and task category"],
    ["PAC-Test", 5400, 9, 600, "official_budget_topup_main", "Grouped by A/B/C/D subsets and controlled variables"],
    ["Generated", new Date().toISOString().slice(0, 19).replace("T", " "), null, null, "scripts/summarize_longbench_pac.py", "CSV and PNG assets are under results/reports/longbench_pac"],
  ];
  styleHeader(sheet.getRange("A3:F3"));
  styleTable(sheet.getRange("A3:F6"));
  sheet.getRange("A3:F6").format.wrapText = true;
  setColumnWidths(sheet, [150, 110, 90, 125, 210, 300, 24, 24, 180, 120, 120, 120], 60);

  const longbenchRows = await loadCsv("longbench_by_model.csv");
  const pacRows = await loadCsv("pac_by_subset_model.csv");
  const modelSummary = [
    ["model", "longbench_rows", "longbench_score", "pac_target_rows", "pac_weighted_accuracy"],
    ...summarizeModelRows(longbenchRows, pacRows),
  ];
  sheet.getRangeByIndexes(8, 0, modelSummary.length, modelSummary[0].length).values = modelSummary;
  styleHeader(sheet.getRangeByIndexes(8, 0, 1, modelSummary[0].length));
  styleTable(sheet.getRangeByIndexes(8, 0, modelSummary.length, modelSummary[0].length));
  sheet.getRange("B10:B18").format.numberFormat = "0";
  sheet.getRange("D10:D18").format.numberFormat = "0";
  sheet.getRange("C10:C18").format.numberFormat = "0.000";
  sheet.getRange("E10:E18").format.numberFormat = "0.000";
  sheet.freezePanes.freezeRows(8);

  await addImage(sheet, "longbench_task_heatmap.png", "LongBench task heatmap", 20, 0, 860, 500);
  await addImage(sheet, "pac_subset_heatmap.png", "PAC subset heatmap", 20, 8, 760, 500);
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const workbook = Workbook.create();
  await addOverview(workbook);

  for (const [fileName, sheetName, tableName] of TABLE_SHEETS) {
    await addTableSheet(workbook, fileName, sheetName, tableName);
  }

  for (const [sheetName, charts] of CHART_SHEETS) {
    await addChartSheet(workbook, sheetName, charts);
  }

  const inspect = await workbook.inspect({
    kind: "sheet,table,drawing",
    maxChars: 8000,
    tableMaxRows: 3,
    tableMaxCols: 5,
  });
  console.log(inspect);

  for (const sheetName of ["Overview", "LongBench Charts", "PAC Overview Charts"]) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.75, format: "png" });
    const previewBytes = new Uint8Array(await preview.arrayBuffer());
    await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName.replace(/\W+/g, "_")}.png`), previewBytes);
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_XLSX);
  console.log(`Wrote ${OUTPUT_XLSX}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
