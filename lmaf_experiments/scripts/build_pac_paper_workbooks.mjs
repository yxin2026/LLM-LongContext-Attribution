import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "results", "reports", "pac_paper_ready");
const payload = JSON.parse(await fs.readFile(path.join(OUT, "pac_paper_payload.json"), "utf8"));

await fs.mkdir(OUT, { recursive: true });
await buildTableWorkbook();
await buildChartWorkbook();

async function buildTableWorkbook() {
  const workbook = Workbook.create();
  writePaperTable(workbook.worksheets.add(payload.table_workbook.sheet_name), payload.table_workbook);
  await verifyAndExport(workbook, "PAC自设数据集表格整理v0.1.xlsx", "pac_table_preview.png");
}

async function buildChartWorkbook() {
  const workbook = Workbook.create();
  await writeChartSheet(workbook.worksheets.add(payload.chart_workbook.sheet_name), payload.chart_workbook);
  writeConclusionSheet(workbook.worksheets.add(payload.chart_workbook.conclusion_sheet), payload.chart_workbook);
  await verifyAndExport(workbook, "PAC自设数据集图表说明v0.1.xlsx", "pac_chart_preview.png", [
    { sheetName: payload.chart_workbook.conclusion_sheet, fileName: "pac_chart_conclusion_preview.png" },
  ]);
}

function writePaperTable(sheet, spec) {
  sheet.showGridLines = false;
  const colCount = spec.headers.length;
  const rows = [
    [spec.title, ...Array(colCount - 1).fill(null)],
    spec.group_headers,
    spec.headers,
    ...spec.rows,
    [spec.note, ...Array(colCount - 1).fill(null)],
  ];
  const rowCount = rows.length;
  sheet.getRangeByIndexes(0, 0, rowCount, colCount).values = rows;
  sheet.getRangeByIndexes(0, 0, 1, colCount).merge();
  sheet.getRangeByIndexes(rowCount - 1, 0, 1, colCount).merge();
  mergeGroupHeaders(sheet, spec.group_headers);

  sheet.getRangeByIndexes(0, 0, rowCount, colCount).format = {
    font: { name: "Times New Roman", size: 10 },
  };
  sheet.getRangeByIndexes(0, 0, 1, colCount).format = {
    font: { bold: true, size: 15, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRangeByIndexes(1, 0, 1, colCount).format = {
    fill: "#D9E2F3",
    font: { bold: true, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  sheet.getRangeByIndexes(2, 0, 1, colCount).format = {
    fill: "#B4C6E7",
    font: { bold: true, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  sheet.getRangeByIndexes(3, 0, spec.rows.length, colCount).format = {
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  sheet.getRangeByIndexes(rowCount - 1, 0, 1, colCount).format = {
    fill: "#FFFFFF",
    font: { color: "#404040", size: 10 },
    horizontalAlignment: "left",
    verticalAlignment: "top",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  sheet.getRangeByIndexes(3, 2, spec.rows.length, colCount - 2).format.numberFormat = "0.0";

  const widths = [19, 16, 13, 13, 13, 15, 13, 13, 13];
  for (let i = 0; i < colCount; i += 1) {
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidth = widths[i] || 13;
  }
  sheet.getRangeByIndexes(0, 0, 1, colCount).format.rowHeight = 28;
  sheet.getRangeByIndexes(1, 0, 2, colCount).format.rowHeight = 30;
  sheet.getRangeByIndexes(3, 0, spec.rows.length, colCount).format.rowHeight = 24;
  sheet.getRangeByIndexes(rowCount - 1, 0, 1, colCount).format.rowHeight = 112;
  sheet.freezePanes.freezeRows(3);
}

async function writeChartSheet(sheet, spec) {
  sheet.showGridLines = false;
  const colCount = spec.headers.length;
  const rowCount = spec.rows.length + 2;
  sheet.getRangeByIndexes(0, 0, rowCount, colCount).values = [
    [spec.title, ...Array(colCount - 1).fill(null)],
    spec.headers,
    ...spec.rows.map((row) => ["", row.x, row.y, row.metric, row.phenomenon, row.conclusion]),
  ];
  sheet.getRangeByIndexes(0, 0, 1, colCount).merge();

  sheet.getRangeByIndexes(0, 0, rowCount, colCount).format = {
    font: { name: "Times New Roman", size: 10 },
    verticalAlignment: "top",
    wrapText: true,
  };
  sheet.getRangeByIndexes(0, 0, 1, colCount).format = {
    font: { bold: true, size: 15 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRangeByIndexes(1, 0, 1, colCount).format = {
    fill: "#B4C6E7",
    font: { bold: true },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  sheet.getRangeByIndexes(2, 0, spec.rows.length, colCount).format = {
    borders: { preset: "all", style: "thin", color: "#000000" },
    verticalAlignment: "top",
    wrapText: true,
  };

  const widths = [36, 26, 20, 36, 42, 42];
  for (let i = 0; i < colCount; i += 1) {
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidth = widths[i];
  }
  sheet.getRangeByIndexes(0, 0, 1, colCount).format.rowHeight = 30;
  sheet.getRangeByIndexes(1, 0, 1, colCount).format.rowHeight = 32;
  sheet.getRangeByIndexes(2, 0, spec.rows.length, colCount).format.rowHeight = 150;

  for (let i = 0; i < spec.rows.length; i += 1) {
    await addImage(sheet, spec.rows[i].image, i + 2, 0, 250, 125);
  }
  sheet.freezePanes.freezeRows(2);
}

function writeConclusionSheet(sheet, spec) {
  sheet.showGridLines = false;
  sheet.getRange("A1:F6").values = [
    [spec.conclusion_title, null, null, null, null, null],
    [null, null, null, null, null, null],
    [spec.conclusion, null, null, null, null, null],
    [null, null, null, null, null, null],
    [null, null, null, null, null, null],
    [null, null, null, null, null, null],
  ];
  sheet.getRange("A1:F1").merge();
  sheet.getRange("A3:F6").merge();
  sheet.getRange("A1:F6").format = {
    font: { name: "Times New Roman", size: 11 },
    wrapText: true,
  };
  sheet.getRange("A1:F1").format = {
    font: { bold: true, size: 15 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange("A3:F6").format = {
    horizontalAlignment: "left",
    verticalAlignment: "top",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#000000" },
  };
  for (let i = 0; i < 6; i += 1) {
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidth = 24;
  }
  sheet.getRange("A1:F1").format.rowHeight = 30;
  sheet.getRange("A3:F6").format.rowHeight = 36;
}

function mergeGroupHeaders(sheet, groups) {
  let start = 0;
  while (start < groups.length) {
    const label = groups[start];
    let end = start;
    while (end + 1 < groups.length && groups[end + 1] === "") end += 1;
    if (label && end > start) {
      sheet.getRangeByIndexes(1, start, 1, end - start + 1).merge();
    }
    start = end + 1;
  }
}

async function addImage(sheet, filePath, row, col, widthPx, heightPx) {
  const data = await fs.readFile(filePath);
  const dataUrl = `data:image/png;base64,${data.toString("base64")}`;
  sheet.images.add({
    dataUrl,
    anchor: {
      from: { row, col },
      extent: { widthPx, heightPx },
    },
  });
}

async function verifyAndExport(workbook, fileName, previewName, extraPreviews = []) {
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);

  const previews = [{ sheetName: workbook.worksheets.getItemAt(0).name, fileName: previewName }, ...extraPreviews];
  for (const item of previews) {
    const preview = await workbook.render({ sheetName: item.sheetName, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(path.join(OUT, item.fileName), new Uint8Array(await preview.arrayBuffer()));
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  const outPath = path.join(OUT, fileName);
  await xlsx.save(outPath);
  const desktopPath = path.join(payload.desktop_dir, fileName);
  await xlsx.save(desktopPath);
  console.log(`Wrote ${outPath}`);
  console.log(`Wrote ${desktopPath}`);
}
