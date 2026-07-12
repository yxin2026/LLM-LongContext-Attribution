import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "results", "reports", "paper_ready_tables");

const payload = JSON.parse(await fs.readFile(path.join(OUT, "paper_ready_tables_payload.json"), "utf8"));
const workbook = Workbook.create();

writePaperSheet(workbook.worksheets.add("公开数据集"), payload.public);
writePaperSheet(workbook.worksheets.add("PAC自设数据集"), payload.pac);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(OUT, { recursive: true });
for (const [sheetName, fileName] of [
  ["公开数据集", "public_benchmarks_paper_table.png"],
  ["PAC自设数据集", "pac_test_paper_table.png"],
]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 2,
    format: "png",
  });
  await fs.writeFile(path.join(OUT, fileName), new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
const workbookPath = path.join(OUT, "paper_ready_benchmark_tables.xlsx");
await xlsx.save(workbookPath);
console.log(`Wrote workbook to ${workbookPath}`);

function writePaperSheet(sheet, spec) {
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

  mergeSameGroup(sheet, spec.group_headers);

  sheet.getRangeByIndexes(0, 0, rowCount, colCount).format = {
    font: { name: "Times New Roman", size: 10 },
  };

  const title = sheet.getRangeByIndexes(0, 0, 1, colCount);
  title.format = {
    font: { bold: true, size: 15, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };

  const group = sheet.getRangeByIndexes(1, 0, 1, colCount);
  group.format = {
    fill: "#D9E2F3",
    font: { bold: true, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#000000" },
  };

  const header = sheet.getRangeByIndexes(2, 0, 1, colCount);
  header.format = {
    fill: "#B4C6E7",
    font: { bold: true, color: "#000000" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#000000" },
  };

  const body = sheet.getRangeByIndexes(3, 0, spec.rows.length, colCount);
  body.format = {
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#000000" },
  };

  const note = sheet.getRangeByIndexes(rowCount - 1, 0, 1, colCount);
  note.format = {
    fill: "#FFFFFF",
    font: { color: "#404040", size: 10 },
    horizontalAlignment: "left",
    verticalAlignment: "top",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#000000" },
  };

  sheet.getRangeByIndexes(3, 2, spec.rows.length, colCount - 2).format.numberFormat = "0.0";

  setWidths(sheet, colCount);
  sheet.getRangeByIndexes(0, 0, 1, colCount).format.rowHeight = 28;
  sheet.getRangeByIndexes(1, 0, 2, colCount).format.rowHeight = 28;
  sheet.getRangeByIndexes(3, 0, spec.rows.length, colCount).format.rowHeight = 22;
  sheet.getRangeByIndexes(rowCount - 1, 0, 1, colCount).format.rowHeight = 76;
  sheet.freezePanes.freezeRows(3);
}

function mergeSameGroup(sheet, groups) {
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

function setWidths(sheet, colCount) {
  const widths = colCount >= 10
    ? [18, 16, 12, 12, 12, 14, 12, 14, 12, 14]
    : [18, 16, 14, 14, 14, 15, 13, 13];
  for (let i = 0; i < colCount; i += 1) {
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidth = widths[i] || 12;
  }
}
