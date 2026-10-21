import { describe, expect, it } from "vitest";
import { parseCsv } from "./client";

describe("parseCsv", () => {
  it("keeps commas, quotes and newlines inside quoted CSV cells", () => {
    const table = parseCsv('model,note,value\n"har,log","line one\nline ""two""",0.25', "T3.csv");
    expect(table.name).toBe("T3.csv");
    expect(table.headers).toEqual(["model", "note", "value"]);
    expect(table.rows).toEqual([{ model: "har,log", note: 'line one\nline "two"', value: "0.25" }]);
  });

  it("supports Windows line endings and empty fields", () => {
    const table = parseCsv("symbol,model\r\nBTCUSDT,har\r\nETHUSDT,\r\n");
    expect(table.rows).toEqual([{ symbol: "BTCUSDT", model: "har" }, { symbol: "ETHUSDT", model: "" }]);
  });
});
