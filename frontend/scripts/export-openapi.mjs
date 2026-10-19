import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const venvPython = resolve("..", ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const candidates = [process.env.PYTHON, existsSync(venvPython) ? venvPython : undefined, process.platform === "win32" ? "py" : "python3", "python"]
  .filter((candidate, index, all) => candidate && all.indexOf(candidate) === index);
let lastError;
for (const executable of candidates) {
  const args = executable === "py" ? ["-3", "../backend/scripts/export_openapi.py"] : ["../backend/scripts/export_openapi.py"];
  const result = spawnSync(executable, args, { stdio: "inherit", shell: process.platform === "win32" && !existsSync(executable) });
  if (!result.error && result.status === 0) process.exit(0);
  lastError = result.error ?? new Error(`Python exited with status ${result.status}`);
}
console.error(`Could not export OpenAPI schema using ${candidates.join(", ")}: ${lastError?.message ?? "unknown error"}`);
process.exit(1);
