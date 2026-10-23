import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LOBApp } from "./LOBApp";

const snapshot = {
  symbol: "BTCUSDT",
  timestamp_utc: "2026-10-21T16:17:00Z",
  bids: [["65000", "0.5"]],
  asks: [["65001", "0.4"]],
  features: { mid_price: 65000.5, relative_spread_bps: 0.15 },
  quality: { book_valid: true, seconds_since_gap: 20, sequence_gap: false },
};

function mockApi(stressThreshold: number | null, meterOverrides: Record<string, unknown> = {}) {
  const status = {
    status: "ready",
    source: "test fixture",
    symbols: {
      BTCUSDT: {
        status: "ready", partition_count: 1, row_count: 599,
        first_date_utc: "2026-10-21", last_date_utc: "2026-10-21", reason: null,
      },
    },
  };
  const meter = {
    status: "ok", horizon_seconds: 5, p_up: 0.4, p_flat: 0.3, p_down: 0.3,
    p_stress: 0.4, stress_probability_threshold: stressThreshold, model_version: "test-model",
    depth_imbalance: { L1: 0.1, L5: 0.1, L10: 0.1, L20: 0.1 },
    ...meterOverrides,
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    let body: unknown;
    if (url.includes("/data-status")) body = status;
    else if (url.includes("/replay/") && url.includes("/at?")) body = { snapshot };
    else if (url.includes("/replay/")) body = { snapshots: [snapshot] };
    else if (url.includes("/meter/")) body = meter;
    else throw new Error(`Unexpected API request: ${url}`);
    return { ok: true, json: async () => body } as Response;
  }));
}

describe("LOBApp stress decision threshold", () => {
  beforeEach(() => {
    mockApi(0.35);
  });

  it("compares stress probability with the threshold stored in the artifact", async () => {
    render(<LOBApp />);
    await waitFor(() => expect(screen.getByText("■ vượt ngưỡng 35.0%")).toBeInTheDocument());
  });

  it("shows that the decision threshold is unavailable instead of assuming 50 percent", async () => {
    mockApi(null);
    render(<LOBApp />);
    await waitFor(() => expect(screen.getByText("Chưa có ngưỡng quyết định trong model artifact")).toBeInTheDocument());
    expect(screen.queryByText(/50\.0%/)).not.toBeInTheDocument();
  });

  it("shows unavailable model output without presenting fixture probabilities as real", async () => {
    mockApi(null, {
      status: "unavailable", p_up: null, p_flat: null, p_down: null, p_stress: null,
      model_version: null, reason: "No trained artifact is available for this symbol and horizon",
    });
    render(<LOBApp />);
    await waitFor(() => expect(screen.getAllByText(/No trained artifact is available/).length).toBeGreaterThan(0));
    expect(screen.getByText("— chưa có xác suất.")).toBeInTheDocument();
    expect(screen.getByText("— meter chưa sẵn sàng.")).toBeInTheDocument();
    expect(screen.queryByText("40.0%")).not.toBeInTheDocument();
    expect(screen.getByText(/chỉ đọc · không đặt lệnh/)).toBeInTheDocument();
  });
});
