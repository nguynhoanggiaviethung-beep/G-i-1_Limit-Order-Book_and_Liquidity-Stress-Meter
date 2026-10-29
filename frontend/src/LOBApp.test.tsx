import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

const eventSources: Array<{
  url: string;
  onmessage: ((event: { data: string }) => void) | null;
  onerror: (() => void) | null;
  close: () => void;
}> = [];

function mockApi(
  stressThreshold: number | null,
  meterOverrides: Record<string, unknown> = {},
  options: { emptyAt?: boolean } = {},
) {
  const status = {
    status: "ready",
    source: "test fixture",
    symbols: Object.fromEntries(["BTCUSDT", "ETHUSDT"].map((symbol) => [symbol, {
        status: "ready", partition_count: 1, row_count: 599,
        first_date_utc: symbol === "ETHUSDT" ? "2026-10-23" : "2026-10-21",
        last_date_utc: symbol === "ETHUSDT" ? "2026-10-23" : "2026-10-21", reason: null,
      }])),
  };
  const meter = {
    status: "ok", horizon_seconds: 5, p_up: 0.4, p_flat: 0.3, p_down: 0.3,
    p_stress: 0.4, stress_probability_threshold: stressThreshold, model_version: "test-model",
    depth_imbalance: { L1: 0.1, L5: 0.1, L10: 0.1, L20: 0.1 },
    ...meterOverrides,
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const isEth = url.includes("/replay/ETHUSDT") || url.includes("/meter/ETHUSDT");
    const selectedSnapshot = isEth
      ? {
          ...snapshot,
          symbol: "ETHUSDT",
          timestamp_utc: "2026-10-23T16:17:00Z",
          features: { mid_price: 3200.5, relative_spread_bps: 0.15 },
        }
      : snapshot;
    let body: unknown;
    if (url.includes("/data-status")) body = status;
    else if (url.includes("/replay/") && url.includes("/at?")) {
      body = options.emptyAt && url.includes("2026-10-07")
        ? { snapshot: null, reason: "No replay book at requested time" }
        : { snapshot: selectedSnapshot };
    }
    else if (url.includes("/replay/")) body = { snapshots: [selectedSnapshot] };
    else if (url.includes("/meter/")) body = meter;
    else throw new Error(`Unexpected API request: ${url}`);
    return { ok: true, json: async () => body } as Response;
  }));
}

describe("LOBApp stress decision threshold", () => {
  beforeEach(() => {
    mockApi(0.35);
    eventSources.length = 0;
    vi.stubGlobal("EventSource", vi.fn((url: string) => {
      const source = { url, onmessage: null, onerror: null, close: vi.fn() };
      eventSources.push(source);
      return source;
    }));
  });

  function renderReplay() {
    render(<LOBApp />);
    fireEvent.click(screen.getByRole("button", { name: "replay" }));
  }

  it("compares stress probability with the threshold stored in the artifact", async () => {
    renderReplay();
    await waitFor(() => expect(screen.getByText("■ vượt ngưỡng 35.0%")).toBeInTheDocument());
  });

  it("shows that the decision threshold is unavailable instead of assuming 50 percent", async () => {
    mockApi(null);
    renderReplay();
    await waitFor(() => expect(screen.getByText("Chưa có ngưỡng quyết định trong model artifact")).toBeInTheDocument());
    expect(screen.queryByText(/50\.0%/)).not.toBeInTheDocument();
  });

  it("shows unavailable model output without presenting fixture probabilities as real", async () => {
    mockApi(null, {
      status: "unavailable", p_up: null, p_flat: null, p_down: null, p_stress: null,
      model_version: null, reason: "No trained artifact is available for this symbol and horizon",
    });
    renderReplay();
    await waitFor(() => expect(screen.getAllByText(/No trained artifact is available/).length).toBeGreaterThan(0));
    expect(screen.getByText("— chưa có xác suất.")).toBeInTheDocument();
    expect(screen.getByText("— meter chưa sẵn sàng.")).toBeInTheDocument();
    expect(screen.queryByText("40.0%")).not.toBeInTheDocument();
    expect(screen.getByText(/chỉ đọc · không đặt lệnh/)).toBeInTheDocument();
  });

  it("clears stale metrics and explains the available range when the requested time has no book", async () => {
    mockApi(0.35, {}, { emptyAt: true });
    renderReplay();
    await waitFor(() => expect(screen.getByText("■ vượt ngưỡng 35.0%")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("thời điểm replay · giờ máy"), {
      target: { value: "2026-10-07T15:16" },
    });

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(
      "Dữ liệu BTCUSDT hiện có từ 2026-10-21 đến 2026-10-21 (UTC)",
    ));
    expect(screen.queryByText("40.0%")).not.toBeInTheDocument();
    expect(screen.getByText("— chưa có xác suất.")).toBeInTheDocument();
    expect(screen.getByText("— meter chưa sẵn sàng.")).toBeInTheDocument();
    expect(screen.getAllByText("Chưa có snapshot hợp lệ.")).toHaveLength(2);
  });

  it("loads ETH at its own latest timestamp when the selected symbol changes", async () => {
    mockApi(0.35);
    renderReplay();
    const timeInput = screen.getByLabelText("thời điểm replay · giờ máy") as HTMLInputElement;
    await waitFor(() => expect(timeInput.value).not.toBe("") );

    fireEvent.change(timeInput, { target: { value: "2026-10-07T15:16" } });
    fireEvent.click(screen.getByRole("button", { name: "ETH/USDT" }));

    await waitFor(() => expect(timeInput.value).toContain("2026-10-23"));
    expect(screen.getByText("3,200.50")).toBeInTheDocument();
    expect(screen.queryByText("65,000.50")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("explains the research page in plain language, with a glossary before the answers", () => {
    render(<LOBApp />);
    fireEvent.click(screen.getByRole("button", { name: "nghiên cứu" }));
    expect(screen.getByRole("heading", { name: "Giải thích thuật ngữ" })).toBeInTheDocument();
    for (const term of ["OBI", "OFI", "Persistence", "MCC", "PR-AUC"]) {
      expect(screen.getByText(term)).toBeInTheDocument();
    }
    expect(screen.getAllByText("Con số cụ thể:")).toHaveLength(5);
    expect(screen.getAllByText("cách đọc hình")).toHaveLength(7);
  });

  it("opens a continuous live stream by default and renders its newest update", async () => {
    render(<LOBApp />);
    await waitFor(() => expect(eventSources.length).toBeGreaterThan(0));
    expect(eventSources.at(-1)?.url).toContain("/api/v1/live/BTCUSDT/stream?horizon_seconds=5");

    eventSources.at(-1)?.onmessage?.({
      data: JSON.stringify({
        schema: "live_market.v1",
        symbol: "BTCUSDT",
        status: "live",
        reason: null,
        updated_at_utc: "2026-10-23T16:17:01Z",
        snapshot,
        meter: {
          status: "ok", horizon_seconds: 5, p_up: 0.4, p_flat: 0.3, p_down: 0.3,
          p_stress: 0.4, stress_probability_threshold: 0.35, model_version: "live-test-model",
          depth_imbalance: { L1: 0.1, L5: 0.1, L10: 0.1, L20: 0.1 }, ofi_5s: 0.5,
        },
        horizon_seconds: 5,
      }),
    });

    await waitFor(() => expect(screen.getByText("65,000.50")).toBeInTheDocument());
    expect(screen.getAllByText("đang nhận dữ liệu").length).toBeGreaterThan(0);
    expect(screen.getByText("■ vượt ngưỡng 35.0%")).toBeInTheDocument();
  });
});
