import sys
import pathlib
import subprocess
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed

START = '2026-04-19'
END   = '2026-10-18'
MAX_WORKERS = 2  # Để 2 luồng giúp 16GB RAM cực kỳ dư dả, không bao giờ tràn bộ nhớ

def make_chunks(start_str, end_str, days=2):
    s = dt.date.fromisoformat(start_str)
    e = dt.date.fromisoformat(end_str)
    chunks = []
    grid_root = pathlib.Path("data_processed/cryptohftdata_6m/grid")
    
    while s <= e:
        t = min(s + dt.timedelta(days=days - 1), e)
        all_done = True
        cur = s
        while cur <= t:
            d_str = cur.isoformat()
            btc_ok = any((grid_root / "symbol=BTCUSDT" / f"date={d_str}").glob("*.parquet"))
            eth_ok = any((grid_root / "symbol=ETHUSDT" / f"date={d_str}").glob("*.parquet"))
            if not (btc_ok and eth_ok):
                all_done = False
                break
            cur += dt.timedelta(days=1)
            
        if not all_done:
            chunks.append((s.isoformat(), t.isoformat()))
        s = t + dt.timedelta(days=1)
    return chunks

def replay_chunk(idx, total, a, b):
    manifest_dir = f"outputs/manifests/archive_6m/part_{a}_{b}"
    replay_cmd = [
        sys.executable, "backend/scripts/replay_cryptohftdata_archive.py",
        "--start-date", a,
        "--end-date", b,
        "--output-root", "data_processed/cryptohftdata_6m/grid",
        "--manifest-root", manifest_dir
    ]
    print(f"⚡ [Replay {idx}/{total}] Đang xử lý: {a} -> {b} ...")
    res = subprocess.run(replay_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        return f"✅ [Replay {idx}/{total}] HOÀN TẤT: {a} -> {b}"
    else:
        err_msg = (res.stderr or res.stdout)[-400:]
        return f"❌ [Replay {idx}/{total}] LỖI ({a} -> {b}):\n{err_msg}"

if __name__ == '__main__':
    chunks = make_chunks(START, END, days=2)
    total = len(chunks)
    print(f"=== CÒN {total} ĐỢT CẦN XỬ LÝ (CÁC ĐỢT ĐÃ XONG ĐƯỢC BỎ QUA TỰ ĐỘNG) ===")

    # 1. Đảm bảo tải đủ data_raw cho các đợt này trước (file đã có sẽ báo cached cực nhanh)
    for i, (a, b) in enumerate(chunks, 1):
        print(f"📥 [Kiểm tra/Tải {i}/{total}] {a} -> {b} ...")
        subprocess.run([
            sys.executable, "backend/scripts/download_cryptohftdata_archive.py",
            "--start-date", a, "--end-date", b
        ])

    # 2. Chạy Replay song song
    print(f"\n=== BẮT ĐẦU REPLAY SONG SONG ({MAX_WORKERS} LUỒNG) ===")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(replay_chunk, i + 1, total, a, b)
            for i, (a, b) in enumerate(chunks)
        ]
        for f in as_completed(futures):
            print(f.result())

    print("\n🎉 HOÀN TẤT TOÀN BỘ 6 THÁNG TRÊN Ổ D:!")
