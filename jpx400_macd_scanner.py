#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPX400 MACD(12,26,9) scanner
- Reads a constituent CSV/XLSX supplied from the official JPX list.
- Downloads daily prices with yfinance in batches.
- Finds MACD golden crosses below zero.
- Exports an Excel workbook and prints the signal list.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

FAST = 12
SLOW = 26
SIGNAL = 9
BATCH_SIZE = 40
MIN_HISTORY = 80


def load_constituents(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。JPX公式の構成銘柄一覧をCSV/XLSXで保存してください。"
        )
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, dtype=str)
    else:
        raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    # Accepted column names
    code_col = next((c for c in raw.columns if c in {"code", "コード", "銘柄コード", "証券コード"}), None)
    name_col = next((c for c in raw.columns if c in {"name", "銘柄名", "会社名", "名称"}), None)
    if not code_col:
        raise ValueError("構成銘柄ファイルに code または 銘柄コード 列が必要です。")
    if not name_col:
        raw["name"] = ""
        name_col = "name"

    out = raw[[code_col, name_col]].copy()
    out.columns = ["code", "name"]
    out["code"] = out["code"].astype(str).str.extract(r"([0-9]{4}|[0-9]{3}[A-Z])", expand=False)
    out = out.dropna(subset=["code"]).drop_duplicates("code")
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out["ticker"] = out["code"] + ".T"

    if len(out) != 400:
        raise ValueError(
            f"構成銘柄が {len(out)} 銘柄です。400銘柄ではないため停止しました。"
            "公式リストの重複・ヘッダー・ファイル形式を確認してください。"
        )
    return out.reset_index(drop=True)


def macd_for_close(close: pd.Series) -> pd.DataFrame:
    # EMA uses adjust=False, the common recursive EMA convention.
    ema_fast = close.ewm(span=FAST, adjust=False, min_periods=FAST).mean()
    ema_slow = close.ewm(span=SLOW, adjust=False, min_periods=SLOW).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=SIGNAL, adjust=False, min_periods=SIGNAL).mean()
    return pd.DataFrame({"MACD": macd, "Signal": sig})


def extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance multi-index layout may be (Ticker, Price) or (Price, Ticker).
        if ticker in data.columns.get_level_values(0):
            frame = data[ticker].copy()
        elif ticker in data.columns.get_level_values(-1):
            frame = data.xs(ticker, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = data.copy()
    if "Close" not in frame.columns:
        return pd.DataFrame()
    return frame


def scan(args):
    constituents = load_constituents(Path(args.constituents))
    asof = pd.Timestamp(args.asof).normalize()
    # yfinance end is exclusive. Include the requested as-of date.
    end_exclusive = (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (asof - pd.Timedelta(days=args.lookback_calendar_days)).strftime("%Y-%m-%d")
    tickers = constituents["ticker"].tolist()

    all_prices = {}
    failures = []
    for offset in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[offset:offset+BATCH_SIZE]
        try:
            downloaded = yf.download(
                tickers=batch,
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=True,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=25,
                multi_level_index=True,
            )
        except Exception as exc:
            downloaded = pd.DataFrame()
            failures.extend({"ticker": t, "reason": f"batch download error: {exc}"} for t in batch)

        for ticker in batch:
            frame = extract_ticker_frame(downloaded, ticker)
            if frame.empty:
                if not any(x["ticker"] == ticker for x in failures):
                    failures.append({"ticker": ticker, "reason": "no price data returned"})
                continue
            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) else pd.to_datetime(close.index)
            close = close[~close.index.duplicated(keep="last")].sort_index()
            close = close[close.index.normalize() <= asof]
            if len(close) < MIN_HISTORY:
                failures.append({"ticker": ticker, "reason": f"insufficient history ({len(close)} rows)"})
                continue
            all_prices[ticker] = close

    results = []
    history_rows = []
    lookup = constituents.set_index("ticker")
    target_date = None
    # Determine actual trading date present in the downloaded series for the requested as-of.
    observed_dates = [s.index.max().normalize() for s in all_prices.values() if not s.empty]
    if observed_dates:
        target_date = max(d for d in observed_dates if d <= asof)

    for ticker, close in all_prices.items():
        code = ticker.replace(".T", "")
        name = lookup.loc[ticker, "name"]
        m = macd_for_close(close)
        joined = pd.concat([close.rename("Close"), m], axis=1).dropna()
        if joined.empty:
            failures.append({"ticker": ticker, "reason": "MACD unavailable"})
            continue
        # For the requested day, require that the ticker has a close on the same target session.
        day_rows = joined.loc[joined.index.normalize() == target_date] if target_date is not None else joined.iloc[0:0]
        if day_rows.empty:
            latest = joined.iloc[-1]
            results.append({
                "code": code, "name": name, "ticker": ticker,
                "status": "NO_CLOSE_ON_TARGET_DATE",
                "date": joined.index[-1].date().isoformat(),
                "close": float(latest["Close"]), "MACD": float(latest["MACD"]),
                "Signal": float(latest["Signal"]), "MACD_prev": None,
                "Signal_prev": None, "golden_cross": False, "below_zero_cross": False,
                "note": "対象日の終値データなし。シグナル判定対象外"
            })
            continue
        idx = day_rows.index[-1]
        pos = joined.index.get_loc(idx)
        today = joined.iloc[pos]
        prev = joined.iloc[pos-1] if pos > 0 else None
        if prev is None:
            cross = False
        else:
            cross = bool(today["MACD"] > today["Signal"] and prev["MACD"] <= prev["Signal"])
        below_zero = bool(cross and today["MACD"] < 0)
        status = "OK"
        results.append({
            "code": code, "name": name, "ticker": ticker,
            "status": status, "date": idx.date().isoformat(),
            "close": float(today["Close"]), "MACD": float(today["MACD"]),
            "Signal": float(today["Signal"]),
            "MACD_prev": float(prev["MACD"]) if prev is not None else None,
            "Signal_prev": float(prev["Signal"]) if prev is not None else None,
            "golden_cross": cross, "below_zero_cross": below_zero,
            "note": ""
        })
        # Daily calculation detail, useful for audit/re-check
        history_rows.append({
            "code": code, "name": name, "ticker": ticker, "date": idx.date().isoformat(),
            "close": float(today["Close"]), "MACD": float(today["MACD"]),
            "Signal": float(today["Signal"]),
            "MACD_prev": float(prev["MACD"]) if prev is not None else None,
            "Signal_prev": float(prev["Signal"]) if prev is not None else None,
            "golden_cross": cross, "below_zero_cross": below_zero
        })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        result_df = pd.DataFrame(columns=[
            "code","name","ticker","status","date","close","MACD","Signal",
            "MACD_prev","Signal_prev","golden_cross","below_zero_cross","note"
        ])
    result_df = result_df.sort_values(["below_zero_cross", "code"], ascending=[False, True])
    signals = result_df[(result_df["status"] == "OK") & (result_df["below_zero_cross"] == True)].copy()
    failed_df = pd.DataFrame(failures, columns=["ticker", "reason"]).drop_duplicates()

    audit = pd.DataFrame([{
        "requested_asof": asof.date().isoformat(),
        "actual_common_target_date": target_date.date().isoformat() if target_date is not None else "",
        "constituents_in_file": len(constituents),
        "price_series_received": len(all_prices),
        "rows_evaluated_on_target_date": int((result_df["status"] == "OK").sum()),
        "signals_found": len(signals),
        "data_failures_or_exclusions": len(failed_df),
        "price_source": "Yahoo Finance via yfinance (unofficial; verify against a licensed source)",
        "price_mode": "auto_adjust=True; adjusted Close",
        "MACD_definition": "EMA12(Close)-EMA26(Close); Signal=EMA9(MACD); ewm(adjust=False)",
        "condition": "today MACD > Signal; prior MACD <= prior Signal; today's MACD < 0",
        "note": "Not investment advice. Check constituent list date and provider data before trading."
    }])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        signals.to_excel(writer, sheet_name="該当銘柄", index=False)
        result_df.to_excel(writer, sheet_name="全400照合", index=False)
        failed_df.to_excel(writer, sheet_name="取得失敗等", index=False)
        audit.to_excel(writer, sheet_name="検証情報", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                letter = col[0].column_letter
                max_len = min(max((len(str(c.value)) if c.value is not None else 0 for c in col), default=10) + 2, 42)
                ws.column_dimensions[letter].width = max(10, max_len)

    print(f"\n対象日: {asof.date()} / 判定に使った市場日: {target_date.date() if target_date is not None else '不明'}")
    print(f"構成銘柄: {len(constituents)} / 株価取得成功: {len(all_prices)} / 判定対象: {(result_df['status']=='OK').sum()} / 取得失敗等: {len(failed_df)}")
    print(f"条件一致: {len(signals)} 銘柄")
    if signals.empty:
        print("条件一致銘柄は確認できませんでした（取得失敗銘柄がある場合は全件確認ではありません）。")
    else:
        print("\nコード | 銘柄名 | 終値 | MACD | Signal")
        for _, row in signals.iterrows():
            print(f"{row['code']} | {row['name']} | {row['close']:.2f} | {row['MACD']:.6f} | {row['Signal']:.6f}")
    print(f"\nExcel出力: {out_path.resolve()}")
    if len(constituents) != 400 or len(failed_df) > 0 or (result_df["status"] != "OK").any():
        print("注意: データ取得失敗/対象日欠損があります。全400銘柄を完全照合できたとは扱わないでください。")


def main():
    parser = argparse.ArgumentParser(description="JPX400 MACD(12,26,9) below-zero golden-cross scanner")
    parser.add_argument("--constituents", default="jpx400_constituents.csv", help="JPX400構成銘柄CSV/XLSX (code,name columns)")
    parser.add_argument("--asof", default=date.today().isoformat(), help="判定日 YYYY-MM-DD。例: 2026-10-14")
    parser.add_argument("--output", default="jpx400_macd_results.xlsx", help="Excel output path")
    parser.add_argument("--lookback-calendar-days", type=int, default=550, help="株価取得の遡り日数")
    args = parser.parse_args()
    try:
        scan(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
