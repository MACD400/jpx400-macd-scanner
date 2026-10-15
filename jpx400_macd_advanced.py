#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JPX400 MACD advanced scanner: confirmed GC/DC and 1-2-session proximity alerts.

Not a prediction of future prices. Proximity uses linear extrapolation of the
last two daily changes in (MACD - signal); it can produce false positives.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

FAST, SLOW, SIGNAL = 12, 26, 9
BATCH_SIZE = 40
MIN_HISTORY = 80


def load_constituents(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"構成銘柄ファイルがありません: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path, dtype=str)
    else:
        raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    code_col = next((c for c in raw if c in {"code", "コード", "銘柄コード", "証券コード"}), None)
    name_col = next((c for c in raw if c in {"name", "銘柄名", "会社名", "名称"}), None)
    if code_col is None:
        raise ValueError("code/銘柄コード列が必要です")
    if name_col is None:
        raw["name"] = ""
        name_col = "name"
    out = raw[[code_col, name_col]].copy()
    out.columns = ["code", "name"]
    out["code"] = out["code"].astype(str).str.upper().str.extract(
        r"([0-9]{4}|[0-9]{3}[A-Z])", expand=False
    )
    out = out.dropna(subset=["code"]).drop_duplicates("code")
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out["ticker"] = out["code"] + ".T"
    if len(out) != 400:
        raise ValueError(f"構成銘柄が{len(out)}件です。400件ではないため停止します")
    return out.reset_index(drop=True)


def macd_frame(close: pd.Series) -> pd.DataFrame:
    fast = close.ewm(span=FAST, adjust=False, min_periods=FAST).mean()
    slow = close.ewm(span=SLOW, adjust=False, min_periods=SLOW).mean()
    macd = fast - slow
    signal = macd.ewm(span=SIGNAL, adjust=False, min_periods=SIGNAL).mean()
    return pd.DataFrame({"Close": close, "MACD": macd, "Signal": signal,
                         "Gap": macd - signal}).dropna()


def ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            frame = data[ticker].copy()
        elif ticker in data.columns.get_level_values(-1):
            frame = data.xs(ticker, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = data.copy()
    return frame if "Close" in frame.columns else pd.DataFrame()


def evaluate(close: pd.Series, target_date: pd.Timestamp) -> dict:
    frame = macd_frame(close)
    if len(frame) < 4:
        raise ValueError("MACDの有効な履歴が不足")
    matching = frame.index[frame.index.normalize() == target_date]
    if len(matching) == 0:
        last = frame.iloc[-1]
        return {"status": "NO_CLOSE_ON_TARGET_DATE",
                "date": frame.index[-1].date().isoformat(),
                "close": float(last["Close"]), "MACD": float(last["MACD"]),
                "Signal": float(last["Signal"]), "Gap": float(last["Gap"]),
                "Gap_prev": None, "Gap_change": None,
                "cross": "NONE", "proximity": "NONE", "estimated_sessions": None,
                "zero_zone": "上" if last["MACD"] >= 0 else "下",
                "note": "対象市場日の終値なし。判定対象外"}
    frame = frame.loc[:matching[-1]]
    now, prev, prev2 = frame.iloc[-1], frame.iloc[-2], frame.iloc[-3]
    gap, before = float(now["Gap"]), float(prev["Gap"])
    gc = gap > 0 and before <= 0
    dc = gap < 0 and before >= 0
    cross = "GC確定" if gc else ("DC確定" if dc else "NONE")

    # A simple heuristic, NOT a forecast: use the average of the last two
    # gap changes and extrapolate a crossing within 1 or 2 sessions.
    d1 = gap - before
    d2 = before - float(prev2["Gap"])
    velocity = (d1 + d2) / 2
    proximity = "NONE"
    estimated = None
    if cross == "NONE" and d1 * d2 > 0:
        if gap < 0 and velocity > 0:
            sessions = -gap / velocity
            if 0 < sessions <= 2:
                proximity = "GC接近"
                estimated = 1 if sessions <= 1 else 2
        elif gap > 0 and velocity < 0:
            sessions = -gap / velocity
            if 0 < sessions <= 2:
                proximity = "DC接近"
                estimated = 1 if sessions <= 1 else 2

    return {"status": "OK", "date": frame.index[-1].date().isoformat(),
            "close": float(now["Close"]), "MACD": float(now["MACD"]),
            "Signal": float(now["Signal"]), "Gap": gap, "Gap_prev": before,
            "Gap_change": d1, "cross": cross, "proximity": proximity,
            "estimated_sessions": estimated,
            "zero_zone": "上" if now["MACD"] >= 0 else "下",
            "note": "接近は直近2回のGap変化による単純外挿。将来のGC/DCを保証しない" if proximity != "NONE" else ""}


def scan(args):
    constituents = load_constituents(Path(args.constituents))
    asof = pd.Timestamp(args.asof).normalize()
    start = (asof - pd.Timedelta(days=args.lookback_calendar_days)).strftime("%Y-%m-%d")
    end = (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices, failures = {}, []
    tickers = constituents["ticker"].tolist()
    for offset in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[offset:offset+BATCH_SIZE]
        try:
            data = yf.download(tickers=batch, start=start, end=end,
                               interval="1d", auto_adjust=True, actions=False,
                               group_by="ticker", threads=True, progress=False,
                               timeout=25, multi_level_index=True)
        except Exception as exc:
            failures.extend({"ticker": t, "reason": f"一括取得失敗: {exc}"} for t in batch)
            continue
        for ticker in batch:
            frame = ticker_frame(data, ticker)
            if frame.empty:
                failures.append({"ticker": ticker, "reason": "価格データなし"})
                continue
            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            idx = pd.to_datetime(close.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            close.index = idx
            close = close[~close.index.duplicated(keep="last")].sort_index()
            close = close[close.index.normalize() <= asof]
            if len(close) < MIN_HISTORY:
                failures.append({"ticker": ticker, "reason": f"履歴不足: {len(close)}日"})
            else:
                prices[ticker] = close

    if not prices:
        raise RuntimeError("株価が1件も取得できませんでした")
    # Majority session is more robust than a single late/outlier ticker.
    observed = pd.Series([s.index.max().normalize() for s in prices.values()])
    target_date = observed.value_counts().index[0]
    lookup = constituents.set_index("ticker")
    rows = []
    for ticker, close in prices.items():
        try:
            outcome = evaluate(close, target_date)
            rows.append({"code": ticker[:-2], "name": lookup.loc[ticker, "name"],
                         "ticker": ticker, **outcome})
        except Exception as exc:
            failures.append({"ticker": ticker, "reason": str(exc)})
    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError("評価できる銘柄がありません")
    ok = all_df[all_df["status"] == "OK"].copy()
    confirmed = ok[ok["cross"] != "NONE"].copy()
    approaching = ok[ok["proximity"] != "NONE"].copy()
    failed = pd.DataFrame(failures, columns=["ticker", "reason"]).drop_duplicates()
    audit = pd.DataFrame([{
        "requested_asof": asof.date().isoformat(),
        "actual_target_session": target_date.date().isoformat(),
        "constituents": len(constituents), "price_series": len(prices),
        "evaluated_on_target": len(ok), "confirmed_crosses": len(confirmed),
        "proximity_alerts": len(approaching),
        "download_or_analysis_failures": len(failed),
        "price_source": "Yahoo Finance via yfinance (unofficial)",
        "price_definition": "auto_adjust=True, adjusted close",
        "macd_definition": "EMA12 - EMA26; Signal EMA9; adjust=False",
        "cross_definition": "GC: Gap>0 and prior Gap<=0; DC: Gap<0 and prior Gap>=0",
        "proximity_definition": "No cross today; two consecutive Gap changes same sign; "
                                "linear extrapolation using average change crosses zero in <=2 sessions",
        "warning": "接近は統計的に検証されていない目安。取引判断に直結させない。"
    }])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        approaching.to_excel(writer, sheet_name="GC_DC接近", index=False)
        confirmed.to_excel(writer, sheet_name="GC_DC確定", index=False)
        all_df.to_excel(writer, sheet_name="全銘柄照合", index=False)
        failed.to_excel(writer, sheet_name="取得失敗等", index=False)
        audit.to_excel(writer, sheet_name="検証情報", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                letter = col[0].column_letter
                width = max((len(str(cell.value)) if cell.value is not None else 0
                             for cell in col), default=8)
                ws.column_dimensions[letter].width = min(45, max(12, width + 2))
    print(f"対象市場日: {target_date.date()} / 判定銘柄: {len(ok)} / "
          f"GC・DC確定: {len(confirmed)} / GC・DC接近: {len(approaching)}")
    print(f"取得失敗等: {len(failed)} / Excel: {path}")
    if len(ok) != 400 or len(failed):
        print("注意: 全400銘柄を完全には照合できていません。")


def main():
    parser = argparse.ArgumentParser(description="JPX400 advanced MACD scanner")
    parser.add_argument("--constituents", default="jpx400_constituents.csv")
    parser.add_argument("--asof", default=date.today().isoformat())
    parser.add_argument("--output", default="jpx400_macd_advanced_results.xlsx")
    parser.add_argument("--lookback-calendar-days", type=int, default=550)
    args = parser.parse_args()
    try:
        scan(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
