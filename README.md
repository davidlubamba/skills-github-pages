#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import requests
import pandas as pd
import numpy as np

try:
    import ccxt  # type: ignore
except Exception as e:
    raise SystemExit("❌ ccxt est requis. Installez-le avec: pip install ccxt")

@dataclass
class Config:
    TELEGRAM_TOKEN: str = "7608427283:AAFSc5rVlD4C9m1IfZO1Q56GGxmJ1-tvvg8"
    TELEGRAM_CHAT_ID: str = "6602171625"
    timeframe: str = "1m"
    scan_interval_sec: int = 45
    lookback_bars: int = 200
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_buy_max: float = 35.0
    rsi_sell_min: float = 65.0
    vol_ma_window: int = 20
    vol_spike_ratio: float = 1.5
    gold_exchanges_try: Tuple[str, ...] = ("bitget", "bybit", "mexc", "bingx")
    notify_errors_to_telegram: bool = True

CFG = Config()

TELEGRAM_API = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}"

def tg_send_message(text: str, parse_mode: Optional[str] = None) -> None:
    try:
        url = f"{TELEGRAM_API}/sendMessage"
        payload = {"chat_id": CFG.TELEGRAM_CHAT_ID, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
    except Exception:
        print("[WARN] Échec d'envoi Telegram:\n", traceback.format_exc())

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

class ExchangeManager:
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
    def get_exchange(self, ex_id: str) -> ccxt.Exchange:
        if ex_id not in self.exchanges:
            ex_class = getattr(ccxt, ex_id)
            ex = ex_class({"enableRateLimit": True, "options": {"defaultType": "spot"}, "timeout": 10000})
            self.exchanges[ex_id] = ex
        return self.exchanges[ex_id]
    def load_markets_safe(self, ex: ccxt.Exchange) -> bool:
        try:
            ex.load_markets()
            return True
        except Exception:
            print(f"[WARN] load_markets échoué pour {ex.id}:\n", traceback.format_exc())
            return False
    def symbol_exists(self, ex: ccxt.Exchange, symbol: str) -> bool:
        try:
            return symbol in ex.markets
        except Exception:
            return False
    def fetch_ohlcv_safe(self, ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[List[List[float]]]:
        try:
            return ex.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        except Exception:
            print(f"[WARN] fetch_ohlcv échoué {ex.id} {symbol}:\n", traceback.format_exc())
            return None

XMAN = ExchangeManager()

@dataclass
class Signal:
    side: str
    price: float
    ts_ms: int
    reason: str

def detect_signals(df: pd.DataFrame) -> Optional[Signal]:
    if df is None or df.empty or len(df) < max(CFG.ema_slow, CFG.rsi_period, CFG.vol_ma_window) + 3:
        return None
    closes = df['close']
    volumes = df['volume']
    df['ema_fast'] = ema(closes, CFG.ema_fast)
    df['ema_slow'] = ema(closes, CFG.ema_slow)
    df['rsi'] = rsi(closes, CFG.rsi_period)
    df['vol_ma'] = volumes.rolling(CFG.vol_ma_window).mean()
    i = -2
    prev_fast = df['ema_fast'].iloc[i-1]
    prev_slow = df['ema_slow'].iloc[i-1]
    curr_fast = df['ema_fast'].iloc[i]
    curr_slow = df['ema_slow'].iloc[i]
    cross_up = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
    cross_dn = (prev_fast >= prev_slow) and (curr_fast < curr_slow)
    curr_rsi = df['rsi'].iloc[i]
    curr_vol = df['volume'].iloc[i]
    curr_vol_ma = df['vol_ma'].iloc[i]
    vol_ok = bool(curr_vol_ma > 0 and curr_vol > CFG.vol_spike_ratio * curr_vol_ma)
    ts_ms = int(df['timestamp'].iloc[i])
    price = float(df['close'].iloc[i])
    if cross_up and (curr_rsi <= CFG.rsi_buy_max) and vol_ok:
        reason = f"EMA{CFG.ema_fast}/{CFG.ema_slow} cross▲, RSI={curr_rsi:.1f}≤{CFG.rsi_buy_max}, Vol spike"
        return Signal(side="BUY", price=price, ts_ms=ts_ms, reason=reason)
    if cross_dn and (curr_rsi >= CFG.rsi_sell_min) and vol_ok:
        reason = f"EMA{CFG.ema_fast}/{CFG.ema_slow} cross▼, RSI={curr_rsi:.1f}≥{CFG.rsi_sell_min}, Vol spike"
        return Signal(side="SELL", price=price, ts_ms=ts_ms, reason=reason)
    return None

class ScalpingBot:
    def __init__(self):
        self.last_alert: Dict[str, Dict[str, int]] = {}
        self.symbol_sources: Dict[str, Tuple[str, str]] = {}
        self._setup_symbols()
    def _setup_symbols(self) -> None:
        try:
            ex = XMAN.get_exchange("binance")
            if XMAN.load_markets_safe(ex) and XMAN.symbol_exists(ex, "BTC/USDT"):
                self.symbol_sources["BTC/USDT"] = ("binance", "BTC/USDT")
                print("[OK] Source prête: binance BTC/USDT")
            else:
                print("[WARN] BTC/USDT indisponible sur Binance.")
        except Exception:
            print("[WARN] Binance init échouée:\n", traceback.format_exc())
        gold_symbols_try = ("XAU/USDT", "GOLD/USDT")
        found = False
        for ex_id in CFG.gold_exchanges_try:
            try:
                ex = XMAN.get_exchange(ex_id)
                if not XMAN.load_markets_safe(ex):
                    continue
                for sym in gold_symbols_try:
                    if XMAN.symbol_exists(ex, sym):
                        self.symbol_sources[sym] = (ex_id, sym)
                        found = True
                        print(f"[OK] Source prête: {ex_id} {sym}")
                        break
                if found:
                    break
            except Exception:
                print(f"[WARN] {ex_id} init échouée:\n", traceback.format_exc())
        if not found:
            print("[WARN] Aucune paire or trouvée (XAU/USDT ou GOLD/USDT). Le bot tournera sans l'or.")
    def _fetch_dataframe(self, ex_id: str, symbol: str) -> Optional[pd.DataFrame]:
        ex = XMAN.get_exchange(ex_id)
        ohlcv = XMAN.fetch_ohlcv_safe(ex, symbol, CFG.timeframe, CFG.lookback_bars)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df
    def _should_alert(self, symbol_key: str, sig: Signal) -> bool:
        if symbol_key not in self.last_alert:
            self.last_alert[symbol_key] = {"BUY": -1, "SELL": -1}
        last_ts = self.last_alert[symbol_key].get(sig.side, -1)
        return sig.ts_ms != last_ts
    def _mark_alerted(self, symbol_key: str, sig: Signal) -> None:
        self.last_alert.setdefault(symbol_key, {"BUY": -1, "SELL": -1})
        self.last_alert[symbol_key][sig.side] = sig.ts_ms
    def _format_signal_msg(self, market_label: str, sig: Signal, ex_id: str) -> str:
        arrow = "🟢 BUY" if sig.side == "BUY" else "🔴 SELL"
        return (
            f"📈 Signal SCALPING détecté sur {market_label} — {arrow}\n"
            f"🔹 {sig.reason}\n"
            f"🔹 Prix: {sig.price:.2f}\n"
            f"🔹 Exchange: {ex_id.upper()} | TF: {CFG.timeframe}\n"
            f"⏱️ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sig.ts_ms/1000))}"
        )
    def scan_once(self) -> None:
        for symbol_key, (ex_id, symbol) in list(self.symbol_sources.items()):
            try:
                df = self._fetch_dataframe(ex_id, symbol)
                if df is None or df.empty:
                    print(f"[WARN] Pas de données {ex_id} {symbol}")
                    continue
                sig = detect_signals(df)
                if sig and self._should_alert(symbol_key, sig):
                    msg = self._format_signal_msg(symbol_key, sig, ex_id)
                    print("[ALERTE]", msg)
                    tg_send_message(msg)
                    self._mark_alerted(symbol_key, sig)
                else:
                    last_price = df['close'].iloc[-1]
                    print(f"[OK] {ex_id} {symbol} dernière bougie close={last_price:.2f} — aucun signal")
            except Exception as e:
                print(f"[ERR] scan {ex_id} {symbol}: {e}\n{traceback.format_exc()}")
                if CFG.notify_errors_to_telegram:
                    tg_send_message(f"⚠️ Erreur scan {ex_id} {symbol}: {e}")
    def run_forever(self) -> None:
        tg_send_message("🤖 Bot de scalping démarré. Surveille les marchés…")
        print("==== Bot démarré. Appuyez sur CTRL+C pour arrêter. ====")
        while True:
            start = time.time()
            self.scan_once()
            elapsed = time.time() - start
            sleep_s = max(3, CFG.scan_interval_sec - int(elapsed))
            time.sleep(sleep_s)

if __name__ == "__main__":
    try:
        bot = ScalpingBot()
        bot.run_forever()
    except KeyboardInterrupt:
        print("[STOP] Interruption utilisateur.")
        tg_send_message("🛑 Bot de scalping arrêté manuellement.")
    except Exception as e:
        err = f"[FATAL] {e}\n{traceback.format_exc()}"
        print(err)
        if CFG.notify_errors_to_telegram:
            tg_send_message(f"⛔ Erreur fatale: {e}")
            
