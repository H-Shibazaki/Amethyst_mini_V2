# version_info.py
# ここがタイトル情報の「単一の情報源」。GUI 等はここだけ参照する。
from __future__ import annotations
from datetime import datetime, timezone, timedelta

import pathlib
import subprocess

# ---- 手動の固定値（最終フォールバック） ---------------------------------
APP_NAME    = "Amethyst_mini リアルタイム疵検知"   # GUI表示名（必要ならだけここを編集）
MANUAL_VER  = "0.2.0"                               # 手動バージョン（タグ未使用時の保険）
MANUAL_DATE = "2025-08-20 12:00 JST"               # 手動更新日（最終手段）

# ---- JST タイムゾーン ---------------------------------------------------
try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:
    JST = timezone(timedelta(hours=9))  # フォールバック（表示はJSTと明記する）

# ---- 主要ファイル（mtime 走査用）。必要に応じて追加 ----------------------
CORE_FILES = [
    "gui.py",             # タイトルを設定している GUI 本体
    "kizu_kenchi.py",     # リアルタイム推論の中核
    "hailo_runner.py",    # Hailo 実行ラッパ
    "image_process.py",   # 画像保存など
]

def _git_describe() -> str | None:
    """Git のタグ/コミットからバージョン取得。失敗したら None。"""
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return out or None
    except Exception:
        return None

def _git_last_commit_iso() -> str | None:
    """直近コミット日時（ローカルタイム）を ISO で取得。失敗したら None。"""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--date=iso-local", "--format=%cd"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return out or None
    except Exception:
        return None

def _latest_mtime_jst() -> str:
    """主要ファイルの mtime 最大を JST で整形。1つも見つからなければ現在時刻を使う。"""
    base = pathlib.Path(__file__).resolve().parent
    mtimes = []
    for rel in CORE_FILES:
        p = (base / rel)
        if p.exists():
            try:
                mtimes.append(p.stat().st_mtime)
            except Exception:
                pass
    ts = max(mtimes) if mtimes else datetime.now().timestamp()
    dt = datetime.fromtimestamp(ts, JST)
    return dt.strftime("%Y-%m-%d %H:%M JST")

def get_version_and_updated() -> tuple[str, str]:
    """(version, updated_str) を返す。順に Git → mtime → 手動値でフォールバック。"""
    ver = _git_describe()
    updated = _git_last_commit_iso()

    if ver and updated:
        # Git が揃えばそのまま。更新日は ISO 文字列を「JST表示」に揃える努力をする。
        try:
            # 例: '2025-08-20 10:23:45+09:00' などをパース
            from datetime import datetime
            dt = datetime.fromisoformat(updated)
            if dt.tzinfo is not None:
                dt = dt.astimezone(JST)
            else:
                dt = dt.replace(tzinfo=JST)
            updated_str = dt.strftime("%Y-%m-%d %H:%M JST")
        except Exception:
            updated_str = updated   # どうしても無理なら原文のまま
        return ver, updated_str

    # Git がなければ mtime を採用
    mt = _latest_mtime_jst()
    if ver:
        return ver, mt

    # 何も拾えなければ手動値
    return MANUAL_VER, MANUAL_DATE

def get_window_title() -> str:
    """GUI のタイトル文字列を返す。"""
    ver, updated = get_version_and_updated()
    return f"{APP_NAME}  v{ver}  |  最終更新: {updated}"
