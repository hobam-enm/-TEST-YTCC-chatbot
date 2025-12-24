# -*- coding: utf-8 -*-
"""
💬 YouTube Comment Compass — ytcc vNext (CACHE-ONLY Own-IP)

✅ 사용자 합의 반영(최종)
1) Deep 수집 유지: 댓글은 가능한 한 전체(페이지네이션) 수집. "총량/영상당 상한" 강제 제한 없음.
2) 자사 IP 모드(토글 ON): PGC 영상 ID는 "ytan 캐시 JSON(cache_token_*.json)"에서만 검색 (키워드만)
   - 캐시가 없으면 자사 IP 모드는 동작하지 않음(에러 안내) — fallback 없음.
3) 자사 IP 모드(토글 ON)에서만: PGC(캐시) + 외부 UGC 혼합(OST 제외)으로 영상 풀 구성.
4) 토글 OFF: 기존처럼 YouTube Search 기반(전 유튜브) 영상 풀 구성.
5) 리포팅: 1차 질문은 ytan 스타일 HTML 리포트(1차 질문 프롬프트.md).
6) 후속질의: 자연스러운 대화형(규칙 준수: OST 제외, 욕설 *** 마스킹, 인용 최소화 등).
7) GitHub 토큰 분리:
   - 세션 저장용: SESSION_GITHUB_*
   - 캐시 동기화(옵션): CACHE_GITHUB_*
"""

# ==========================================================
# region [1. Imports & Basic Setup]
# ==========================================================
import os
import re
import io
import csv
import time
import json
import base64
import random
import hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import requests
from streamlit.components.v1 import html as st_html

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai

KST = timezone(timedelta(hours=9))
BASE_DIR = "/tmp"
SESS_DIR = os.path.join(BASE_DIR, "sessions")
CACHE_DIR_DEFAULT = os.path.join(BASE_DIR, "yt_cache")
os.makedirs(SESS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR_DEFAULT, exist_ok=True)

PROMPT_FILE_1ST = "1차 질문 프롬프트.md"  # repo에 함께 두는 것을 권장
OST_EXCLUDE_RE = re.compile(r"\b(ost|o\.s\.t)\b", re.I)

# endregion


# ==========================================================
# region [2. Secrets / Config]
# ==========================================================
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        parts = re.split(r"[,\n]+", x)
        return [p.strip() for p in parts if p.strip()]
    return [str(x).strip()] if str(x).strip() else []

_YT_FALLBACK = _as_list(os.environ.get("YT_API_KEYS", ""))
_GEM_FALLBACK = _as_list(os.environ.get("GEMINI_API_KEYS", ""))

YT_API_KEYS     = _as_list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS = _as_list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK

# Gemini models (optional)
GEMINI_MODEL_REPORT = st.secrets.get("GEMINI_MODEL_REPORT", "gemini-3.0-flash-preview")
GEMINI_MODEL_CHAT   = st.secrets.get("GEMINI_MODEL_CHAT", "gemini-2.5-flash")

GEMINI_TIMEOUT = 180
GEMINI_MAX_TOKENS_REPORT = 8192
GEMINI_MAX_TOKENS_CHAT   = 2048

# GitHub (세션 저장용) — 별도 토큰/레포
SESSION_GITHUB_TOKEN  = st.secrets.get("SESSION_GITHUB_TOKEN", "") or os.environ.get("SESSION_GITHUB_TOKEN", "")
SESSION_GITHUB_REPO   = st.secrets.get("SESSION_GITHUB_REPO", "")  or os.environ.get("SESSION_GITHUB_REPO", "")
SESSION_GITHUB_BRANCH = st.secrets.get("SESSION_GITHUB_BRANCH", "main") or os.environ.get("SESSION_GITHUB_BRANCH", "main")

# GitHub (캐시 동기화용, 옵션) — 별도 토큰/레포
def _cfg(key: str, default: str = "") -> str:
    # secrets에 키가 존재하면(빈 문자열 포함) 그 값을 그대로 사용
    try:
        if key in st.secrets:
            v = st.secrets[key]
            return "" if v is None else str(v)
    except Exception:
        pass
    return str(os.environ.get(key, default) or default)

CACHE_GITHUB_TOKEN  = _cfg("CACHE_GITHUB_TOKEN", "")
CACHE_GITHUB_REPO   = _cfg("CACHE_GITHUB_REPO", "")
CACHE_GITHUB_BRANCH = _cfg("CACHE_GITHUB_BRANCH", "main")
# repo 루트면 빈 문자열("") 또는 "." 사용 가능
CACHE_GITHUB_PATH   = _cfg("CACHE_GITHUB_PATH", "")


# endregion


# ==========================================================
# region [3. Session State]
# ==========================================================
def ensure_state():
    defaults = {
        "chat": [],
        "report_html": "",
        "report_done": False,

        "last_schema": None,
        "last_csv": "",
        "last_df": None,
        "sample_text": "",

        "own_ip_enabled": False,
        "include_replies": True,

        # cache state
        "cache_dir": CACHE_DIR_DEFAULT,
        "cache_files": [],         # list of local paths
        "cache_loaded": False,
        "cache_video_rows": [],    # list of dicts with id/title/date/description

        "video_source_mode": "search",  # "search" | "own_ip_cache"
        "loaded_session_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()
# endregion


# ==========================================================
# region [4. Time Helpers]
# ==========================================================
def now_kst() -> datetime:
    return datetime.now(tz=KST)

def to_iso_kst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).isoformat(timespec="seconds")

def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None:
        dt_kst = dt_kst.replace(tzinfo=KST)
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_utc_to_kst(dt_str: str) -> datetime:
    if not dt_str:
        return now_kst()
    s = str(dt_str).strip()
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST)
    except Exception:
        return now_kst()

# endregion


# ==========================================================
# region [5. Rotating Keys & YouTube Client]
# ==========================================================
class RotatingKeys:
    def __init__(self, keys: List[str], idx_seed: str = "default"):
        self.keys = [k for k in keys if k]
        if not self.keys:
            self._idx = 0
        else:
            h = int(hashlib.md5(idx_seed.encode("utf-8")).hexdigest(), 16)
            self._idx = h % len(self.keys)

    def current(self) -> str:
        if not self.keys:
            return ""
        return self.keys[self._idx]

    def rotate(self) -> str:
        if not self.keys:
            return ""
        self._idx = (self._idx + 1) % len(self.keys)
        return self.current()

class RotatingYouTube:
    def __init__(self, keys: List[str], idx_seed: str = "yt"):
        self.rot = RotatingKeys(keys, idx_seed=idx_seed)
        self.service = None
        self._build()

    def _build(self):
        key = self.rot.current()
        if not key:
            raise RuntimeError("YouTube API Key가 비어 있습니다. (secrets/env 설정 필요)")
        self.service = build("youtube", "v3", developerKey=key)

    def execute(self, factory: Callable):
        attempts = max(1, len(self.rot.keys))
        last_err = None
        for _ in range(attempts):
            try:
                return factory(self.service).execute()
            except HttpError as e:
                last_err = e
                status = getattr(e, "status_code", None) or getattr(getattr(e, "resp", None), "status", None)
                msg = str(e).lower()
                quota_like = (status in (403, 429)) or ("quota" in msg) or ("rate" in msg) or ("userratelimit" in msg)
                if quota_like and len(self.rot.keys) > 1:
                    self.rot.rotate()
                    self._build()
                    time.sleep(0.25)
                    continue
                raise
        raise last_err

# endregion


# ==========================================================
# region [6. Gemini Helpers]
# ==========================================================
def _configure_genai(api_key: str):
    genai.configure(api_key=api_key)

def call_gemini_rotating(model_name: str, keys: List[str], system_prompt: Optional[str], user_prompt: str,
                        max_output_tokens: int, temperature: float = 0.3) -> str:
    if not keys:
        raise RuntimeError("Gemini API Key가 비어 있습니다. (secrets/env 설정 필요)")
    rk = RotatingKeys(keys, idx_seed=f"gem_{model_name}")
    last_err = None
    for _ in range(max(1, len(rk.keys))):
        try:
            _configure_genai(rk.current())
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt if system_prompt else None,
            )
            resp = model.generate_content(
                user_prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
                safety_settings={
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                },
                request_options={"timeout": GEMINI_TIMEOUT},
            )
            txt = (getattr(resp, "text", "") or "").strip()
            if txt:
                return txt
            try:
                return (resp.candidates[0].content.parts[0].text or "").strip()
            except Exception:
                return ""
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if ("429" in msg or "quota" in msg or "rate" in msg) and len(rk.keys) > 1:
                rk.rotate()
                time.sleep(0.25)
                continue
            raise
    raise last_err

# endregion


# ==========================================================
# region [7. Prompt Load & HTML Report Extraction]
# ==========================================================
def load_prompt_file(path: str) -> str:
    try:
        if os.path.exists(path):
            return open(path, "r", encoding="utf-8").read()
    except Exception:
        pass
    try:
        base = os.path.dirname(__file__)  # type: ignore
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return open(cand, "r", encoding="utf-8").read()
    except Exception:
        pass
    return ""

def extract_report_html(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"```html\s*|```", "", s, flags=re.I)
    m = re.search(r"<!--\s*REPORT_START\s*-->(.*?)<!--\s*REPORT_END\s*-->", s, flags=re.S | re.I)
    if not m:
        if "<div" in s and "</div>" in s:
            return s.strip()
        return None
    body = m.group(1).strip()
    body = "\n".join([ln.lstrip("\t ") for ln in body.splitlines()])
    return body.strip() if body else None

# endregion


# ==========================================================
# region [8. Light Schema Parsing]
# ==========================================================
LIGHT_PROMPT = r"""
너는 사용자의 자연어 요청을 'YouTube 댓글 수집/분석 스키마'로 변환하는 파서다.
아래 출력 포맷을 반드시 지켜라(6줄 고정). 날짜/시간은 KST 기준.
- 기간이 명확하지 않으면 최근 24시간으로 설정.
- 키워드는 핵심 1개(가장 중요한 것)만.
- 엔티티는 보조(작품/인물/채널/이슈 등) 0~3개.
- 옵션 include_replies: true/false만.

[출력 포맷]
start_kst: YYYY-MM-DD HH:MM:SS
end_kst:   YYYY-MM-DD HH:MM:SS
keywords:  <키워드 1개>
entities:  <엔티티1>, <엔티티2>, <엔티티3>
include_replies: true|false
note: <짧은 메모>

[사용자 요청]
{USER_QUERY}
""".strip()

def parse_light_block_to_schema(text: str) -> Dict:
    out = {"start_iso": "", "end_iso": "", "keywords": [], "entities": [], "options": {"include_replies": True}, "note": ""}
    if not text:
        end_dt = now_kst()
        start_dt = end_dt - timedelta(hours=24)
        out["start_iso"] = to_iso_kst(start_dt)
        out["end_iso"] = to_iso_kst(end_dt)
        return out

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kv = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
            kv[k.strip().lower()] = v.strip()

    def _parse_dt(v: str) -> datetime:
        v = v.strip()
        try:
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        except Exception:
            pass
        try:
            return datetime.fromisoformat(v).astimezone(KST)
        except Exception:
            return now_kst()

    end_dt = _parse_dt(kv.get("end_kst", "")) if kv.get("end_kst") else now_kst()
    start_dt = _parse_dt(kv.get("start_kst", "")) if kv.get("start_kst") else (end_dt - timedelta(hours=24))
    if start_dt > end_dt:
        start_dt = end_dt - timedelta(hours=24)

    kw = kv.get("keywords", "").strip()
    entities = kv.get("entities", "").strip()
    inc = kv.get("include_replies", "true").strip().lower()

    out["start_iso"] = to_iso_kst(start_dt)
    out["end_iso"] = to_iso_kst(end_dt)
    out["keywords"] = [kw] if kw else []
    out["entities"] = [e.strip() for e in entities.split(",") if e.strip()] if entities else []
    out["options"]["include_replies"] = (inc == "true")
    out["note"] = kv.get("note", "")
    return out

# endregion


# ==========================================================
# region [9. Cache Loader (ytan cache_token_*.json)]
# ==========================================================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).lower()

def scan_cache_files(cache_dir: str) -> List[str]:
    try:
        if not cache_dir or not os.path.exists(cache_dir):
            return []
        files = []
        for fn in os.listdir(cache_dir):
            if fn.startswith("cache_token_") and fn.endswith(".json"):
                files.append(os.path.join(cache_dir, fn))
        return sorted(files)
    except Exception:
        return []

def load_cache_videos(cache_paths: List[str]) -> List[Dict]:
    videos: List[Dict] = []
    for p in cache_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "videos" in data and isinstance(data["videos"], list):
                for v in data["videos"]:
                    if not isinstance(v, dict):
                        continue
                    vid = v.get("id") or v.get("video_id") or ""
                    if not vid:
                        continue
                    videos.append({
                        "id": vid,
                        "title": v.get("title", "") or "",
                        "date": v.get("date") or v.get("publishedAt") or "",
                        "description": v.get("description", "") or "",
                    })
            elif isinstance(data, list):
                for v in data:
                    if not isinstance(v, dict):
                        continue
                    vid = v.get("id") or v.get("video_id") or ""
                    if not vid:
                        continue
                    videos.append({
                        "id": vid,
                        "title": v.get("title", "") or "",
                        "date": v.get("date") or v.get("publishedAt") or "",
                        "description": v.get("description", "") or "",
                    })
        except Exception:
            continue

    seen = set()
    out = []
    for v in videos:
        if v["id"] in seen:
            continue
        out.append(v); seen.add(v["id"])
    return out

def filter_pgc_ids_from_cache(videos: List[Dict], keyword: str, start_dt: datetime, end_dt: datetime) -> List[str]:
    nk = normalize_text(keyword)
    if not nk:
        return []
    ids = []
    for v in videos:
        title = normalize_text(v.get("title", ""))
        desc = normalize_text(v.get("description", ""))
        if nk not in title and nk not in desc:
            continue
        v_dt = parse_utc_to_kst(v.get("date", ""))
        if start_dt.date() <= v_dt.date() <= end_dt.date():
            ids.append(v["id"])
    return list(dict.fromkeys(ids))

# endregion


# ==========================================================
# region [10. GitHub Cache Sync (optional)]
# ==========================================================
def _gh_headers(token: str) -> Dict[str, str]:
    # Fine-grained PAT은 Bearer 방식을 요구하는 경우가 많아 Bearer를 기본으로 사용
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ytcc-cache-sync",
    }

def github_list_dir(repo: str, branch: str, path_in_repo: str, token: str) -> List[Dict]:
    # GitHub Contents API: path는 URL 인코딩 필요(공백/한글 등 대비)
    safe_path = quote((path_in_repo or "").strip("/"))
    url = f"https://api.github.com/repos/{repo}/contents/{safe_path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(token))
    if r.status_code != 200:
        # Streamlit Cloud에서 r.text를 그대로 던지면 redaction이 걸려 디버깅이 어려움.
        # message만 안전하게 추출해서 에러로 올림.
        msg = ""
        try:
            j = r.json()
            if isinstance(j, dict):
                msg = j.get("message", "") or ""
        except Exception:
            msg = ""
        hint = ""
        if r.status_code in (401,):
            hint = " (401: 토큰이 없거나 권한/형식이 잘못됨 — Fine-grained PAT이면 repo 권한과 Bearer 형식 확인)"
        elif r.status_code in (403,):
            hint = " (403: 권한 부족 또는 rate limit — repo 접근 권한/limit 확인)"
        elif r.status_code in (404,):
            hint = " (404: repo/경로/브랜치가 틀림 — CACHE_GITHUB_REPO/PATH/BRANCH 확인)"
        raise RuntimeError(f"GitHub 목록 실패: {r.status_code} {msg}{hint}")
    data = r.json()
    return data if isinstance(data, list) else []

def github_download_file(download_url: str, token: str) -> bytes:
    r = requests.get(download_url, headers=_gh_headers(token))
    if r.status_code != 200:
        msg = ""
        try:
            j = r.json()
            if isinstance(j, dict):
                msg = j.get("message", "") or ""
        except Exception:
            msg = ""
        raise RuntimeError(f"GitHub 다운로드 실패: {r.status_code} {msg}")
    return r.content

def sync_cache_from_github(cache_dir: str) -> Tuple[bool, str, List[str]]:
    if not (CACHE_GITHUB_TOKEN and CACHE_GITHUB_REPO):
        return False, "CACHE_GITHUB_TOKEN / CACHE_GITHUB_REPO 설정이 없습니다.", []
    os.makedirs(cache_dir, exist_ok=True)
    items = github_list_dir(CACHE_GITHUB_REPO, CACHE_GITHUB_BRANCH, CACHE_GITHUB_PATH, CACHE_GITHUB_TOKEN)
    picked = [it for it in items if it.get("type") == "file" and str(it.get("name", "")).startswith("cache_token_") and str(it.get("name", "")).endswith(".json")]
    if not picked:
        return False, f"경로({CACHE_GITHUB_PATH})에서 cache_token_*.json 파일을 찾지 못했습니다.", []

    saved = []
    for it in picked:
        b = github_download_file(it.get("download_url", ""), CACHE_GITHUB_TOKEN)
        local_path = os.path.join(cache_dir, it.get("name", f"cache_{len(saved)}.json"))
        with open(local_path, "wb") as f:
            f.write(b)
        saved.append(local_path)
    return True, f"캐시 {len(saved)}개 다운로드 완료", saved

# endregion


# ==========================================================
# region [11. Video ID Sourcing]
# ==========================================================
def extract_video_ids_from_text(text: str) -> List[str]:
    if not text:
        return []
    ids = []
    for m in re.finditer(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", text):
        ids.append(m.group(1))
    for m in re.finditer(r"\b([A-Za-z0-9_-]{11})\b", text):
        ids.append(m.group(1))
    out = []
    for vid in ids:
        if vid not in out:
            out.append(vid)
    return out

def yt_search_videos(rt: RotatingYouTube, keyword: str, max_results: int,
                     order: str = "relevance",
                     published_after: Optional[str] = None,
                     published_before: Optional[str] = None) -> List[Dict]:
    items, token = [], None
    while len(items) < max_results:
        params = {"q": keyword, "part": "snippet", "type": "video", "order": order, "maxResults": min(50, max_results - len(items))}
        if published_after:
            params["publishedAfter"] = published_after
        if published_before:
            params["publishedBefore"] = published_before
        if token:
            params["pageToken"] = token

        resp = rt.execute(lambda s: s.search().list(**params))
        for it in resp.get("items", []):
            vid = it.get("id", {}).get("videoId", "")
            sn = it.get("snippet", {}) or {}
            if not vid:
                continue
            items.append({"video_id": vid, "title": sn.get("title", ""), "channel_id": sn.get("channelId", ""), "published_at": sn.get("publishedAt", "")})
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)

    seen = set()
    out = []
    for it in items:
        if it["video_id"] not in seen:
            out.append(it); seen.add(it["video_id"])
    return out

def build_video_pool(schema: Dict, own_ip_enabled: bool, cache_videos: List[Dict], extra_video_ids: Optional[List[str]] = None) -> Tuple[List[str], Dict]:
    extra_video_ids = list(dict.fromkeys(extra_video_ids or []))
    start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
    end_dt = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
    kw = (schema.get("keywords") or [""])[0].strip()

    published_after = kst_to_rfc3339_utc(start_dt)
    published_before = kst_to_rfc3339_utc(end_dt)
    rt = RotatingYouTube(YT_API_KEYS, idx_seed="yt_search")

    if own_ip_enabled:
        if not cache_videos:
            raise RuntimeError("자사 IP 모드 ON: cache_token_*.json 캐시가 없습니다. (캐시 업로드/동기화 후 다시 시도)")
        pgc_ids = filter_pgc_ids_from_cache(cache_videos, kw, start_dt, end_dt)
        if not pgc_ids:
            raise RuntimeError("자사 IP 모드 ON: 캐시에서 해당 키워드/기간의 자사(PGC) 영상을 찾지 못했습니다. (키워드/기간 확인)")

        ugc_ids = []
        if kw:
            hits = yt_search_videos(rt, kw, max_results=200, order="relevance", published_after=published_after, published_before=published_before)
            for h in hits:
                if h["video_id"] in pgc_ids:
                    continue
                if OST_EXCLUDE_RE.search(h.get("title", "") or ""):
                    continue
                ugc_ids.append(h["video_id"])
            ugc_ids = list(dict.fromkeys(ugc_ids))

        video_ids = list(dict.fromkeys(pgc_ids + ugc_ids + extra_video_ids))
        return video_ids, {"mode": "own_ip_cache", "pgc_count": len(pgc_ids), "ugc_count": len(ugc_ids),
                           "published_after": published_after, "published_before": published_before}

    # legacy search
    base_queries = []
    if kw:
        base_queries.append(kw)
        for e in (schema.get("entities") or [])[:3]:
            if e.strip():
                base_queries.extend([f"{kw} {e.strip()}", f"{e.strip()} {kw}"])
    else:
        base_queries.append("유튜브")

    all_hits = []
    for q in base_queries:
        all_hits.extend(yt_search_videos(rt, q, max_results=120, order="relevance", published_after=published_after, published_before=published_before))
    all_ids = [h["video_id"] for h in all_hits]
    video_ids = list(dict.fromkeys(all_ids + extra_video_ids))
    return video_ids, {"mode": "search", "pgc_count": 0, "ugc_count": 0, "published_after": published_after, "published_before": published_before}

# endregion


# ==========================================================
# region [12. Video Statistics]
# ==========================================================
def yt_video_statistics(rt: RotatingYouTube, video_ids: List[str]) -> List[Dict]:
    out = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        resp = rt.execute(lambda s: s.videos().list(part="snippet,statistics", id=",".join(chunk), maxResults=50))
        for it in resp.get("items", []):
            vid = it.get("id", "")
            sn = it.get("snippet", {}) or {}
            stt = it.get("statistics", {}) or {}
            out.append({
                "video_id": vid,
                "video_title": sn.get("title", ""),
                "channel_id": sn.get("channelId", ""),
                "publishedAt": sn.get("publishedAt", ""),
                "viewCount": int(stt.get("viewCount", 0) or 0),
                "likeCount": int(stt.get("likeCount", 0) or 0),
                "commentCount": int(stt.get("commentCount", 0) or 0),
            })
        time.sleep(0.1)
    return out

# endregion


# ==========================================================
# region [13. Deep Comment Collection (NO caps)]
# ==========================================================
def yt_all_replies(rt: RotatingYouTube, parent_id: str) -> List[Dict]:
    replies, token = [], None
    while True:
        try:
            resp = rt.execute(lambda s: s.comments().list(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText"))
        except HttpError:
            break
        for c in resp.get("items", []):
            sn = c.get("snippet", {}) or {}
            replies.append({"comment_id": c.get("id", ""), "parent_id": parent_id, "isReply": 1,
                            "author": sn.get("authorDisplayName", ""), "text": sn.get("textDisplay", "") or "",
                            "publishedAt": sn.get("publishedAt", ""), "likeCount": int(sn.get("likeCount", 0) or 0)})
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.15)
    return replies

def yt_all_comments_sync(rt: RotatingYouTube, video_id: str, title: str = "", include_replies: bool = True) -> List[Dict]:
    rows, token = [], None
    while True:
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(
                part="snippet,replies" if include_replies else "snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText",
                order="time",
            ))
        except HttpError:
            break

        for it in resp.get("items", []):
            top = (it.get("snippet", {}) or {}).get("topLevelComment", {}).get("snippet", {}) or {}
            thread_id = (it.get("snippet", {}) or {}).get("topLevelComment", {}).get("id", "")
            rows.append({"video_id": video_id, "video_title": title, "comment_id": thread_id, "parent_id": "", "isReply": 0,
                         "author": top.get("authorDisplayName", ""), "text": top.get("textDisplay", "") or "",
                         "publishedAt": top.get("publishedAt", ""), "likeCount": int(top.get("likeCount", 0) or 0)})
            if include_replies:
                total_reply = int((it.get("snippet", {}) or {}).get("totalReplyCount", 0) or 0)
                if total_reply > 0:
                    rows.extend([{"video_id": video_id, "video_title": title, **r} for r in yt_all_replies(rt, thread_id)])

        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.15)
    return rows

def parallel_collect_comments_streaming(video_rows: List[Dict], include_replies: bool, progress_bar=None) -> Tuple[str, int]:
    run_id = now_kst().strftime("%Y%m%d_%H%M%S") + "_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    csv_path = os.path.join(SESS_DIR, f"comments_{run_id}.csv")
    fieldnames = ["video_id", "video_title", "comment_id", "parent_id", "isReply", "author", "text", "publishedAt", "likeCount"]

    def _task(v: Dict) -> Tuple[str, int, List[Dict]]:
        vid = v["video_id"]
        title = v.get("video_title", "")
        rt_local = RotatingYouTube(YT_API_KEYS, idx_seed=f"yt_{vid}")
        rows = yt_all_comments_sync(rt_local, vid, title=title, include_replies=include_replies)
        return vid, len(rows), rows

    total = 0
    done = 0
    n = len(video_rows)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        max_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_task, v) for v in video_rows]
            for fu in as_completed(futures):
                _, cnt, rows = fu.result()
                for r in rows:
                    writer.writerow(r)
                total += cnt
                done += 1
                if progress_bar:
                    progress_bar.progress(min(0.85, done / n * 0.85), text=f"댓글 수집중… ({done}/{n}개 영상 완료, 누적 {total:,}개)")
    return csv_path, total

# endregion


# ==========================================================
# region [14. Sampling for LLM (ytcc 유지)]
# ==========================================================
def serialize_comments_for_llm_from_file(csv_path: str, max_chars_per_comment: int = 280, max_total_chars: int = 420_000) -> Tuple[str, int, int]:
    if not os.path.exists(csv_path):
        return "", 0, 0
    df_all = pd.read_csv(csv_path)
    if df_all.empty:
        return "", 0, 0

    df_top_likes = df_all.sort_values("likeCount", ascending=False).head(1000)
    df_remaining = df_all.drop(df_top_likes.index)
    df_random = df_remaining.sample(n=min(1000, len(df_remaining))) if not df_remaining.empty else pd.DataFrame()
    df = pd.concat([df_top_likes, df_random], ignore_index=True)

    lines, total_chars = [], 0
    for _, r in df.iterrows():
        text = str(r.get("text", "") or "").replace("\n", " ").strip()
        if not text:
            continue
        prefix = f"[{'R' if int(r.get('isReply', 0)) == 1 else 'T'}|♥{int(r.get('likeCount', 0))}] "
        author_clean = str(r.get('author', '')).replace('\n', ' ')
        body = text[:max_chars_per_comment] + '…' if len(text) > max_chars_per_comment else text
        line = prefix + f"{author_clean}: " + body
        if total_chars + len(line) + 1 > max_total_chars:
            break
        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines), len(lines), total_chars

# endregion


# ==========================================================
# region [15. Follow-up system prompt]
# ==========================================================
def system_prompt_followup() -> str:
    return (
        "너는 유튜브 댓글 데이터(샘플)에 근거해 질문에 답하는 대화형 분석가다.\n"
        "답변은 자연스럽고 대화체로, 한국어로 써라.\n\n"
        "[공통 규칙]\n"
        "1) OST/가수/음원/노래 관련 언급은 분석에서 완전 제외(언급하지도 말 것).\n"
        "2) 욕설/비속어/비하는 반드시 ***로 마스킹.\n"
        "3) 댓글 원문은 길게 붙이지 말고, 필요할 때만 짧게 1~3개 예시로 제시.\n"
        "4) 추정은 '가능성/경향'으로 표현하고, 확정적으로 단정하지 말 것.\n"
        "5) 사용자 질문 범위를 벗어나지 말고, 데이터(샘플)에서 근거를 잡아라.\n"
    )

# endregion


# ==========================================================
# region [16. Pipeline: First report + Follow-up chat]
# ==========================================================
def build_first_turn_payload(schema: Dict, video_meta: Dict, sample_text: str) -> str:
    start_dt = datetime.fromisoformat(schema["start_iso"]).astimezone(KST)
    end_dt = datetime.fromisoformat(schema["end_iso"]).astimezone(KST)
    kw_main = schema.get("keywords", [])
    ents = schema.get("entities", [])
    mode = video_meta.get("mode", "search")
    if mode == "own_ip_cache":
        target_str = f"자사 캐시(PGC) {video_meta.get('pgc_count', 0)}개 + 외부 UGC {video_meta.get('ugc_count', 0)}개 (OST 제외)"
    else:
        target_str = "검색 기반(전체 유튜브)"
    return (
        f"분석 주제(키워드): {kw_main[0] if kw_main else '(없음)'}\n"
        f"보조 엔티티: {', '.join(ents) if ents else '(없음)'}\n"
        f"분석 기간(KST): {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"영상 소스: {target_str}\n"
        f"댓글 데이터 샘플:\n{sample_text}\n"
    )

def run_pipeline_first_turn(user_query: str, own_ip_enabled: bool, include_replies: bool) -> str:
    prog = st.progress(0, text="준비 중…")
    if not GEMINI_API_KEYS:
        return "오류: GEMINI_API_KEYS가 설정되지 않았습니다."
    if not YT_API_KEYS:
        return "오류: YT_API_KEYS가 설정되지 않았습니다."

    prog.progress(0.06, text="질문 해석중…")
    light = call_gemini_rotating(GEMINI_MODEL_CHAT, GEMINI_API_KEYS, None, LIGHT_PROMPT.replace("{USER_QUERY}", user_query), 512, 0.1)
    schema = parse_light_block_to_schema(light)
    st.session_state["last_schema"] = schema

    # Cache-only requirement
    cache_videos = st.session_state.get("cache_video_rows", []) if own_ip_enabled else []
    prog.progress(0.12, text="영상 ID 수집중…")
    extra_ids = extract_video_ids_from_text(user_query)

    try:
        video_ids, video_meta = build_video_pool(schema, own_ip_enabled, cache_videos, extra_video_ids=extra_ids)
    except Exception as e:
        return f"오류: {e}"
    if not video_ids:
        return "오류: 영상 ID를 찾지 못했습니다."

    prog.progress(0.30, text="영상 메타 수집중…")
    rt = RotatingYouTube(YT_API_KEYS, idx_seed="yt_stats")
    stats = yt_video_statistics(rt, video_ids)
    df_videos = pd.DataFrame(stats)
    st.session_state["last_df"] = df_videos

    prog.progress(0.40, text="댓글 수집 시작… (상한 없음)")
    csv_path, _ = parallel_collect_comments_streaming(df_videos.to_dict("records"), include_replies=include_replies, progress_bar=prog)
    st.session_state["last_csv"] = csv_path

    prog.progress(0.88, text="샘플링 구성중…")
    sample_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
    st.session_state["sample_text"] = sample_text

    prog.progress(0.92, text="AI 리포트 생성중… (HTML)")
    sys_prompt = load_prompt_file(PROMPT_FILE_1ST).strip()
    if not sys_prompt:
        return "오류: 1차 질문 프롬프트.md 파일을 찾지 못했습니다."
    sys_prompt += "\n\n추가 규칙: 반드시 <!--REPORT_START--> 와 <!--REPORT_END--> 마커로 리포트 전체를 감싸라."

    raw = call_gemini_rotating(GEMINI_MODEL_REPORT, GEMINI_API_KEYS, sys_prompt, build_first_turn_payload(schema, video_meta, sample_text), GEMINI_MAX_TOKENS_REPORT, 0.25)
    html_report = extract_report_html(raw) or f"<div>{raw}</div>"

    st.session_state["report_html"] = html_report
    st.session_state["report_done"] = True
    st.session_state["video_source_mode"] = video_meta.get("mode", "search")
    return html_report

def run_followup_turn(user_message: str) -> str:
    schema = st.session_state.get("last_schema") or {}
    sample_text = st.session_state.get("sample_text", "") or ""
    meta = (
        f"[분석 주제] {(schema.get('keywords') or [''])[0]}\n"
        f"[기간(KST ISO)] {schema.get('start_iso','')} ~ {schema.get('end_iso','')}\n"
        f"[영상 소스 모드] {st.session_state.get('video_source_mode','search')}\n"
    )
    history = st.session_state.get("chat", [])
    recent = []
    for m in history[-6:]:
        if m.get("role") == "assistant" and (m.get("is_report") or False):
            continue
        txt = m.get("content", "")
        if len(txt) > 2000:
            txt = txt[:2000] + "…"
        recent.append(f"{m.get('role','user')}: {txt}")
    prompt = f"{meta}\n[최근 대화]\n" + "\n".join(recent) + f"\n\n[댓글 샘플]\n{sample_text}\n\n[사용자 질문]\n{user_message}\n"
    ans = call_gemini_rotating(GEMINI_MODEL_CHAT, GEMINI_API_KEYS, system_prompt_followup(), prompt, GEMINI_MAX_TOKENS_CHAT, 0.35)
    return (ans or "").strip()

# endregion


# ==========================================================
# region [17. Downloads + Session Save]
# ==========================================================
def _safe_read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""

def render_downloads():
    csv_path = st.session_state.get("last_csv", "")
    df_videos = st.session_state.get("last_df")
    if csv_path and os.path.exists(csv_path):
        st.download_button("⬇️ 댓글 CSV 다운로드", data=_safe_read_bytes(csv_path), file_name=os.path.basename(csv_path), mime="text/csv", use_container_width=True)
    if isinstance(df_videos, pd.DataFrame) and not df_videos.empty:
        buf = io.BytesIO()
        df_videos.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 영상 메타 CSV 다운로드", data=buf.getvalue(), file_name=f"videos_{now_kst().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)

def github_upload_bytes(repo: str, branch: str, path_in_repo: str, content_bytes: bytes, token: str, message: str):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    payload = {"message": message, "content": base64.b64encode(content_bytes).decode("utf-8"), "branch": branch}
    r0 = requests.get(url, headers=headers)
    if r0.status_code == 200:
        sha = (r0.json() or {}).get("sha")
        if sha:
            payload["sha"] = sha
    r = requests.put(url, headers=headers, data=json.dumps(payload))
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub 업로드 실패: {r.status_code} {r.text}")

def save_session_to_github():
    if not (SESSION_GITHUB_TOKEN and SESSION_GITHUB_REPO):
        return False, "SESSION_GITHUB_TOKEN / SESSION_GITHUB_REPO 설정이 없습니다."
    if not st.session_state.get("last_csv") or not os.path.exists(st.session_state["last_csv"]):
        return False, "저장할 댓글 CSV가 없습니다."
    df = st.session_state.get("last_df")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, "저장할 영상 메타가 없습니다."

    sess_name = now_kst().strftime("%Y%m%d_%H%M%S") + "_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    base = f"sessions/{sess_name}"

    qa = {"chat": st.session_state.get("chat", []),
          "last_schema": st.session_state.get("last_schema"),
          "report_html": st.session_state.get("report_html", ""),
          "video_source_mode": st.session_state.get("video_source_mode", "search")}
    github_upload_bytes(SESSION_GITHUB_REPO, SESSION_GITHUB_BRANCH, f"{base}/qa.json", json.dumps(qa, ensure_ascii=False, indent=2).encode("utf-8"), SESSION_GITHUB_TOKEN, f"save session {sess_name} (qa)")
    github_upload_bytes(SESSION_GITHUB_REPO, SESSION_GITHUB_BRANCH, f"{base}/comments.csv", _safe_read_bytes(st.session_state["last_csv"]), SESSION_GITHUB_TOKEN, f"save session {sess_name} (comments)")
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    github_upload_bytes(SESSION_GITHUB_REPO, SESSION_GITHUB_BRANCH, f"{base}/videos.csv", buf.getvalue(), SESSION_GITHUB_TOKEN, f"save session {sess_name} (videos)")
    return True, f"세션 저장 완료: {sess_name}"

# endregion


# ==========================================================
# region [18. Main UI]
# ==========================================================
st.set_page_config(page_title="ytcc (CACHE-ONLY Own-IP)", page_icon="💬", layout="wide")
st.title("💬 ytcc — CACHE-ONLY Own-IP")
st.caption("자사 IP 토글 ON은 ytan 캐시(cache_token_*.json) 로드가 필수. 캐시 없으면 실행 불가.")

with st.sidebar:
    st.subheader("⚙️ 수집 설정")
    st.session_state["include_replies"] = st.checkbox("대댓글(Reply) 포함", value=bool(st.session_state.get("include_replies", True)))

    st.divider()
    st.subheader("🏷️ 자사 IP 모드 (CACHE ONLY)")
    st.session_state["own_ip_enabled"] = st.toggle("자사 IP (캐시 PGC + UGC 혼합)", value=bool(st.session_state.get("own_ip_enabled", False)))
    st.caption("자사 IP 모드 ON이면 cache_token_*.json이 반드시 필요합니다.")

    cache_dir = st.text_input("캐시 디렉토리(서버 로컬)", value=st.session_state.get("cache_dir", CACHE_DIR_DEFAULT))
    st.session_state["cache_dir"] = cache_dir

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 캐시 스캔", use_container_width=True):
            files = scan_cache_files(cache_dir)
            st.session_state["cache_files"] = files
            vids = load_cache_videos(files)
            st.session_state["cache_video_rows"] = vids
            st.session_state["cache_loaded"] = True
            st.success(f"캐시 파일 {len(files)}개 / 영상 {len(vids)}개 로드")
    with col2:
        if st.button("☁️ GitHub 캐시 동기화", use_container_width=True, disabled=not (CACHE_GITHUB_TOKEN and CACHE_GITHUB_REPO)):
            try:
                ok, msg, saved = sync_cache_from_github(cache_dir)
            except Exception as e:
                ok, msg, saved = False, str(e), []
            if ok:
                st.success(msg)
                st.session_state["cache_files"] = saved
                vids = load_cache_videos(saved)
                st.session_state["cache_video_rows"] = vids
                st.session_state["cache_loaded"] = True
                st.info(f"캐시 영상 {len(vids)}개")
            else:
                st.error(msg)

    uploaded = st.file_uploader("cache_token_*.json 업로드(복수 가능)", type=["json"], accept_multiple_files=True)
    if uploaded:
        os.makedirs(cache_dir, exist_ok=True)
        saved = []
        for uf in uploaded:
            name = uf.name
            if not (name.startswith("cache_token_") and name.endswith(".json")):
                name = f"cache_token_uploaded_{hashlib.md5(name.encode()).hexdigest()[:8]}.json"
            local = os.path.join(cache_dir, name)
            with open(local, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(local)
        st.session_state["cache_files"] = sorted(list(dict.fromkeys(st.session_state.get("cache_files", []) + saved)))
        vids = load_cache_videos(st.session_state["cache_files"])
        st.session_state["cache_video_rows"] = vids
        st.session_state["cache_loaded"] = True
        st.success(f"업로드 반영: 캐시 파일 {len(st.session_state['cache_files'])}개 / 영상 {len(vids)}개")

    if st.session_state.get("cache_loaded"):
        st.caption(f"현재 로드된 캐시 영상 수: {len(st.session_state.get('cache_video_rows', [])):,}")

    st.divider()
    st.subheader("💾 Export / Save")
    render_downloads()
    if st.button("☁️ GitHub에 세션 저장", use_container_width=True, disabled=not (SESSION_GITHUB_TOKEN and SESSION_GITHUB_REPO)):
        ok, msg = save_session_to_github()
        (st.success if ok else st.error)(msg)

st.divider()

if not st.session_state.get("report_done", False):
    st.markdown("### 1) 분석 질문을 입력하세요")
    user_query = st.text_area("요청(자연어)", height=120)
    run = st.button("🚀 분석 시작", type="primary", use_container_width=True)

    if run and user_query.strip():
        if st.session_state.get("own_ip_enabled", False) and not st.session_state.get("cache_video_rows"):
            st.error("자사 IP 모드 ON: cache_token_*.json 캐시를 먼저 로드/업로드/동기화하세요.")
            st.stop()

        html_report = run_pipeline_first_turn(user_query.strip(), bool(st.session_state.get("own_ip_enabled", False)), bool(st.session_state.get("include_replies", True)))
        st.session_state["chat"] = [{"role": "user", "content": user_query.strip()}, {"role": "assistant", "content": html_report, "is_report": True}]
        st.rerun()
else:
    st.markdown("### ✅ 첫 리포트 (HTML)")
    report_html = st.session_state.get("report_html", "")
    if report_html:
        st_html(report_html, height=820, scrolling=True)

    st.divider()
    st.markdown("### 2) 후속 질문 (대화형)")
    for msg in st.session_state.get("chat", []):
        if msg.get("role") == "assistant" and msg.get("is_report"):
            continue
        with st.chat_message(msg.get("role", "user")):
            st.write(msg.get("content", ""))

    user_msg = st.chat_input("후속 질문을 입력하세요")
    if user_msg:
        st.session_state["chat"].append({"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            with st.spinner("답변 생성중…"):
                ans = run_followup_turn(user_msg)
                st.write(ans)
        st.session_state["chat"].append({"role": "assistant", "content": ans})

    st.divider()
    if st.button("🔄 세션 초기화(처음으로)", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        ensure_state()
        st.rerun()

# endregion
