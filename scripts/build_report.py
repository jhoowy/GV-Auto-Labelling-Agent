"""Build the self-contained dataset statistics report (EN + KR) into reports/.

    python scripts/build_report.py

Reads data/manifest/metadata.jsonl (+ ingest_ready.jsonl for the source query)
and blobs/thumbnails/. Output HTML files under reports/ are git-excluded.
"""
from __future__ import annotations

import base64
import collections
import html
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "manifest" / "metadata.jsonl"
MANI = ROOT / "data" / "manifest" / "ingest_ready.jsonl"
THUMB = ROOT / "blobs" / "thumbnails"
OUT_DIR = ROOT / "reports"

meta = {json.loads(l)["video_id"]: json.loads(l) for l in META.read_text().splitlines() if l.strip()}
mani = {json.loads(l)["video_id"]: json.loads(l) for l in MANI.read_text().splitlines() if l.strip()}
vids = list(meta)
N = len(vids)


def lang_of(v):
    return (meta[v].get("default_audio_language") or meta[v].get("default_language") or "unknown").split("-")[0]


durs = [meta[v]["duration_s"] for v in vids if meta[v].get("duration_s")]
total_hours = sum(durs) / 3600
median = sorted(durs)[len(durs) // 2]
lang_c = collections.Counter(lang_of(v) for v in vids)
cat_c = collections.Counter(meta[v].get("category") or "?" for v in vids)
query_c = collections.Counter(mani[v].get("query") for v in vids if v in mani)
tags_c = collections.Counter()
for v in vids:
    for t in (meta[v].get("tags") or []):
        tags_c[t.lower()] += 1

DUR_BUCKETS = [(0, 60, "<1m"), (60, 180, "1–3m"), (180, 300, "3–5m"),
               (300, 600, "5–10m"), (600, 1200, "10–20m"), (1200, 10 ** 9, ">20m")]
dur_hist = collections.Counter()
for d in durs:
    for lo, hi, lab in DUR_BUCKETS:
        if lo <= d < hi:
            dur_hist[lab] += 1
            break

by_query: dict[str, list[str]] = collections.defaultdict(list)
for v in vids:
    if v in mani:
        by_query[mani[v]["query"]].append(v)
samples = []
for q in sorted(by_query, key=lambda k: -query_c[k]):
    best = max(by_query[q], key=lambda v: meta[v].get("view_count") or 0)
    samples.append((q, best))

PALETTE = ["#3b6ea5", "#4f9d8b", "#d99630", "#b5643f", "#7a6ea8", "#8592a2"]

STRINGS = {
    "en": {
        "lang_label": "English", "other_lang": "한국어", "other_file": "dataset_report.ko.html",
        "title": "Dataset Report — Gameplay Moderation PoC",
        "eyebrow": "Auto-labelling PoC · Dataset snapshot",
        "h1": "Gameplay video collection — statistics report",
        "intro": ("Bootstrap dataset for shot-level content-moderation labelling "
                  "(PEGI subset: gambling, bad language, sex). Videos gathered by "
                  "game-title search across Korean and English queries, downloaded, "
                  "and enriched with full metadata."),
        "m_videos": "videos", "m_hours": "hours", "m_queries": "queries", "m_langs": "languages",
        "m_meta": "metadata: title · description · tags · channel · language · thumbnail",
        "s_videos": "Videos collected", "s_videos_sub": "downloaded + metadata",
        "s_footage": "Total footage", "s_footage_sub": "median {med} each",
        "s_queries": "Search queries", "s_queries_sub": "KR + EN game titles",
        "s_langs": "Languages", "s_langs_sub": "{ko} KR / {en} EN",
        "s_gaming": "Gaming category", "s_gaming_sub": "{g} of {n}",
        "s_desc": "With description", "s_desc_sub": "{t} with tags",
        "p_dur": "Duration distribution", "p_dur_cap": "clip length, {n} videos · median {med}",
        "p_lang": "Spoken language", "p_lang_cap": "default audio language (video metadata)",
        "p_cat": "Video category", "p_cat_cap": "predominantly gaming footage",
        "p_tags": "Top tags", "p_tags_cap": "most frequent creator tags across the set",
        "p_query": "Collection query",
        "p_query_cap": "search term each video was sourced from ({q} queries · KR + EN game titles)",
        "samples_h": "Representative samples", "samples_sub": "most-viewed video per query · {n} thumbnails",
        "foot_l": "Content-moderation auto-labelling PoC — dataset snapshot",
        "noimg": "no thumbnail", "views": "views",
        "langs": {"en": "English", "ko": "Korean", "ja": "Japanese", "vi": "Vietnamese",
                  "zxx": "No speech", "unknown": "Unknown"},
    },
    "ko": {
        "lang_label": "한국어", "other_lang": "English", "other_file": "dataset_report.en.html",
        "title": "데이터셋 리포트 — 게임플레이 모더레이션 PoC",
        "eyebrow": "자동 라벨링 PoC · 데이터셋 스냅샷",
        "h1": "게임플레이 영상 수집 — 통계 리포트",
        "intro": ("shot 단위 콘텐츠 모더레이션 라벨링을 위한 부트스트랩 데이터셋 "
                  "(PEGI 서브셋: 도박, 저속어, 성적 표현). 한국어·영어 게임 타이틀 "
                  "검색으로 수집·다운로드하고 전체 메타데이터로 보강했다."),
        "m_videos": "영상", "m_hours": "시간", "m_queries": "쿼리", "m_langs": "언어",
        "m_meta": "메타데이터: 제목 · 설명 · 태그 · 채널 · 언어 · 썸네일",
        "s_videos": "수집 영상", "s_videos_sub": "다운로드 + 메타데이터",
        "s_footage": "총 영상 길이", "s_footage_sub": "중앙값 {med}",
        "s_queries": "검색 쿼리", "s_queries_sub": "한/영 게임 타이틀",
        "s_langs": "언어", "s_langs_sub": "한국어 {ko} / 영어 {en}",
        "s_gaming": "게임 카테고리", "s_gaming_sub": "{n}개 중 {g}개",
        "s_desc": "설명 포함", "s_desc_sub": "태그 포함 {t}개",
        "p_dur": "길이 분포", "p_dur_cap": "클립 길이, {n}개 영상 · 중앙값 {med}",
        "p_lang": "음성 언어", "p_lang_cap": "기본 오디오 언어 (영상 메타데이터)",
        "p_cat": "영상 카테고리", "p_cat_cap": "대부분 게임 영상",
        "p_tags": "상위 태그", "p_tags_cap": "데이터셋 전체에서 가장 빈번한 크리에이터 태그",
        "p_query": "수집 쿼리",
        "p_query_cap": "각 영상이 수집된 검색어 ({q}개 쿼리 · 한/영 게임 타이틀)",
        "samples_h": "대표 샘플", "samples_sub": "쿼리별 최다 조회 영상 · 썸네일 {n}개",
        "foot_l": "콘텐츠 모더레이션 자동 라벨링 PoC — 데이터셋 스냅샷",
        "noimg": "썸네일 없음", "views": "조회",
        "langs": {"en": "영어", "ko": "한국어", "ja": "일본어", "vi": "베트남어",
                  "zxx": "음성 없음", "unknown": "미상"},
    },
}


def esc(s):
    return html.escape(str(s if s is not None else ""))


def fmt_dur(s):
    s = int(s or 0)
    return f"{s // 60}:{s % 60:02d}"


_thumb_cache: dict[str, str | None] = {}


def thumb_data_uri(vid: str, width: int = 360) -> str | None:
    if vid in _thumb_cache:
        return _thumb_cache[vid]
    src = THUMB / f"{vid}.jpg"
    if not src.exists():
        _thumb_cache[vid] = None
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        dst = tf.name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"scale={width}:-1", "-q:v", "5", dst], check=True)
    b = base64.b64encode(Path(dst).read_bytes()).decode()
    Path(dst).unlink()
    uri = f"data:image/jpeg;base64,{b}"
    _thumb_cache[vid] = uri
    return uri


def hbars(counter, names=None, palette=False):
    mx = max(counter.values()) if counter else 1
    rows = []
    for i, (k, val) in enumerate(counter.most_common()):
        label = names.get(k, k) if names else k
        color = PALETTE[i % len(PALETTE)] if palette else "var(--accent)"
        rows.append(
            f'<div class="row"><div class="rlabel" title="{esc(label)}">{esc(label)}</div>'
            f'<div class="rtrack"><div class="rbar" style="width:{val / mx * 100:.1f}%;background:{color}"></div></div>'
            f'<div class="rval"><b>{val}</b><span>{val / N * 100:.0f}%</span></div></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def vbars(hist):
    mx = max(hist.values()) if hist else 1
    cols = []
    for _, _, lab in DUR_BUCKETS:
        val = hist.get(lab, 0)
        cols.append(
            f'<div class="vcol"><div class="vwrap"><div class="vnum">{val}</div>'
            f'<div class="vbar" style="height:{max(val / mx * 100, 1.5):.1f}%"></div></div>'
            f'<div class="vlab">{lab}</div></div>'
        )
    return '<div class="vbars">' + "".join(cols) + "</div>"


CSS = """
*{box-sizing:border-box}
:root{--bg:#eef1f5;--surface:#fff;--surface2:#f5f8fb;--border:#dde3ec;--ink:#17202b;
 --muted:#5a6675;--faint:#8592a2;--accent:#c17d1c;--radius:10px;
 --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#0e1217;--surface:#151b23;--surface2:#1b222c;
 --border:#28313d;--ink:#e8edf3;--muted:#96a2b1;--faint:#616d7c;--accent:#e0a94a}}
:root[data-theme="light"]{--bg:#eef1f5;--surface:#fff;--surface2:#f5f8fb;--border:#dde3ec;
 --ink:#17202b;--muted:#5a6675;--faint:#8592a2;--accent:#c17d1c}
:root[data-theme="dark"]{--bg:#0e1217;--surface:#151b23;--surface2:#1b222c;--border:#28313d;
 --ink:#e8edf3;--muted:#96a2b1;--faint:#616d7c;--accent:#e0a94a}
html{color-scheme:light dark}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:clamp(20px,4vw,52px) clamp(16px,3vw,32px) 72px}
.langbar{font-family:var(--mono);font-size:12px;color:var(--faint);margin-bottom:18px}
.langbar a{color:var(--accent)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;
 color:var(--accent);font-weight:600}
header.top{border-bottom:1px solid var(--border);padding-bottom:26px;margin-bottom:30px}
header.top h1{font-size:clamp(26px,4.2vw,40px);line-height:1.08;margin:10px 0 8px;
 letter-spacing:-.02em;text-wrap:balance;font-weight:680}
header.top p{color:var(--muted);max-width:64ch;margin:0}
.meta-line{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px 18px;font-family:var(--mono);
 font-size:12.5px;color:var(--faint)}
.meta-line b{color:var(--muted);font-weight:600}
.grid-sum{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:14px;margin-bottom:34px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px}
.stat .big{font-size:30px;font-weight:600;letter-spacing:-.02em;color:var(--ink);
 font-variant-numeric:tabular-nums}
.stat .slab{margin-top:3px;font-size:13.5px;color:var(--ink);font-weight:560}
.stat .sub{font-size:11.5px;color:var(--faint);font-family:var(--mono);margin-top:2px}
section.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
 padding:22px 24px;margin-bottom:18px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:720px){.panels{grid-template-columns:1fr}}
.panel h2{font-size:15px;margin:0 0 2px;letter-spacing:-.01em}
.panel .cap{font-size:12.5px;color:var(--faint);margin:0 0 18px;font-family:var(--mono)}
.bars{display:flex;flex-direction:column;gap:9px}
.row{display:grid;grid-template-columns:132px 1fr 66px;align-items:center;gap:12px}
.rlabel{font-size:13px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rtrack{background:var(--surface2);border-radius:5px;height:16px;overflow:hidden}
.rbar{height:100%;border-radius:5px;min-width:2px}
.rval{font-family:var(--mono);font-size:12.5px;text-align:right;color:var(--faint);
 font-variant-numeric:tabular-nums}
.rval b{color:var(--ink);font-weight:600}.rval span{margin-left:5px}
.vbars{display:flex;align-items:flex-end;gap:10px;height:180px;padding-top:8px}
.vcol{flex:1;display:flex;flex-direction:column;align-items:center;height:100%;gap:8px}
.vwrap{flex:1;width:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:6px}
.vbar{width:74%;max-width:52px;background:linear-gradient(180deg,var(--accent),color-mix(in srgb,var(--accent) 70%,transparent));
 border-radius:5px 5px 0 0;min-height:3px}
.vnum{font-family:var(--mono);font-size:12.5px;color:var(--muted);font-weight:600}
.vlab{font-family:var(--mono);font-size:11.5px;color:var(--faint)}
.tags{display:flex;flex-wrap:wrap;gap:8px}
.tag{font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--surface2);
 border:1px solid var(--border);border-radius:20px;padding:5px 10px;display:inline-flex;
 align-items:center;gap:7px;opacity:calc(.62 + .38*var(--w))}
.tag i{font-style:normal;color:var(--accent);font-weight:600}
.sec-head{display:flex;align-items:baseline;justify-content:space-between;margin:38px 0 16px;
 gap:16px;flex-wrap:wrap}
.sec-head h2{font-size:19px;margin:0;letter-spacing:-.01em}
.sec-head p{margin:0;color:var(--faint);font-size:12.5px;font-family:var(--mono)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(214px,1fr));gap:16px}
.sample{margin:0;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
 overflow:hidden;display:flex;flex-direction:column}
.sample img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:var(--surface2)}
.sample .noimg{aspect-ratio:16/9;display:grid;place-items:center;color:var(--faint);
 font-family:var(--mono);font-size:12px;background:var(--surface2)}
.sample figcaption{padding:11px 13px 13px;display:flex;flex-direction:column;gap:5px}
.squery{font-family:var(--mono);font-size:11px;letter-spacing:.04em;color:var(--accent);
 text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stitle{font-size:13px;font-weight:560;line-height:1.35;color:var(--ink);
 display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.smeta{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin-top:2px}
.smeta .mono{font-family:var(--mono);font-size:11px;color:var(--faint);font-variant-numeric:tabular-nums}
.chip{font-family:var(--mono);font-size:10.5px;color:var(--muted);background:var(--surface2);
 border:1px solid var(--border);border-radius:4px;padding:1px 6px}
footer{margin-top:44px;padding-top:20px;border-top:1px solid var(--border);color:var(--faint);
 font-family:var(--mono);font-size:11.5px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
"""


def render(lang: str) -> str:
    t = STRINGS[lang]
    med = fmt_dur(median)
    summary = "".join([
        f'<div class="stat"><div class="big mono">{N}</div><div class="slab">{t["s_videos"]}</div><div class="sub">{t["s_videos_sub"]}</div></div>',
        f'<div class="stat"><div class="big mono">{total_hours:.1f}h</div><div class="slab">{t["s_footage"]}</div><div class="sub">{t["s_footage_sub"].format(med=med)}</div></div>',
        f'<div class="stat"><div class="big mono">{len(query_c)}</div><div class="slab">{t["s_queries"]}</div><div class="sub">{t["s_queries_sub"]}</div></div>',
        f'<div class="stat"><div class="big mono">{len(lang_c)}</div><div class="slab">{t["s_langs"]}</div><div class="sub">{t["s_langs_sub"].format(ko=lang_c.get("ko",0),en=lang_c.get("en",0))}</div></div>',
        f'<div class="stat"><div class="big mono">{cat_c.get("Gaming",0)*100//N}%</div><div class="slab">{t["s_gaming"]}</div><div class="sub">{t["s_gaming_sub"].format(g=cat_c.get("Gaming",0),n=N)}</div></div>',
        f'<div class="stat"><div class="big mono">{sum(1 for v in vids if (meta[v].get("description") or "").strip())}</div><div class="slab">{t["s_desc"]}</div><div class="sub">{t["s_desc_sub"].format(t=sum(1 for v in vids if meta[v].get("tags")))}</div></div>',
    ])
    tmx = tags_c.most_common(1)[0][1] if tags_c else 1
    tag_chips = "".join(f'<span class="tag" style="--w:{v/tmx:.2f}">{esc(tag)}<i>{v}</i></span>'
                        for tag, v in tags_c.most_common(18))
    cards = []
    for q, vid in samples:
        uri = thumb_data_uri(vid)
        m = meta[vid]
        title = m.get("title") or vid
        img = (f'<img src="{uri}" alt="{esc(title)}" loading="lazy">' if uri
               else f'<div class="noimg">{t["noimg"]}</div>')
        cards.append(
            f'<figure class="sample">{img}<figcaption>'
            f'<div class="squery">{esc(q)}</div>'
            f'<div class="stitle" title="{esc(title)}">{esc(title)}</div>'
            f'<div class="smeta"><span class="chip">{esc(t["langs"].get(lang_of(vid), lang_of(vid)))}</span>'
            f'<span class="mono">{fmt_dur(m.get("duration_s"))}</span>'
            f'<span class="mono">{(m.get("view_count") or 0):,} {t["views"]}</span>'
            f'</div></figcaption></figure>'
        )
    return f"""<title>{t["title"]}</title>
<style>{CSS}</style>
<div class="wrap">
<div class="langbar">{t["lang_label"]} · <a href="{t["other_file"]}">{t["other_lang"]}</a></div>
<header class="top">
  <div class="eyebrow">{t["eyebrow"]}</div>
  <h1>{t["h1"]}</h1>
  <p>{t["intro"]}</p>
  <div class="meta-line">
    <span><b>{N}</b> {t["m_videos"]}</span><span><b>{total_hours:.1f}</b> {t["m_hours"]}</span>
    <span><b>{len(query_c)}</b> {t["m_queries"]}</span><span><b>{len(lang_c)}</b> {t["m_langs"]}</span>
    <span>{t["m_meta"]}</span>
  </div>
</header>
<div class="grid-sum">{summary}</div>
<div class="panels">
  <section class="panel"><h2>{t["p_dur"]}</h2><p class="cap">{t["p_dur_cap"].format(n=N,med=med)}</p>{vbars(dur_hist)}</section>
  <section class="panel"><h2>{t["p_lang"]}</h2><p class="cap">{t["p_lang_cap"]}</p>{hbars(lang_c, names=t["langs"], palette=True)}</section>
  <section class="panel"><h2>{t["p_cat"]}</h2><p class="cap">{t["p_cat_cap"]}</p>{hbars(cat_c, palette=True)}</section>
  <section class="panel"><h2>{t["p_tags"]}</h2><p class="cap">{t["p_tags_cap"]}</p><div class="tags">{tag_chips}</div></section>
</div>
<section class="panel"><h2>{t["p_query"]}</h2><p class="cap">{t["p_query_cap"].format(q=len(query_c))}</p>{hbars(query_c)}</section>
<div class="sec-head"><h2>{t["samples_h"]}</h2><p>{t["samples_sub"].format(n=len(samples))}</p></div>
<div class="gallery">{''.join(cards)}</div>
<footer><span>{t["foot_l"]}</span><span>{N} · {total_hours:.1f}h · {len(query_c)} {t["m_queries"]}</span></footer>
</div>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "ko"):
        path = OUT_DIR / f"dataset_report.{lang}.html"
        path.write_text(render(lang), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
