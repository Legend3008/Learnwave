"""
URL Content Fetcher Service
Server-side URL fetching with specialized handlers for YouTube, Wikipedia, arXiv, GitHub, and general websites.
Works exactly like NotebookLM — the backend does all the heavy lifting.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Shared session with browser-like headers
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})
TIMEOUT = 15


@dataclass
class FetchResult:
    title: str
    content: str
    word_count: int
    url: str
    url_type: str


def classify_url(url: str) -> str:
    """Classify a URL into its source type."""
    u = url.lower()
    if re.search(r"youtube\.com/watch|youtu\.be/|youtube\.com/embed", u):
        return "youtube"
    if re.search(r"(?:^|\.)([\w]{2,3}\.)?wikipedia\.org/wiki/", u):
        return "wikipedia"
    if re.search(r"arxiv\.org/(abs|pdf)/", u):
        return "arxiv"
    if re.search(r"github\.com/[\w.\-]+/[\w.\-]+", u):
        return "github"
    return "website"


def fetch_url(url: str) -> FetchResult:
    """
    Master dispatcher — fetches content from any supported URL type.
    Raises ValueError on failure.
    """
    url_type = classify_url(url)
    logger.info("Fetching [%s]: %s", url_type, url)

    fetchers = {
        "youtube": _fetch_youtube,
        "wikipedia": _fetch_wikipedia,
        "arxiv": _fetch_arxiv,
        "github": _fetch_github,
        "website": _fetch_website,
    }

    try:
        result = fetchers[url_type](url)
        if not result.content or len(result.content.strip()) < 50:
            raise ValueError(f"Content too short ({len(result.content.strip())} chars)")
        logger.info(
            "Fetched [%s] '%s': %d words",
            result.url_type, result.title, result.word_count,
        )
        return result
    except ValueError:
        raise
    except Exception as e:
        logger.error("Fetch failed [%s] %s: %s", url_type, url, e)
        raise ValueError(f"Failed to fetch {url_type} content: {str(e)[:200]}")


# ═══════════════════════════════════════════════════════════
#  YOUTUBE
# ═══════════════════════════════════════════════════════════

def _extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"youtube\.com/watch\?v=([^&#]+)",
        r"youtu\.be/([^?&#]+)",
        r"youtube\.com/embed/([^?&#]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _fetch_youtube(url: str) -> FetchResult:
    video_id = _extract_video_id(url)
    if not video_id:
        raise ValueError("Cannot extract YouTube video ID from URL")

    # Step 1: Get video title via oEmbed (always works for public videos)
    title = f"YouTube Video ({video_id})"
    try:
        oembed = _session.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=TIMEOUT,
        )
        if oembed.ok:
            data = oembed.json()
            title = data.get("title", title)
            author = data.get("author_name", "")
        else:
            author = ""
    except Exception:
        author = ""

    # Step 2: Get transcript via youtube-transcript-api
    transcript_text = ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        # transcript is a list of snippet dicts with 'text', 'start', 'duration'
        parts = []
        for snippet in transcript:
            text = snippet.text if hasattr(snippet, 'text') else snippet.get('text', '')
            parts.append(text)
        transcript_text = " ".join(parts)
    except Exception as e:
        logger.warning("YouTube transcript fetch failed for %s: %s", video_id, e)

    # Step 3: Get description by scraping the page
    description = ""
    try:
        resp = _session.get(url, timeout=TIMEOUT)
        if resp.ok:
            html = resp.text
            # Extract description from meta or embedded JSON
            desc_match = re.search(r'"shortDescription":"((?:[^"\\]|\\.)*)"', html)
            if desc_match:
                import json as _json
                description = _json.loads('"' + desc_match.group(1) + '"')
    except Exception as e:
        logger.debug("YouTube description scrape failed: %s", e)

    # Build content
    parts = [f"# {title}"]
    if author:
        parts.append(f"Channel: {author}")
    parts.append(f"URL: {url}")

    if transcript_text:
        parts.append("\n## Transcript\n")
        parts.append(transcript_text)

    if description and len(description) > 30:
        parts.append("\n## Description\n")
        parts.append(description)

    if not transcript_text and not description:
        raise ValueError(
            "Could not retrieve transcript or description for this video. "
            "The video may be private, age-restricted, or have no captions."
        )

    content = "\n".join(parts)
    return FetchResult(
        title=title,
        content=content,
        word_count=len(content.split()),
        url=url,
        url_type="youtube",
    )


# ═══════════════════════════════════════════════════════════
#  WIKIPEDIA
# ═══════════════════════════════════════════════════════════

def _fetch_wikipedia(url: str) -> FetchResult:
    # Extract article title and language from URL
    match = re.search(r"([\w\-]+)\.wikipedia\.org/wiki/([^#?]+)", url)
    if not match:
        raise ValueError("Invalid Wikipedia URL")

    lang = match.group(1)
    article = match.group(2)
    title_decoded = requests.utils.unquote(article).replace("_", " ")

    # Use the MediaWiki API for full plaintext extract (no CORS issues server-side)
    api_url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=query&titles={article}"
        f"&prop=extracts&explaintext=1&format=json"
    )
    resp = _session.get(api_url, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = page.get("extract", "")
    page_title = page.get("title", title_decoded)

    if not extract:
        raise ValueError(f"Wikipedia article not found or empty: {title_decoded}")

    content = f"# {page_title}\n\nSource: {url}\n\n{extract}"
    return FetchResult(
        title=page_title,
        content=content,
        word_count=len(content.split()),
        url=url,
        url_type="wikipedia",
    )


# ═══════════════════════════════════════════════════════════
#  ARXIV
# ═══════════════════════════════════════════════════════════

def _fetch_arxiv(url: str) -> FetchResult:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
    if not match:
        raise ValueError("Invalid arXiv URL")

    paper_id = match.group(1)
    atom_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
    resp = _session.get(atom_url, timeout=TIMEOUT)
    resp.raise_for_status()

    # Parse Atom XML
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"arXiv paper not found: {paper_id}")

    title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
    summary = (entry.findtext("atom:summary", "", ns) or "").strip()
    authors = [
        a.findtext("atom:name", "", ns)
        for a in entry.findall("atom:author", ns)
    ]
    published = entry.findtext("atom:published", "", ns)
    categories = [
        c.get("term", "")
        for c in entry.findall("atom:category", ns)
    ]

    parts = [
        f"# {title}",
        f"\nAuthors: {', '.join(authors)}",
        f"Published: {published}",
    ]
    if categories:
        parts.append(f"Categories: {', '.join(categories)}")
    parts.append(f"\n## Abstract\n\n{summary}")
    parts.append(f"\nSource: https://arxiv.org/abs/{paper_id}")

    content = "\n".join(parts)
    return FetchResult(
        title=title or f"arXiv:{paper_id}",
        content=content,
        word_count=len(content.split()),
        url=url,
        url_type="arxiv",
    )


# ═══════════════════════════════════════════════════════════
#  GITHUB
# ═══════════════════════════════════════════════════════════

def _fetch_github(url: str) -> FetchResult:
    match = re.search(r"github\.com/([\w.\-]+)/([\w.\-]+)", url)
    if not match:
        raise ValueError("Invalid GitHub URL")

    owner, repo = match.group(1), match.group(2)
    headers = {"Accept": "application/vnd.github.v3+json"}

    # Repo metadata
    api_resp = _session.get(
        f"https://api.github.com/repos/{owner}/{repo}",
        headers=headers, timeout=TIMEOUT,
    )
    api_resp.raise_for_status()
    repo_data = api_resp.json()

    # README
    readme = ""
    try:
        readme_resp = _session.get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers=headers, timeout=TIMEOUT,
        )
        if readme_resp.ok:
            import base64
            readme_data = readme_resp.json()
            readme = base64.b64decode(readme_data.get("content", "")).decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("GitHub README fetch failed: %s", e)

    parts = [
        f"# {repo_data.get('full_name', f'{owner}/{repo}')}",
    ]
    desc = repo_data.get("description")
    if desc:
        parts.append(f"\n{desc}")
    parts.extend([
        f"\nLanguage: {repo_data.get('language', 'N/A')}",
        f"Stars: {repo_data.get('stargazers_count', 0)} | Forks: {repo_data.get('forks_count', 0)}",
        f"License: {(repo_data.get('license') or {}).get('name', 'N/A')}",
    ])
    topics = repo_data.get("topics", [])
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")
    if readme:
        parts.append(f"\n## README\n\n{readme}")

    content = "\n".join(parts)
    return FetchResult(
        title=f"{repo_data.get('full_name', f'{owner}/{repo}')} — {desc or 'GitHub Repository'}",
        content=content,
        word_count=len(content.split()),
        url=url,
        url_type="github",
    )


# ═══════════════════════════════════════════════════════════
#  GENERAL WEBSITE
# ═══════════════════════════════════════════════════════════

def _fetch_website(url: str) -> FetchResult:
    resp = _session.get(url, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove noise elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                              "aside", "iframe", "noscript", "form"]):
        tag.decompose()
    for el in soup.find_all(attrs={"role": ["banner", "navigation", "complementary"]}):
        el.decompose()
    for el in soup.find_all(class_=re.compile(r"(sidebar|menu|nav|footer|header|ad|cookie|popup|modal)", re.I)):
        el.decompose()

    # Find main content area
    main = (
        soup.find("article") or
        soup.find("main") or
        soup.find(attrs={"role": "main"}) or
        soup.find("div", class_=re.compile(r"(content|article|post|entry)", re.I)) or
        soup.body or
        soup
    )

    # Extract title
    title = None
    for getter in [
        lambda: soup.title.string.strip() if soup.title and soup.title.string else None,
        lambda: soup.find("meta", property="og:title")["content"].strip() if soup.find("meta", property="og:title") else None,
        lambda: main.find("h1").get_text(strip=True) if main.find("h1") else None,
    ]:
        try:
            title = getter()
            if title:
                break
        except Exception:
            pass
    if not title:
        from urllib.parse import urlparse
        title = urlparse(url).hostname or url

    # Extract structured text
    lines: list[str] = []
    for el in main.descendants:
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            heading_text = el.get_text(strip=True)
            if heading_text:
                lines.append(f"\n{'#' * level} {heading_text}\n")
        elif el.name == "p":
            text = el.get_text(separator=" ", strip=True)
            if text and len(text) > 10:
                lines.append(text + "\n")
        elif el.name == "li":
            text = el.get_text(separator=" ", strip=True)
            if text:
                lines.append(f"• {text}")
        elif el.name == "blockquote":
            text = el.get_text(separator=" ", strip=True)
            if text:
                lines.append(f"> {text}\n")
        elif el.name == "pre":
            text = el.get_text()
            if text:
                lines.append(f"```\n{text}\n```\n")

    content = "\n".join(lines)
    # Deduplicate consecutive identical lines
    deduped = []
    for line in content.split("\n"):
        if not deduped or line.strip() != deduped[-1].strip():
            deduped.append(line)
    content = "\n".join(deduped)

    # Clean up whitespace
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    if len(content) < 50:
        raise ValueError(
            "Page content too short or empty. "
            "The site may require JavaScript to render content."
        )

    content = f"# {title}\n\nSource: {url}\n\n{content}"
    return FetchResult(
        title=title,
        content=content,
        word_count=len(content.split()),
        url=url,
        url_type="website",
    )
