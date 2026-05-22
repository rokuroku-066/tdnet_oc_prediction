from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .text_extractor import TextExtractor
from .validators import validate_disclosures

LOGGER = logging.getLogger(__name__)


@dataclass
class TDnetPublicClient:
    sleep_sec: float = 0.2
    timeout_sec: float = 20.0
    max_pages: int = 20
    extract_pdf: bool = True

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "tdnet-oc-prediction-research/0.1"})
        self.text_extractor = TextExtractor()

    @staticmethod
    def _daily_page_url(d: date, page: int) -> str:
        return f"https://www.release.tdnet.info/inbs/I_list_{page:03d}_{d.strftime('%Y%m%d')}.html"

    def _extract_pdf_text(self, pdf_url: str) -> str:
        try:
            response = self.session.get(pdf_url, timeout=self.timeout_sec)
            response.raise_for_status()
            return self.text_extractor.extract_pdf_bytes(response.content)
        except requests.RequestException as exc:
            LOGGER.warning("failed to fetch pdf %s: %s", pdf_url, exc)
            return ""
        except ValueError as exc:
            LOGGER.warning("failed to parse pdf content %s: %s", pdf_url, exc)
            return ""
        except TypeError as exc:
            LOGGER.warning("invalid pdf payload for %s: %s", pdf_url, exc)
            return ""

    @staticmethod
    def _make_disclosure_id(pdf_url: str, stock_code: str, disclosure_date: str, disclosure_time: str, title: str) -> str:
        parsed = urlparse(pdf_url)
        name = Path(parsed.path).name
        if name:
            stem = Path(name).stem
            if stem:
                return stem
        seed = f"{stock_code}|{disclosure_date}|{disclosure_time}|{title}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()

    @staticmethod
    def _infer_company_name(text_cells: list[str], stock_code: str) -> str:
        code_idx = -1
        for idx, cell in enumerate(text_cells):
            if re.search(rf"\b{re.escape(stock_code)}\b", cell):
                code_idx = idx
                break
        if code_idx >= 0:
            for cell in text_cells[code_idx + 1 :]:
                cleaned = cell.strip()
                if cleaned and cleaned != stock_code and not re.search(r"\b[0-2]\d:[0-5]\d\b", cleaned):
                    return cleaned
        return ""

    def _parse_page(self, html: str, page_url: str, disclosure_date: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.select("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            text_cells = [td.get_text(" ", strip=True) for td in tds]
            code_match = next((re.search(r"\b(\d{4})\b", t) for t in text_cells if re.search(r"\b\d{4}\b", t)), None)
            if not code_match:
                continue
            stock_code = code_match.group(1)
            time_match = next((re.search(r"\b([0-2]\d:[0-5]\d)\b", t) for t in text_cells if re.search(r"\b[0-2]\d:[0-5]\d\b", t)), None)
            disclosure_time = time_match.group(1) if time_match else ""

            links = tr.find_all("a", href=True)
            pdf_link = next((a for a in links if a["href"].lower().endswith(".pdf")), None)
            if pdf_link is None:
                continue

            title = pdf_link.get_text(" ", strip=True)
            if not title:
                title = max(text_cells, key=len, default="")
            href = pdf_link["href"]
            pdf_url = urljoin(page_url, href)
            company_name = self._infer_company_name(text_cells, stock_code)
            body_text = self._extract_pdf_text(pdf_url) if self.extract_pdf else ""
            rows.append(
                {
                    "disclosure_id": self._make_disclosure_id(pdf_url, stock_code, disclosure_date, disclosure_time, title),
                    "stock_code": str(stock_code),
                    "company_name": company_name,
                    "disclosure_date": disclosure_date,
                    "disclosure_time": disclosure_time,
                    "title": title or "",
                    "body_text": body_text or "",
                    "source_url": page_url,
                    "pdf_url": pdf_url,
                }
            )
        return rows

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        all_rows: list[dict] = []
        for d in pd.date_range(start=start_date, end=end_date, freq="D"):
            d_str = d.strftime("%Y-%m-%d")
            for page in range(1, self.max_pages + 1):
                page_url = self._daily_page_url(d.date(), page)
                try:
                    resp = self.session.get(page_url, timeout=self.timeout_sec)
                except requests.RequestException as exc:
                    LOGGER.warning("failed to fetch %s: %s", page_url, exc)
                    break
                if resp.status_code == 404:
                    break
                if resp.status_code >= 400:
                    LOGGER.warning("non-200 on %s: %s", page_url, resp.status_code)
                    break
                resp.encoding = resp.apparent_encoding
                rows = self._parse_page(resp.text, page_url, d_str)
                if not rows:
                    break
                all_rows.extend(rows)
                time.sleep(self.sleep_sec)
            time.sleep(self.sleep_sec)

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df
        return validate_disclosures(df)
