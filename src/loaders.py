import requests  # type: ignore
from typing import Any, Iterator

import bs4

from .document import Document

DEFAULT_HEADER_TEMPLATE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.35/36 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class WebLoader:
    def __init__(
        self,
        urls: list[str],
        requests_kwargs: dict[str, Any] = {},
        bs_kwargs: dict[str, Any] = {},
    ) -> None:
        self.urls = urls
        self.bs_kwargs = bs_kwargs
        session = requests.Session()
        session.headers = dict(DEFAULT_HEADER_TEMPLATE)
        session.verify = True

        self.session = session

    def load(
        self,
    ) -> list[Document]:
        return list(self.aload())

    def aload(
        self,
    ) -> Iterator[Document]:
        for url in self.urls:
            soup = self._scrape(url)
            text = soup.get_text()
            metadata = self._build_metadata(soup, url)
            yield Document(text=text, metadata=metadata)

    def _scrape(self, url: str):
        html_doc = self.session.get(url, **{})
        return bs4.BeautifulSoup(html_doc.text, "html.parser", **(self.bs_kwargs or {}))

    def _build_metadata(self, soup: Any, url: str) -> dict[str, Any]:
        metadata = {"source": url}
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get(
                "content", "No description found."
            )
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")
        return metadata


class TextLoader:
    def __init__(
        self,
        paths: list[str],
    ) -> None:
        self.paths = paths

    def load(
        self,
    ) -> list[Document]:
        return list(self.aload())

    def aload(
        self,
    ) -> Iterator[Document]:
        for path in self.paths:
            text = self._read_file(path)
            metadata = self._build_metadata(text, path)
            yield Document(text=text, metadata=metadata)

    def _build_metadata(self, text: Any, path: str) -> dict[str, Any]:
        metadata = {"source": path.split("/")[-1].split(".")[0]}
        return metadata

    def _read_file(self, path: str) -> str:
        with open(path) as fd:
            text = fd.read()
        return text
