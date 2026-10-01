from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent


class LocalReferenceParser(HTMLParser):
    def __init__(self, html_file):
        super().__init__()
        self.html_file = html_file
        self.missing_references = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name not in {"href", "src"} or not value:
                continue

            parsed = urlsplit(value.strip())
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue

            path = unquote(parsed.path)
            target = (ROOT / path.lstrip("/")) if path.startswith("/") else (
                self.html_file.parent / path
            )
            target = target.resolve()

            if not target.is_relative_to(ROOT) or not target.exists():
                self.missing_references.append((self.getpos()[0], value))


def main():
    html_files = sorted(ROOT.rglob("*.html"))
    failures = []

    for html_file in html_files:
        parser = LocalReferenceParser(html_file)
        parser.feed(html_file.read_text(encoding="utf-8"))
        for line, reference in parser.missing_references:
            relative_file = html_file.relative_to(ROOT)
            failures.append(f"{relative_file}:{line}: missing local reference {reference}")

    if failures:
        print("\n".join(failures))
        return 1

    print(f"Checked local references in {len(html_files)} HTML files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
