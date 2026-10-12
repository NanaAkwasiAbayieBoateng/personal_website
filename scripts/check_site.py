import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
FRONT_MATTER_FIELD = re.compile(r"^([a-z][\w-]*):[ \t]*(.*?)[ \t]*$", re.IGNORECASE)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(?:<([^>]+)>|([^\s]+))")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(?<!`)`+[^`\n]*`+")
EMBEDDED_CODE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>")


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


class ElementIdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)


class BlogPostLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.post_names = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        parsed = urlsplit(href)
        if parsed.path == "article.html":
            self.post_names.extend(parse_qs(parsed.query).get("post", []))


def main():
    html_files = sorted(ROOT.rglob("*.html"))
    failures = []

    for html_file in html_files:
        parser = LocalReferenceParser(html_file)
        parser.feed(html_file.read_text(encoding="utf-8"))
        for line, reference in parser.missing_references:
            relative_file = html_file.relative_to(ROOT)
            failures.append(f"{relative_file}:{line}: missing local reference {reference}")

    markdown_files = sorted((ROOT / "blog").glob("*.md"))
    blog_index = ROOT / "blog" / "index.html"
    index_parser = BlogPostLinkParser()
    index_parser.feed(blog_index.read_text(encoding="utf-8"))
    markdown_names = {markdown_file.name for markdown_file in markdown_files}
    linked_names = set()
    for post_name in index_parser.post_names:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}-[\w .-]+\.md", post_name):
            failures.append(f"blog/index.html: invalid Markdown post link {post_name}")
            continue
        if post_name in linked_names:
            failures.append(f"blog/index.html: duplicate Markdown post link {post_name}")
        linked_names.add(post_name)
        if post_name not in markdown_names:
            failures.append(f"blog/index.html: missing Markdown article {post_name}")
    for post_name in sorted(markdown_names - linked_names):
        failures.append(f"blog/index.html: Markdown article is not linked {post_name}")

    for markdown_file in markdown_files:
        source = markdown_file.read_text(encoding="utf-8")
        lines = source.splitlines()
        relative_file = markdown_file.relative_to(ROOT)

        if not lines or lines[0].strip() != "---":
            failures.append(f"{relative_file}:1: missing front matter opening delimiter")
            continue

        try:
            closing_delimiter = next(
                index for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            )
        except StopIteration:
            failures.append(f"{relative_file}:1: missing front matter closing delimiter")
            continue

        values = {}
        for line_number, line in enumerate(lines[1:closing_delimiter], start=2):
            field = FRONT_MATTER_FIELD.match(line)
            if not field:
                continue
            name, value = field.groups()
            name = name.lower()
            if name in values:
                failures.append(f"{relative_file}:{line_number}: duplicate front matter field {name}")
            values[name] = value.strip().strip("\"'")

        for required_field in ("title", "subtitle"):
            if not values.get(required_field):
                failures.append(f"{relative_file}: missing or empty front matter field {required_field}")

        date_match = re.match(r"^(\d{4}-\d{2}-\d{2})-", markdown_file.name)
        try:
            if not date_match:
                raise ValueError
            datetime.strptime(date_match.group(1), "%Y-%m-%d")
        except ValueError:
            failures.append(f"{relative_file}: filename must start with a valid YYYY-MM-DD date")

        body_lines = lines[closing_delimiter + 1:]
        body = "\n".join(body_lines)
        if not body.strip():
            failures.append(f"{relative_file}: article body is empty")

        visible_lines = []
        fence_character = None
        fence_length = 0
        for line in body_lines:
            fence = FENCE.match(line)
            if fence:
                marker = fence.group(1)
                if fence_character is None:
                    fence_character = marker[0]
                    fence_length = len(marker)
                elif marker[0] == fence_character and len(marker) >= fence_length:
                    fence_character = None
                    fence_length = 0
                continue
            if fence_character is None and not line.startswith(("    ", "\t")):
                visible_lines.append(line)

        visible_body = "\n".join(visible_lines)
        visible_body = EMBEDDED_CODE.sub(
            lambda match: "\n" * match.group().count("\n"), visible_body
        )
        visible_body = INLINE_CODE.sub(
            lambda match: "\n" * match.group().count("\n"), visible_body
        )
        markdown_references = []
        for match in MARKDOWN_LINK.finditer(visible_body):
            reference = (match.group(1) or match.group(2)).strip().strip("\"'")
            markdown_references.append(
                (visible_body.count("\n", 0, match.start()) + closing_delimiter + 2,
                 reference)
            )
        for line_number, line in enumerate(visible_body.splitlines(), start=closing_delimiter + 2):
            match = REFERENCE_LINK.match(line)
            if match:
                reference = (match.group(1) or match.group(2)).strip().strip("\"'")
                markdown_references.append((line_number, reference))

        anchor_parser = ElementIdParser()
        anchor_parser.feed(visible_body)
        for line_number, reference in markdown_references:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc:
                continue
            if not parsed.path and reference.startswith("#"):
                fragment = unquote(parsed.fragment)
                if not fragment:
                    failures.append(f"{relative_file}:{line_number}: empty in-article link target")
                elif fragment not in anchor_parser.ids:
                    failures.append(
                        f"{relative_file}:{line_number}: missing in-article anchor #{fragment}"
                    )
                continue
            if not parsed.path:
                continue
            path = unquote(parsed.path)
            target = (ROOT / path.lstrip("/")) if path.startswith("/") else (
                markdown_file.parent / path
            )
            target = target.resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                failures.append(
                    f"{relative_file}:{line_number}: missing local reference {reference}"
                )

        html_parser = LocalReferenceParser(markdown_file)
        html_parser.feed(visible_body)
        for line_number, reference in html_parser.missing_references:
            failures.append(
                f"{relative_file}:{line_number + closing_delimiter + 1}: "
                f"missing local reference {reference}"
            )

    if failures:
        print("\n".join(failures))
        return 1

    print(
        f"Checked local references in {len(html_files)} HTML files and "
        f"front matter and links in {len(markdown_files)} Markdown articles."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
