(() => {
const articleTitle = document.querySelector('#article-title');
const articleMeta = document.querySelector('#article-meta');
const articleStatus = document.querySelector('#article-status');
const renderedArticle = document.querySelector('#article-content');
const postName = new URLSearchParams(window.location.search).get('post');
const validPostName = /^\d{4}-\d{2}-\d{2}-[\w .-]+\.md$/;

const humanizePostName = (name) => name
  .replace(/^\d{4}-\d{2}-\d{2}-/, '')
  .replace(/\.md$/i, '')
  .replace(/[_-]+/g, ' ')
  .replace(/\s+/g, ' ')
  .trim();

const parseFrontMatter = (markdown) => {
  const match = markdown.match(/^---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)/);
  if (!match) return { content: markdown, values: {} };

  const values = {};
  match[1].split(/\r?\n/).forEach((line) => {
    const field = line.match(/^([a-z][\w-]*):\s*(.*)$/i);
    if (field) values[field[1].toLowerCase()] = field[2].replace(/^["']|["']$/g, '').trim();
  });
  return { content: markdown.slice(match[0].length), values };
};

const findFirstHeading = (markdown) => {
  let inCodeFence = false;
  for (const line of markdown.split(/\r?\n/)) {
    if (/^\s*(```|~~~)/.test(line)) {
      inCodeFence = !inCodeFence;
      continue;
    }
    const heading = !inCodeFence && line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/);
    if (heading) return heading[1].replace(/[*_`]/g, '').trim();
  }
  return '';
};

const compactText = (markdown) => markdown
  .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
  .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  .replace(/[`*_>#~]/g, '')
  .replace(/\s+/g, ' ')
  .trim();

const firstParagraph = (markdown) => markdown
  .split(/\r?\n\s*\r?\n/)
  .map((paragraph) => paragraph.trim())
  .find((paragraph) => paragraph && !/^(?:#{1,6}\s|```|~~~|[-*_]{3,}|<)/.test(paragraph)) || '';

const loadArticle = async () => {
  if (!postName || !validPostName.test(postName)) {
    throw new Error('This article link is invalid. Return to the blog to choose an article.');
  }
  if (!window.marked || !window.DOMPurify) {
    throw new Error('The article renderer could not load. Check your connection and refresh the page.');
  }

  const response = await fetch(encodeURIComponent(postName));
  if (!response.ok) throw new Error(`The article could not be loaded (HTTP ${response.status}).`);

  const rawMarkdown = await response.text();
  const { content, values } = parseFrontMatter(rawMarkdown);
  const firstHeading = findFirstHeading(content);
  const fallbackHeading = /^(?:introduction|contents|table of contents)$/i.test(firstHeading)
    ? ''
    : firstHeading;
  const title = values.title || fallbackHeading || humanizePostName(postName);
  const markdown = firstHeading && firstHeading.toLowerCase() === title.toLowerCase()
    ? content.replace(/^\s{0,3}#{1,6}\s+.+?(?:\r?\n|$)/, '')
    : content;
  const words = content.trim().split(/\s+/).filter(Boolean).length;
  const readingTime = Math.max(1, Math.ceil(words / 220));
  const dateMatch = postName.match(/^(\d{4})-(\d{2})-(\d{2})-/);
  const date = dateMatch
    ? new Intl.DateTimeFormat('en-US', { month: 'short', day: '2-digit', year: 'numeric' })
      .format(new Date(`${dateMatch[1]}-${dateMatch[2]}-${dateMatch[3]}T00:00:00`))
    : '';
  const summary = values.subtitle || compactText(firstParagraph(content)).slice(0, 240);

  articleTitle.textContent = title;
  document.title = `${title} | Nana Boateng`;
  articleMeta.textContent = [date, `${readingTime} min read`].filter(Boolean).join(' · ');
  if (summary) articleMeta.insertAdjacentText('beforeend', ` — ${summary}`);
  renderedArticle.innerHTML = window.DOMPurify.sanitize(window.marked.parse(markdown));
  articleStatus.hidden = true;
};

loadArticle().catch((error) => {
  articleTitle.textContent = 'Article unavailable';
  articleStatus.textContent = error.message;
  articleStatus.setAttribute('role', 'alert');
});
})();
