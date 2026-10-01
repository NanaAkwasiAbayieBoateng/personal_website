const menuToggle = document.querySelector('.menu-toggle');
const mainNav = document.querySelector('.main-nav');
const yearNode = document.getElementById('year');
const siteHeader = document.querySelector('.site-header');

if (yearNode) yearNode.textContent = new Date().getFullYear();

if (menuToggle && mainNav) {
  const setMenuOpen = (isOpen) => {
    mainNav.classList.toggle('open', isOpen);
    menuToggle.setAttribute('aria-expanded', String(isOpen));
    menuToggle.setAttribute('aria-label', isOpen ? 'Close menu' : 'Open menu');
  };

  menuToggle.addEventListener('click', () => {
    setMenuOpen(menuToggle.getAttribute('aria-expanded') !== 'true');
  });

  mainNav.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => setMenuOpen(false));
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && menuToggle.getAttribute('aria-expanded') === 'true') {
      setMenuOpen(false);
      menuToggle.focus();
    }
  });

  document.addEventListener('click', (event) => {
    if (
      menuToggle.getAttribute('aria-expanded') === 'true' &&
      !mainNav.contains(event.target) &&
      !menuToggle.contains(event.target)
    ) {
      setMenuOpen(false);
    }
  });

  window.addEventListener('resize', () => {
    if (window.innerWidth > 960) setMenuOpen(false);
  });
}

if (mainNav) {
  const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
  mainNav.querySelectorAll('a').forEach((link) => {
    const linkPath = new URL(link.href, window.location.href).pathname.replace(/\/+$/, '') || '/';
    const isCurrentPage = linkPath === currentPath;
    const isBlogArticle = currentPath.includes('/blog/') && linkPath.endsWith('/blog/index.html');
    if (isCurrentPage || isBlogArticle) {
      link.setAttribute('aria-current', 'page');
    }
  });
}

if (siteHeader) {
  const updateHeaderState = () => {
    siteHeader.classList.toggle('is-scrolled', window.scrollY > 8);
  };
  window.addEventListener('scroll', updateHeaderState, { passive: true });
  updateHeaderState();
}

const revealItems = document.querySelectorAll(
  '.hero-copy, .hero-visual, .stat-grid > div, .section-heading, .feature-stack article, .program-card, .impact-card, .post-card, .timeline article, .contact-card'
);
if ('IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  revealItems.forEach((item) => {
    item.classList.add('reveal');
    revealObserver.observe(item);
  });
}

const backToTop = document.createElement('button');
backToTop.className = 'back-to-top';
backToTop.type = 'button';
backToTop.setAttribute('aria-label', 'Back to top');
backToTop.textContent = '↑';
document.body.append(backToTop);

const updateBackToTop = () => {
  backToTop.classList.toggle('is-visible', window.scrollY > 500);
};
window.addEventListener('scroll', updateBackToTop, { passive: true });
updateBackToTop();
backToTop.addEventListener('click', () => {
  const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
  window.scrollTo({ top: 0, behavior });
});

const blogCards = [...document.querySelectorAll('[data-blog-category]')];
const blogSearch = document.querySelector('#blog-search');
const blogFilters = document.querySelectorAll('[data-blog-filter]');
const blogResults = document.querySelector('.blog-results');
let activeBlogFilter = 'all';

const updateBlogList = () => {
  if (!blogCards.length) return;
  const searchTerm = blogSearch ? blogSearch.value.trim().toLowerCase() : '';
  let visibleCount = 0;

  blogCards.forEach((card) => {
    const matchesCategory = activeBlogFilter === 'all' || card.dataset.blogCategory === activeBlogFilter;
    const matchesSearch = !searchTerm || card.textContent.toLowerCase().includes(searchTerm);
    card.hidden = !(matchesCategory && matchesSearch);
    if (!card.hidden) visibleCount += 1;
  });

  if (blogResults) {
    blogResults.textContent = visibleCount
      ? `Showing ${visibleCount} ${visibleCount === 1 ? 'article' : 'articles'}`
      : 'No articles match. Try another search or topic.';
  }
};

blogFilters.forEach((button) => {
  button.addEventListener('click', () => {
    activeBlogFilter = button.dataset.blogFilter;
    blogFilters.forEach((filter) => filter.setAttribute('aria-pressed', String(filter === button)));
    updateBlogList();
  });
});

if (blogSearch) blogSearch.addEventListener('input', updateBlogList);

const articleContent = document.querySelector('.article-content');
if (articleContent) {
  const progress = document.createElement('div');
  progress.className = 'reading-progress';
  progress.setAttribute('role', 'progressbar');
  progress.setAttribute('aria-label', 'Article reading progress');
  progress.setAttribute('aria-valuemin', '0');
  progress.setAttribute('aria-valuemax', '100');
  progress.innerHTML = '<span></span>';
  document.body.append(progress);
  const progressBar = progress.querySelector('span');

  const updateReadingProgress = () => {
    const articleTop = articleContent.getBoundingClientRect().top + window.scrollY;
    const articleHeight = articleContent.offsetHeight;
    const available = articleHeight - window.innerHeight;
    const readingProgress = available > 0 ? (window.scrollY - articleTop) / available : 1;
    const percent = Math.min(100, Math.max(0, readingProgress * 100));
    progressBar.style.width = `${percent}%`;
    progress.setAttribute('aria-valuenow', String(Math.round(percent)));
  };

  window.addEventListener('scroll', updateReadingProgress, { passive: true });
  window.addEventListener('resize', updateReadingProgress);
  updateReadingProgress();

  articleContent.querySelectorAll('.code-example').forEach((example) => {
    const code = example.querySelector('pre code');
    if (!code) return;

    const copyButton = document.createElement('button');
    copyButton.className = 'copy-code';
    copyButton.type = 'button';
    copyButton.textContent = 'Copy code';
    const status = document.createElement('p');
    status.className = 'copy-status';
    status.setAttribute('aria-live', 'polite');
    example.append(copyButton, status);

    copyButton.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        copyButton.textContent = 'Copied!';
        status.textContent = 'Code copied to clipboard.';
      } catch {
        copyButton.textContent = 'Copy unavailable';
        status.textContent = 'Clipboard access is unavailable. Select and copy the code manually.';
      }
      window.setTimeout(() => {
        copyButton.textContent = 'Copy code';
      }, 1800);
    });
  });
}

const contactForm = document.querySelector('.contact-form');
if (contactForm) {
  contactForm.addEventListener('submit', (event) => {
    event.preventDefault();
    if (!contactForm.reportValidity()) return;

    const formData = new FormData(contactForm);
    const subject = encodeURIComponent(`Website inquiry from ${formData.get('name')}`);
    const body = encodeURIComponent(
      `Name: ${formData.get('name')}\nEmail: ${formData.get('email')}\nCompany: ${formData.get('company') || 'Not provided'}\n\n${formData.get('message')}`
    );
    const status = contactForm.querySelector('.form-status');
    if (status) status.textContent = 'Opening a draft in your email app. Add Nana’s current email address before sending.';
    window.location.href = `mailto:?subject=${subject}&body=${body}`;
  });
}
