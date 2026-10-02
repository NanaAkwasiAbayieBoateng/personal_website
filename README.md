# Nana Boateng Portfolio

This is a lightweight static website. Keep the home page at the site root so static hosts can serve it as the default document.

## Project structure

```text
.
├── index.html                 # Home page and site entry point
├── pages/                     # About, expertise, and contact pages
├── blog/                      # Blog index and individual articles
└── assets/
    ├── css/styles.css         # Shared site styles
    └── js/script.js           # Shared interactions
```

Pages use relative links, so the site can be served from a root domain or a subdirectory without requiring a build step.

## Preview locally

Open `index.html` in a browser, or serve this directory with any static file server. For example, when Python is available:

```powershellco
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## GitHub Actions

The `Validate site` workflow runs on pushes and pull requests. It checks that
local file links and asset references in the HTML pages point to files in the
repository.
