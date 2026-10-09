# Cristian Lungu's machine learning notebook

Source for [www.clungu.com](https://www.clungu.com): a static Jekyll 4 site, published with GitHub Pages Actions. Markdown posts, local layouts, one stylesheet, and small vanilla JavaScript enhancements. No remote theme, frontend framework, Node build, database, or paid hosting is required.

## Preview locally

Use Ruby 3.3 (see `.ruby-version`) and Bundler. macOS's system Ruby is too old; with Homebrew, add `$(brew --prefix ruby@3.3)/bin` to your `PATH`.

```sh
bundle install
./scripts/serve.sh
```

Open <http://127.0.0.1:4000/>. The preview reloads on changes and includes drafts and future-dated posts. Pass Jekyll options after the script name, for example `./scripts/serve.sh --port 4001`. Restart after changing `_config.yml` or dependencies.

```sh
# The same production build used by GitHub Pages:
JEKYLL_ENV=production bundle exec jekyll build
bundle exec ruby scripts/check-site.rb
```

The output in `_site/` is ignored by Git. `.ruby-version` and `Gemfile.lock` keep local and Actions dependencies aligned. After changing `Gemfile`, run `bundle install` and commit the lockfile. Use `bundle update` only when intentionally updating dependencies.

## Write a post

Create `_posts/YYYY-MM-DD-your-title.md`:

```yaml
---
title: "A better way to evaluate a model"
description: "A concise, useful summary for article listings and search previews."
categories: [tutorial]
tags: [python, evaluation]
mathjax: true
code_collapse: false
comments: true
---
```

Write ordinary Markdown below the front matter. `description` is recommended: otherwise the first paragraph becomes the listing excerpt. Do not put notebook timestamps or tag lists in that first paragraph. Publication dates are shown honestly, including on the homepage.

Maths and comments default to enabled for posts. Set `mathjax: false` to avoid loading MathJax when it isn't needed, or `comments: false` to hide the discussion section. Comments connect to GitHub/Utterances only when the reader clicks **Load comments**; keep the Utterances GitHub app installed on the configured repository.

Existing filenames and the `/:categories/:title/` permalink format are preserved. **Changing a published filename or category can change its URL.** Files without YAML front matter are not rendered as posts. Future posts appear in preview but are not published until their date arrives and a new build runs.

### Code, highlighting, and folding

Use fenced blocks with a language name. [Rouge](https://rouge.jneen.net/) highlights Python, JavaScript, TypeScript, SQL, Bash, Ruby, and many other languages at build time:

````markdown
```python
def squared_error(actual, predicted):
    return (actual - predicted) ** 2
```
````

Every rendered code block gets a language label, **Copy**, and **Hide code / Show code** controls. Set `code_collapse: true` to start a post's blocks hidden. Blocks inside native `<details>` keep their author's disclosure behavior instead. Without JavaScript, highlighted code remains visible. Long lines scroll within their block, not across the page.

For an optional explanation or example that also collapses without JavaScript, use native HTML details with Markdown enabled:

````markdown
<details markdown="1">
<summary>Show the complete implementation</summary>

```python
print("Readable, even without JavaScript.")
```

</details>
````

When a code example contains Liquid syntax such as double curly braces, wrap the example in Liquid's `raw` / `endraw` tags so Jekyll doesn't interpret it.

### LaTeX maths

Use `$...$` inline and `$$...$$` on separate lines for display equations:

```markdown
The prediction is $\hat{y} = w^\top x + b$.

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$
```

MathJax 3 is version-pinned and loaded only on pages with `mathjax: true`. Legacy Kramdown math output is supported. Code blocks are excluded from maths processing. If a post contains literal dollar signs rather than equations, use `mathjax: false` or put the literal in inline code.

### Notebooks and images

Keep notebook sources in `_notebooks/`; they are not executed or published by the build. With Jupyter installed separately:

```sh
jupyter nbconvert --to markdown "_notebooks/my-experiment.ipynb" \
  --output "2026-10-09-my-experiment" --output-dir _posts
```

Use the actual publication date, add front matter, move generated image directories into `assets/images/`, and update their links. Use Jekyll's `relative_url` filter for new images and internal links so they also work under a project-site base path:

```liquid
![A model comparison]({{ '/assets/images/my-experiment/comparison.png' | relative_url }})
```

Legacy post bodies include some hard-coded root-relative links; those assume the existing root-domain deployment.

## Customize the site

| Area | Source |
| --- | --- |
| Name, biography, contact links, domain, analytics | `_config.yml` |
| Main navigation | `_data/navigation.yml` |
| Three featured articles and their editorial summaries | `_data/featured.yml` |
| Homepage and consulting-page copy | `_layouts/home.html`, `_layouts/consulting.html` |
| About page and privacy information | `_pages/about.md`, `_pages/privacy.md` |
| Page structure and reusable components | `_layouts/`, `_includes/` |
| Responsive design and syntax palette | `assets/css/main.css` |
| Search, code controls, navigation, comments, analytics consent | `assets/js/site.js` |
| LaTeX configuration | `assets/js/math.js` |
| Social sharing artwork | `assets/images/social-card.svg` and its 1200 x 630 PNG export |

The About page uses the original `assets/images/profile-3.jpeg` portrait at its natural aspect ratio. Smaller author photos use `profile-3-avatar.jpeg`, a face-focused 336 x 336 crop, to keep circular avatars sharp without downloading the full portrait. Both paths are configured under `author` in `_config.yml`; preserve natural colour rather than applying a desaturation filter.

Featured `path` values must match published post source paths exactly. Search works locally over rendered titles, descriptions, categories, and tags; all posts remain accessible without JavaScript. Archives, old category/tag anchors, RSS (`/feed.xml`), a sitemap (`/sitemap.xml`), metadata, and a custom 404 page are generated statically.

The design uses system fonts and a local SVG illustration, with no font service or UI library requests. The one legacy article that needs jQuery opts in with `legacy_jquery: true`; new posts do not load it.

## Google Analytics 4

Universal Analytics IDs (`UA-...`) no longer work. Create a GA4 web data stream for your domain, then copy its **Measurement ID** (not its numeric property ID) into `_config.yml`:

```yaml
analytics:
  measurement_id: "G-YOURMEASUREMENTID"
```

Replace that illustrative value with your real ID. The repository deliberately ships with an empty value rather than tracking to an invented property.

Analytics is active only when **all three** conditions hold: a production build, a valid `G-...` ID, and the reader explicitly selecting **Allow analytics**. Do Not Track and Global Privacy Control are respected. Advertising features are disabled. The footer's **Analytics preferences** control allows withdrawal; a normal local preview never loads Google Analytics.

After deployment, opt in using a browser without a tracking blocker and confirm page views in your GA4 Realtime report. If the banner doesn't appear, check the measurement ID, build environment, and stored browser preference. Review `_pages/privacy.md` for your own business/legal requirements before enabling analytics.

## Publish on GitHub Pages

Push to `master`, or run **Publish blog** manually in the Actions tab. `.github/workflows/pages.yml` builds and checks the static site, uploads `_site/`, then deploys it through GitHub Pages. Source Markdown and old URLs stay in the repository.

In **Settings > Pages > Build and deployment**, set **Source** to **GitHub Actions**. Keep the existing `www.clungu.com` custom domain and DNS settings; `CNAME` remains in place. No hosting migration is needed. If Pages is still configured to deploy from a branch, the workflow cannot publish.

The workflow uses GitHub's supplied base path. To check a project-site build locally:

```sh
JEKYLL_ENV=production bundle exec jekyll build --baseurl /preview
bundle exec ruby scripts/check-site.rb _site /preview
```

Rebuild with an empty base path before serving at the domain root. Publishing a redesign does not automatically change your repository's Pages setting, create a Google Analytics property, or configure DNS.
