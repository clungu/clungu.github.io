# Machine Learning Blog

Source for [www.clungu.com](https://www.clungu.com), a Jekyll blog hosted on the free GitHub Pages service. Dated Markdown posts live in `_posts/`; source notebooks live in `_notebooks/`. The site uses the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme.

## Preview locally

Install Ruby 3.3 (a version manager or Homebrew `ruby@3.3` on macOS) and Bundler. Check that `ruby --version` reports 3.3; macOS's system Ruby is too old. For Homebrew's keg-only Ruby, add `$(brew --prefix ruby@3.3)/bin` to your `PATH` before running `gem install bundler` if Bundler is not already available.

From the repository root:

```sh
bundle install
./scripts/serve.sh
```

Open <http://127.0.0.1:4000/>. The script watches for changes, reloads the browser, shows drafts and future-dated posts, and prints build traces on errors. Pass additional Jekyll options after the script name (for example, `./scripts/serve.sh --port 4001`). Restart the script when changing `_config.yml` or the Gemfile. To build without running a server, use `bundle exec jekyll build`; its output is in `_site/` and is ignored by Git.

The `.ruby-version` and `Gemfile.lock` pin the local and GitHub Actions build to the same dependencies. After changing `Gemfile`, run `bundle install` and commit the updated lockfile. To intentionally refresh dependencies, run `bundle update` and preview before committing.

## Write and publish

Create `_posts/YYYY-MM-DD-your-title.md` with YAML front matter. Existing dated Markdown filenames, permalinks, and image assets stay in place. For example:

```markdown
---
title: "A new experiment"
categories: [tutorial]
tags: [machine-learning]
mathjax: true
---

Write your post here. Set `mathjax: true` only when the post needs math notation.
```

Preview the post locally, then push to `master`. Posts dated in the future appear in the local preview but not in the production build until their date arrives. Files without YAML front matter are not rendered as Jekyll posts; add it for any new post.

For a notebook-based post, keep the `.ipynb` source in `_notebooks/` and convert a copy to Markdown with [Jupyter nbconvert](https://nbconvert.readthedocs.io/) (install Jupyter separately if needed):

```sh
jupyter nbconvert --to markdown "_notebooks/my-experiment.ipynb" \
  --output "2026-09-30-my-experiment" --output-dir _posts
```

Add front matter to `_posts/2026-09-30-my-experiment.md`. If nbconvert produces an image directory alongside the Markdown, move that directory to `assets/images/` and update the Markdown image URLs to start with `/assets/images/` so they work at the post's permalink. Commit both the Markdown and its images; notebook execution is not part of the Jekyll build. The date above is an example: use your actual publication date.

## GitHub Pages deployment

`.github/workflows/pages.yml` builds the same Jekyll site on pushes to `master` (or by manual **Run workflow**) and deploys the `_site` artifact with GitHub Pages Actions. No paid hosting or external deployment service is needed. The custom domain remains defined in `CNAME`.

**One-time migration:** In the repository's GitHub **Settings > Pages > Build and deployment**, change **Source** from **Deploy from a branch** to **GitHub Actions**. Until that setting is changed, the new workflow cannot publish the site. Keep the existing custom domain in Pages settings; verify `www.clungu.com` still points at the deployed site after the first run. Check the **Actions** tab for build/deploy errors.

Google Universal Analytics (`UA-...`) has been retired, so the obsolete tracking snippet has been removed. If analytics are needed again, configure a current GA4 measurement ID using the theme's `google-gtag` provider.
