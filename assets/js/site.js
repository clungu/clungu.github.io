(function () {
  'use strict';

  document.documentElement.classList.add('js');

  let nextId = 0;

  function ensureId(element, prefix) {
    if (!element.id) {
      let id;
      do {
        id = prefix + '-' + (++nextId);
      } while (document.getElementById(id));
      element.id = id;
    }
    return element.id;
  }

  function initNavigation() {
    const toggle = document.querySelector('[data-nav-toggle]');
    const nav = document.getElementById('site-nav');
    if (!toggle || !nav) return;

    function setOpen(open) {
      nav.classList.toggle('is-open', open);
      toggle.setAttribute('aria-expanded', String(open));
    }

    toggle.hidden = false;
    toggle.setAttribute('aria-controls', nav.id);
    setOpen(false);
    toggle.addEventListener('click', function () {
      setOpen(toggle.getAttribute('aria-expanded') !== 'true');
    });
    nav.addEventListener('click', function (event) {
      if (event.target.closest('a[href]')) setOpen(false);
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
        setOpen(false);
        toggle.focus();
      }
    });
  }

  function initCode() {
    document.querySelectorAll('.prose pre').forEach(function (pre) {
      const code = pre.querySelector('code');
      if (!code) return;
      if (!pre.hasAttribute('tabindex')) pre.tabIndex = 0;
      if (pre.closest('.code-block')) return;

      const prose = pre.closest('.prose');
      let content = pre;
      let language = '';
      for (let node = code; node && node !== prose; node = node.parentElement) {
        const match = node.className.match(/\blanguage-([^\s]+)/);
        if (!language && match) language = match[1];
      }
      for (let node = pre.parentElement; node && node !== prose; node = node.parentElement) {
        if (node.matches('details')) break;
        if (node.matches('div.highlighter-rouge, figure.highlight')) content = node;
      }

      const block = document.createElement('div');
      block.className = 'code-block';
      content.before(block);
      block.append(content);

      const toolbar = document.createElement('div');
      toolbar.className = 'code-toolbar';
      const label = document.createElement('span');
      label.className = 'code-language';
      label.textContent = language || 'Text';
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'code-toggle';
      toggle.setAttribute('aria-controls', ensureId(content, 'code-content'));
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.className = 'code-copy';
      copy.textContent = 'Copy';
      const status = document.createElement('span');
      status.className = 'code-status';
      status.setAttribute('role', 'status');
      toolbar.append(label, toggle, copy, status);
      block.prepend(toolbar);

      function setCollapsed(collapsed) {
        content.hidden = collapsed;
        toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.textContent = collapsed ? 'Show code' : 'Hide code';
      }

      setCollapsed(prose.dataset.codeCollapse === 'true' && !pre.closest('details'));
      toggle.addEventListener('click', function () {
        setCollapsed(!content.hidden);
      });
      let resetCopy;
      copy.addEventListener('click', async function () {
        clearTimeout(resetCopy);
        copy.disabled = true;
        copy.textContent = 'Copy';
        status.textContent = '';
        const codes = content === pre ? [code] : content.querySelectorAll('pre code');
        const text = Array.from(codes, function (item) { return item.textContent; }).join('\n');
        try {
          if (!navigator.clipboard || !navigator.clipboard.writeText) {
            throw new Error('Clipboard access is unavailable.');
          }
          await navigator.clipboard.writeText(text);
          copy.textContent = 'Copied';
          status.textContent = 'Code copied to clipboard.';
          resetCopy = setTimeout(function () {
            copy.textContent = 'Copy';
            status.textContent = '';
          }, 2500);
        } catch (error) {
          status.textContent = 'Copy failed. Select the code and copy it manually.';
        } finally {
          copy.disabled = false;
        }
      });
    });
  }

  function initToc() {
    const toc = document.querySelector('[data-toc]');
    const list = toc && toc.querySelector('[data-toc-list]');
    if (!list) return;
    const headings = Array.from(document.querySelectorAll('.prose h1[id], .prose h2[id], .prose h3[id]'))
      .filter(function (heading) { return heading.id && heading.textContent.trim(); });
    if (headings.length < 2) return;

    list.replaceChildren();
    headings.forEach(function (heading) {
      const item = document.createElement('li');
      if (heading.tagName === 'H3') item.className = 'toc-subheading';
      const link = document.createElement('a');
      link.href = '#' + encodeURIComponent(heading.id);
      link.textContent = heading.textContent.trim();
      item.append(link);
      list.append(item);
    });
    toc.hidden = false;
  }

  function initSearch() {
    document.querySelectorAll('[data-search]').forEach(function (section) {
      const controls = section.querySelector('[data-search-controls]');
      const input = section.querySelector('[data-search-input]');
      const topic = section.querySelector('[data-search-topic]');
      if (!controls || !input || !topic) return;
      const items = Array.from(section.querySelectorAll('[data-search-item]'), function (item) {
        return {
          element: item,
          text: (item.dataset.searchText || '').toLowerCase(),
          topics: (item.dataset.topics || '').split(/\s+/)
        };
      });
      const count = section.querySelector('[data-search-count]');
      const empty = section.querySelector('[data-search-empty]');

      function filter(updateUrl) {
        const query = input.value.trim();
        const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
        let visible = 0;
        items.forEach(function (item) {
          const matches = tokens.every(function (token) { return item.text.includes(token); }) &&
            (!topic.value || item.topics.includes(topic.value));
          item.element.hidden = !matches;
          if (matches) visible++;
        });
        if (count) count.textContent = visible + (visible === 1 ? ' article' : ' articles');
        if (empty) empty.hidden = visible !== 0;
        if (updateUrl) {
          const url = new URL(window.location.href);
          if (query) url.searchParams.set('q', query);
          else url.searchParams.delete('q');
          if (topic.value) url.searchParams.set('topic', topic.value);
          else url.searchParams.delete('topic');
          window.history.replaceState(window.history.state, '', url);
        }
      }

      function restore() {
        const params = new URLSearchParams(window.location.search);
        input.value = params.get('q') || '';
        const requestedTopic = params.get('topic') || '';
        topic.value = Array.from(topic.options).some(function (option) {
          return option.value === requestedTopic;
        }) ? requestedTopic : '';
        filter(false);
      }

      controls.hidden = false;
      input.addEventListener('input', function () { filter(true); });
      topic.addEventListener('change', function () { filter(true); });
      controls.addEventListener('submit', function (event) {
        event.preventDefault();
        filter(true);
      });
      window.addEventListener('popstate', restore);
      restore();
    });
  }

  function initAnalytics() {
    const config = document.getElementById('site-analytics');
    if (!config) return;
    let measurementId;
    try {
      const parsed = JSON.parse(config.textContent);
      measurementId = parsed && parsed.measurementId;
    } catch (error) {
      console.warn('Analytics configuration could not be read.', error);
      return;
    }
    if (typeof measurementId !== 'string' || !/^G-[A-Z0-9]+$/.test(measurementId)) {
      console.warn('Analytics requires a valid GA4 measurement ID.');
      return;
    }

    const storageKey = 'clungu-analytics-consent';
    const banner = document.querySelector('[data-consent-banner]');
    const status = banner && banner.querySelector('[data-consent-status]');
    const accept = banner && banner.querySelector('[data-consent-accept]');
    const reject = banner && banner.querySelector('[data-consent-reject]');
    const settings = Array.from(document.querySelectorAll('[data-privacy-settings]'));
    const denied = {
      analytics_storage: 'denied',
      ad_storage: 'denied',
      ad_user_data: 'denied',
      ad_personalization: 'denied'
    };
    let choice = null;
    let initialized = false;
    let returnFocus = null;

    function privacyRequested() {
      return navigator.globalPrivacyControl === true ||
        navigator.doNotTrack === '1' || window.doNotTrack === '1' ||
        navigator.msDoNotTrack === '1';
    }

    function validChoice(value) {
      return value === 'granted' || value === 'denied' ? value : null;
    }

    try {
      choice = validChoice(window.localStorage.getItem(storageKey));
    } catch (error) {
      console.warn('Analytics consent storage is unavailable; choices apply to this page only.', error);
    }

    function clearAnalyticsCookies() {
      try {
        const names = document.cookie.split(';').map(function (cookie) {
          return cookie.trim().split('=')[0];
        }).filter(function (name) { return /^_ga(?:_|$)/.test(name); });
        const domains = [''];
        const hostname = window.location.hostname;
        if (hostname) {
          domains.push(hostname, '.' + hostname);
          if (!/^[\d.]+$/.test(hostname) && !hostname.includes(':')) {
            const labels = hostname.split('.');
            for (let i = 1; i < labels.length - 1; i++) {
              const domain = labels.slice(i).join('.');
              domains.push(domain, '.' + domain);
            }
          }
        }
        const paths = new Set(['/']);
        let path = '';
        window.location.pathname.split('/').filter(Boolean).forEach(function (part) {
          path += '/' + part;
          paths.add(path);
          paths.add(path + '/');
        });
        names.forEach(function (name) {
          domains.forEach(function (domain) {
            paths.forEach(function (cookiePath) {
              document.cookie = name + '=; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=' +
                cookiePath + (domain ? '; Domain=' + domain : '');
            });
          });
        });
      } catch (error) {
        console.warn('Existing analytics cookies could not be removed.', error);
      }
    }

    function updateStatus() {
      const blocked = privacyRequested();
      if (accept) accept.disabled = blocked;
      if (!status) return;
      if (blocked) {
        status.textContent = 'Analytics is disabled because your browser sends a Do Not Track or Global Privacy Control signal.';
      } else if (choice === 'granted') {
        status.textContent = 'Optional Google Analytics is enabled. Choose No thanks to turn it off.';
      } else {
        status.textContent = 'Optional Google Analytics is off. It will only be enabled if you accept.';
      }
    }

    function showBanner(show) {
      if (banner) banner.hidden = !show;
      settings.forEach(function (button) {
        button.setAttribute('aria-expanded', String(show && !!banner));
      });
      updateStatus();
    }

    function applyConsent() {
      const granted = choice === 'granted' && !privacyRequested();
      window['ga-disable-' + measurementId] = !granted;
      if (!granted) {
        if (initialized) window.gtag('consent', 'update', denied);
        clearAnalyticsCookies();
        updateStatus();
        return;
      }

      if (!initialized) {
        window.dataLayer = window.dataLayer || [];
        window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };
        window.gtag('consent', 'default', denied);
        window.gtag('js', new Date());
      }
      window.gtag('consent', 'update', Object.assign({}, denied, { analytics_storage: 'granted' }));
      if (!initialized) {
        initialized = true;
        window.gtag('config', measurementId, {
          allow_google_signals: false,
          allow_ad_personalization_signals: false
        });
        const script = document.createElement('script');
        script.async = true;
        script.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(measurementId);
        document.head.append(script);
      }
      updateStatus();
    }

    function choose(value) {
      choice = value;
      applyConsent();
      try {
        window.localStorage.setItem(storageKey, value);
      } catch (error) {
        console.warn('Analytics consent could not be saved; your choice still applies to this page.', error);
      }
      const moveFocus = banner && banner.contains(document.activeElement);
      showBanner(false);
      if (moveFocus && (returnFocus || settings[0])) {
        (returnFocus || settings[0]).focus({ preventScroll: true });
      }
    }

    settings.forEach(function (button) {
      button.hidden = false;
      if (banner) button.setAttribute('aria-controls', ensureId(banner, 'analytics-consent'));
      button.addEventListener('click', function () {
        returnFocus = button;
        showBanner(true);
        const target = accept && !accept.disabled ? accept : reject;
        if (target) target.focus();
      });
    });
    if (accept) accept.addEventListener('click', function () {
      if (!privacyRequested()) choose('granted');
      else updateStatus();
    });
    if (reject) reject.addEventListener('click', function () { choose('denied'); });
    window.addEventListener('storage', function (event) {
      if (event.key !== storageKey && event.key !== null) return;
      choice = validChoice(event.newValue);
      applyConsent();
      showBanner(!choice || (privacyRequested() && choice !== 'denied'));
    });
    applyConsent();
    showBanner(!choice || (privacyRequested() && choice !== 'denied'));
  }

  function initComments() {
    document.querySelectorAll('[data-comments]').forEach(function (container) {
      const button = container.querySelector('[data-load-comments]');
      const status = container.querySelector('[data-comments-status]');
      if (!button) return;
      let loading = false;
      let loaded = false;
      button.addEventListener('click', function () {
        if (loading || loaded) return;
        const repository = container.dataset.repository || '';
        if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) {
          if (status) status.textContent = 'Comments are unavailable: no valid repository is configured.';
          return;
        }
        loading = true;
        button.disabled = true;
        if (status) {
          status.hidden = false;
          status.textContent = 'Loading comments...';
        }
        const script = document.createElement('script');
        script.src = 'https://utteranc.es/client.js';
        script.setAttribute('repo', repository);
        script.setAttribute('issue-term', 'pathname');
        script.setAttribute('theme', 'github-light');
        script.crossOrigin = 'anonymous';
        script.async = true;
        script.onload = function () {
          loading = false;
          loaded = true;
          button.hidden = true;
          if (status) {
            status.textContent = '';
            status.hidden = true;
          }
        };
        script.onerror = function () {
          loading = false;
          script.remove();
          button.disabled = false;
          button.textContent = 'Retry loading comments';
          if (status) status.textContent = 'Comments could not be loaded. Please check your connection and try again.';
        };
        container.append(script);
      });
    });
  }

  initNavigation();
  initCode();
  initToc();
  initSearch();
  initAnalytics();
  initComments();
}());
