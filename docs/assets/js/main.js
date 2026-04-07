/* ===== mlx-tune Docs - Main JS ===== */

(function () {
  'use strict';

  // --- Theme Toggle ---
  const THEME_KEY = 'mlx-tune-theme';

  function getPreferredTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;
    return 'light'; // default to light
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    const btn = document.querySelector('.theme-toggle');
    if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
  }

  // Apply theme immediately (before DOM ready to avoid flash)
  setTheme(getPreferredTheme());

  document.addEventListener('DOMContentLoaded', function () {
    // Theme toggle button
    const themeBtn = document.querySelector('.theme-toggle');
    if (themeBtn) {
      themeBtn.textContent = getPreferredTheme() === 'dark' ? '☀️' : '🌙';
      themeBtn.addEventListener('click', function () {
        const current = document.documentElement.getAttribute('data-theme');
        setTheme(current === 'dark' ? 'light' : 'dark');
      });
    }

    // --- Mobile Menu Toggle ---
    const mobileToggle = document.querySelector('.mobile-toggle');
    const navLinks = document.querySelector('.nav-links');
    if (mobileToggle && navLinks) {
      mobileToggle.addEventListener('click', function () {
        navLinks.classList.toggle('mobile-open');
        mobileToggle.textContent = navLinks.classList.contains('mobile-open') ? '✕' : '☰';
      });
    }

    // --- Copy to Clipboard ---
    // Hero install command
    const heroInstall = document.querySelector('.hero-install');
    if (heroInstall) {
      heroInstall.addEventListener('click', function () {
        const text = heroInstall.getAttribute('data-copy') || heroInstall.textContent.trim();
        copyToClipboard(text, heroInstall);
      });
    }

    // Code block copy buttons
    document.querySelectorAll('.code-block-wrapper').forEach(function (wrapper) {
      const btn = wrapper.querySelector('.copy-btn');
      const code = wrapper.querySelector('code');
      if (btn && code) {
        btn.addEventListener('click', function () {
          copyToClipboard(code.textContent, btn, 'Copied!');
        });
      }
    });

    function copyToClipboard(text, element, tooltipText) {
      navigator.clipboard.writeText(text).then(function () {
        // Show tooltip
        const tooltip = element.querySelector('.copied-tooltip');
        if (tooltip) {
          tooltip.classList.add('show');
          setTimeout(function () { tooltip.classList.remove('show'); }, 1500);
        } else if (tooltipText) {
          const orig = element.textContent;
          element.textContent = tooltipText;
          setTimeout(function () { element.textContent = orig; }, 1500);
        }
      });
    }

    // --- Sidebar Scroll Spy ---
    const sidebarLinks = document.querySelectorAll('.sidebar-nav a');
    if (sidebarLinks.length > 0) {
      const headings = [];
      sidebarLinks.forEach(function (link) {
        const id = link.getAttribute('href');
        if (id && id.startsWith('#')) {
          const el = document.querySelector(id);
          if (el) headings.push({ el: el, link: link });
        }
      });

      function updateScrollSpy() {
        const scrollPos = window.scrollY + 120;
        let current = headings[0];
        for (var i = 0; i < headings.length; i++) {
          if (headings[i].el.offsetTop <= scrollPos) {
            current = headings[i];
          }
        }
        sidebarLinks.forEach(function (l) { l.classList.remove('active'); });
        if (current) current.link.classList.add('active');
      }

      window.addEventListener('scroll', updateScrollSpy, { passive: true });
      updateScrollSpy();
    }

    // --- Collapsible Sections ---
    document.querySelectorAll('.collapsible-header').forEach(function (header) {
      header.addEventListener('click', function () {
        header.classList.toggle('open');
        const content = header.nextElementSibling;
        if (content && content.classList.contains('collapsible-content')) {
          content.classList.toggle('open');
        }
      });
    });

    // --- Example Code Toggles ---
    document.querySelectorAll('.example-toggle').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const card = btn.closest('.example-card');
        const code = card.querySelector('.example-code');
        if (code) {
          code.classList.toggle('open');
          btn.textContent = code.classList.contains('open') ? 'Hide Code' : 'View Code';
        }
      });
    });

    // --- Active Nav Link ---
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(function (link) {
      const href = link.getAttribute('href');
      if (href === currentPage || (currentPage === '' && href === 'index.html')) {
        link.classList.add('active');
      }
    });

    // --- Search Modal (Cmd+K / Ctrl+K) ---
    var searchOverlay = document.querySelector('.search-overlay');
    var searchInput = document.querySelector('.search-input');
    var searchResults = document.querySelector('.search-results');
    var searchTrigger = document.querySelector('.search-trigger');

    var PAGE_ICONS = {
      Home: '#', LLM: 'T', VLM: '\u{1F441}', OCR: '\u{1F4C4}',
      Audio: '\u{1F3A4}', Workflow: '\u{1F504}', Examples: '\u{1F4DD}',
      Help: '\u{2753}', API: '{}'
    };

    function openSearch() {
      if (!searchOverlay) return;
      searchOverlay.classList.add('open');
      searchInput.value = '';
      searchInput.focus();
      renderResults('');
    }

    function closeSearch() {
      if (!searchOverlay) return;
      searchOverlay.classList.remove('open');
    }

    function renderResults(query) {
      if (!searchResults || !window.MLX_SEARCH_INDEX) return;
      var q = query.toLowerCase().trim();

      if (!q) {
        // Show recent / popular
        var popular = window.MLX_SEARCH_INDEX.slice(0, 8);
        searchResults.innerHTML = popular.map(function (item) {
          return resultHTML(item, '');
        }).join('');
        return;
      }

      // Score each entry
      var scored = [];
      var terms = q.split(/\s+/);
      window.MLX_SEARCH_INDEX.forEach(function (item) {
        var score = 0;
        var titleLower = item.title.toLowerCase();
        var sectionLower = item.section.toLowerCase();
        var tagsLower = item.tags.toLowerCase();

        terms.forEach(function (term) {
          if (titleLower.indexOf(term) !== -1) score += 10;
          if (titleLower === q) score += 20; // exact title match
          if (sectionLower.indexOf(term) !== -1) score += 5;
          if (tagsLower.indexOf(term) !== -1) score += 3;
        });

        if (score > 0) scored.push({ item: item, score: score });
      });

      scored.sort(function (a, b) { return b.score - a.score; });
      var top = scored.slice(0, 10);

      if (top.length === 0) {
        searchResults.innerHTML =
          '<div class="search-empty">' +
          '<div class="search-empty-icon">\u{1F50D}</div>' +
          'No results for "<strong>' + escapeHTML(query) + '</strong>"' +
          '</div>';
        return;
      }

      searchResults.innerHTML = top.map(function (r) {
        return resultHTML(r.item, q);
      }).join('');
    }

    function resultHTML(item, query) {
      var icon = PAGE_ICONS[item.page] || '#';
      var title = query ? highlightMatch(item.title, query) : escapeHTML(item.title);
      return '<a class="search-result" href="' + item.url + '" data-url="' + item.url + '">' +
        '<div class="search-result-icon">' + icon + '</div>' +
        '<div class="search-result-text">' +
        '<div class="search-result-title">' + title + '</div>' +
        '<div class="search-result-section">' + escapeHTML(item.section) + '</div>' +
        '</div>' +
        '<span class="search-result-badge">' + escapeHTML(item.page) + '</span>' +
        '</a>';
    }

    function highlightMatch(text, query) {
      var escaped = escapeHTML(text);
      var terms = query.split(/\s+/);
      terms.forEach(function (term) {
        if (!term) return;
        var re = new RegExp('(' + escapeRegex(term) + ')', 'gi');
        escaped = escaped.replace(re, '<mark>$1</mark>');
      });
      return escaped;
    }

    function escapeHTML(s) {
      var div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }

    function escapeRegex(s) {
      return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // Keyboard navigation within results
    var activeIdx = -1;

    function navigateResults(direction) {
      var items = searchResults.querySelectorAll('.search-result');
      if (!items.length) return;
      activeIdx = Math.max(-1, Math.min(items.length - 1, activeIdx + direction));
      items.forEach(function (el, i) {
        el.classList.toggle('active', i === activeIdx);
        if (i === activeIdx) el.scrollIntoView({ block: 'nearest' });
      });
    }

    function selectActiveResult() {
      var items = searchResults.querySelectorAll('.search-result');
      if (activeIdx >= 0 && activeIdx < items.length) {
        window.location.href = items[activeIdx].getAttribute('href');
        closeSearch();
      }
    }

    // Event bindings
    if (searchTrigger) {
      searchTrigger.addEventListener('click', openSearch);
    }

    if (searchOverlay) {
      searchOverlay.addEventListener('click', function (e) {
        if (e.target === searchOverlay) closeSearch();
      });
    }

    if (searchInput) {
      searchInput.addEventListener('input', function () {
        activeIdx = -1;
        renderResults(searchInput.value);
      });
      searchInput.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowDown') { e.preventDefault(); navigateResults(1); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); navigateResults(-1); }
        else if (e.key === 'Enter') { e.preventDefault(); selectActiveResult(); }
      });
    }

    // Global Cmd+K / Ctrl+K
    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (searchOverlay && searchOverlay.classList.contains('open')) {
          closeSearch();
        } else {
          openSearch();
        }
      }
      if (e.key === 'Escape' && searchOverlay && searchOverlay.classList.contains('open')) {
        closeSearch();
      }
    });

    // --- Back to Top Button ---
    var backToTop = document.querySelector('.back-to-top');
    if (backToTop) {
      window.addEventListener('scroll', function () {
        backToTop.classList.toggle('visible', window.scrollY > 400);
      }, { passive: true });
      backToTop.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  });
})();
