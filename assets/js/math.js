(function () {
  'use strict';

  document.querySelectorAll('.prose script[type^="math/tex"]').forEach(function (script) {
    const display = /;\s*mode\s*=\s*display/i.test(script.type);
    const delimiters = display ? ['\\[', '\\]'] : ['\\(', '\\)'];
    script.replaceWith(document.createTextNode(delimiters[0] + script.textContent + delimiters[1]));
  });

  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    },
    options: {
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    startup: {
      typeset: true
    }
  };
}());
