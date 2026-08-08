(() => {
  const year = document.querySelector('#year');
  if (year) year.textContent = new Date().getFullYear();

  const links = [...document.querySelectorAll('.section-nav a[href^="#"]')];
  const items = links
    .map((link) => ({ link, section: document.querySelector(link.getAttribute('href')) }))
    .filter((item) => item.section);
  if (!items.length) return;

  const sidebarScroller = document.querySelector('.profile-inner');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let activeId = '';
  let ticking = false;

  function revealActiveLink(link) {
    if (!sidebarScroller || sidebarScroller.scrollHeight <= sidebarScroller.clientHeight + 1) return;
    const scrollerRect = sidebarScroller.getBoundingClientRect();
    const linkRect = link.getBoundingClientRect();
    const margin = 18;
    let delta = 0;

    if (linkRect.bottom > scrollerRect.bottom - margin) {
      delta = linkRect.bottom - scrollerRect.bottom + margin;
    } else if (linkRect.top < scrollerRect.top + margin) {
      delta = linkRect.top - scrollerRect.top - margin;
    }

    if (Math.abs(delta) > 1) {
      sidebarScroller.scrollBy({
        top: delta,
        behavior: prefersReducedMotion ? 'auto' : 'smooth'
      });
    }
  }

  function setActive(item) {
    if (!item || item.section.id === activeId) return;
    activeId = item.section.id;
    for (const candidate of items) {
      if (candidate === item) candidate.link.setAttribute('aria-current', 'true');
      else candidate.link.removeAttribute('aria-current');
    }
    revealActiveLink(item.link);
  }

  function updateActiveSection() {
    ticking = false;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
    const documentHeight = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
    const atBottom = window.scrollY + viewportHeight >= documentHeight - 4;

    if (atBottom) {
      setActive(items[items.length - 1]);
      return;
    }

    // A section becomes active when its top passes roughly the upper third of
    // the viewport. Unlike the old IntersectionObserver margins, this remains
    // reliable for tall Experience sections and short Contact sections.
    const anchorY = Math.max(96, Math.min(viewportHeight * 0.32, 280));
    let current = items[0];
    for (const item of items) {
      if (item.section.getBoundingClientRect().top <= anchorY) current = item;
      else break;
    }
    setActive(current);
  }

  function requestUpdate() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(updateActiveSection);
  }

  window.addEventListener('scroll', requestUpdate, { passive: true });
  window.addEventListener('resize', requestUpdate, { passive: true });
  window.addEventListener('hashchange', requestUpdate);
  links.forEach((link) => link.addEventListener('click', () => {
    const item = items.find((candidate) => candidate.link === link);
    if (item) setActive(item);
  }));

  updateActiveSection();
})();
