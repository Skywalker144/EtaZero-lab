(() => {
  const root = document.documentElement;
  const system = window.matchMedia('(prefers-color-scheme: dark)');
  let preference = 'system';
  try {
    const saved = JSON.parse(localStorage.getItem('etazero-ui') || '{}');
    if (['light', 'dark', 'system'].includes(saved?.theme)) preference = saved.theme;
  } catch { /* Browser storage is optional. */ }

  function applyTheme() {
    root.dataset.themePreference = preference;
    root.dataset.theme = preference === 'system' ? (system.matches ? 'dark' : 'light') : preference;
  }
  applyTheme();
  system.addEventListener('change', applyTheme);
  document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('theme');
    select.value = preference;
    select.addEventListener('change', () => {
      preference = select.value;
      applyTheme();
      savePreferences();
    });
  });
})();
