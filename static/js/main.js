/* CowCareAI Main Platform Client JS */

document.addEventListener('DOMContentLoaded', () => {
  // Theme Toggle Management
  const themeBtn = document.getElementById('themeToggleBtn');
  const storedTheme = localStorage.getItem('cowcare_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', storedTheme);
  updateThemeIcon(storedTheme);

  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('cowcare_theme', newTheme);
      updateThemeIcon(newTheme);
    });
  }

  function updateThemeIcon(theme) {
    if (!themeBtn) return;
    const icon = themeBtn.querySelector('i');
    if (icon) {
      icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
  }

  // Auto Dismiss Flash Messages after 5 seconds
  const flashAlerts = document.querySelectorAll('.flash');
  flashAlerts.forEach(alert => {
    setTimeout(() => {
      alert.style.opacity = '0';
      setTimeout(() => alert.remove(), 300);
    }, 5000);
  });
});

// Acknowledge Alert Handler
function acknowledgeAlert(alertId, btnElement) {
  fetch(`/api/alert/${alertId}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      if (btnElement) {
        btnElement.classList.remove('btn-action');
        btnElement.classList.add('badge-status', 'green');
        btnElement.innerHTML = '<i class="fas fa-check"></i> Acknowledged';
        btnElement.disabled = true;
      }
    }
  })
  .catch(err => console.error('Acknowledge alert error:', err));
}