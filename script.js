/**
 * script.js – LearnFlow AI client-side logic
 * Covers: Chart.js charts, quiz timer, option selection, form validation, animations
 */

// ─── Auto-dismiss alerts ──────────────────────────────────────────────
document.querySelectorAll('.alert-dismissible').forEach(alert => {
  setTimeout(() => {
    alert.style.transition = 'opacity .5s';
    alert.style.opacity = '0';
    setTimeout(() => alert.remove(), 500);
  }, 4000);
});

// ─── Quiz option selection (styled radio) ────────────────────────────
document.querySelectorAll('.option-label').forEach(label => {
  label.addEventListener('click', () => {
    const name = label.querySelector('input[type="radio"]')?.name;
    if (!name) return;
    document.querySelectorAll(`input[name="${name}"]`).forEach(r => {
      r.closest('.option-label')?.classList.remove('selected');
    });
    label.classList.add('selected');
    label.querySelector('input[type="radio"]').checked = true;
  });
});

// ─── Quiz Timer ───────────────────────────────────────────────────────
const timerEl = document.getElementById('quiz-timer');
if (timerEl) {
  let seconds = parseInt(timerEl.dataset.seconds || '300', 10);

  const format = s => {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  };

  timerEl.textContent = format(seconds);

  const interval = setInterval(() => {
    seconds--;
    timerEl.textContent = format(seconds);

    if (seconds <= 60) {
      timerEl.closest('.timer-box')?.classList.add('warning');
    }
    if (seconds <= 0) {
      clearInterval(interval);
      document.getElementById('quiz-form')?.submit();
    }
  }, 1000);
}

// ─── Engagement / rating buttons ─────────────────────────────────────
document.querySelectorAll('.rating-group').forEach(group => {
  group.querySelectorAll('.rating-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.rating-btn').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      const hiddenInput = group.closest('form')?.querySelector(`#${group.dataset.target}`);
      if (hiddenInput) hiddenInput.value = btn.dataset.value;
    });
  });
});

// ─── Animate stat values (count-up) ──────────────────────────────────
function animateCounter(el) {
  const target = parseFloat(el.dataset.target || el.textContent);
  const isFloat = el.dataset.float === 'true';
  const duration = 1200;
  const start = performance.now();

  const step = now => {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const value = target * ease;
    el.textContent = isFloat ? value.toFixed(1) : Math.round(value);
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

document.querySelectorAll('[data-counter]').forEach(el => {
  const observer = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      animateCounter(el);
      observer.disconnect();
    }
  });
  observer.observe(el);
});

// ─── Progress bar animation ───────────────────────────────────────────
document.querySelectorAll('.progress-bar[data-width]').forEach(bar => {
  setTimeout(() => {
    bar.style.width = bar.dataset.width + '%';
  }, 200);
});

// ─── Chart.js – Analytics Page ────────────────────────────────────────
if (document.getElementById('weeklyHoursChart')) {
  fetch('/api/chart-data')
    .then(r => r.json())
    .then(data => {
      const baseFont = { family: "'Inter', sans-serif" };
      const gridColor = 'rgba(0,0,0,.06)';
      const tooltipStyle = {
        backgroundColor: '#1F2937',
        titleFont: { ...baseFont, weight: '700', size: 13 },
        bodyFont: { ...baseFont, size: 12 },
        padding: 10,
        cornerRadius: 8,
      };

      // Weekly study hours bar chart
      new Chart(document.getElementById('weeklyHoursChart'), {
        type: 'bar',
        data: {
          labels: data.weekly_hours.labels,
          datasets: [{
            label: 'Study Hours',
            data: data.weekly_hours.data,
            backgroundColor: 'rgba(79,70,229,.8)',
            borderRadius: 6,
            borderSkipped: false,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: tooltipStyle },
          scales: {
            y: { beginAtZero: true, grid: { color: gridColor }, ticks: { font: baseFont } },
            x: { grid: { display: false }, ticks: { font: baseFont } }
          }
        }
      });

      // Quiz score trend line chart
      new Chart(document.getElementById('quizTrendChart'), {
        type: 'line',
        data: {
          labels: data.quiz_trend.labels,
          datasets: [{
            label: 'Quiz Score (%)',
            data: data.quiz_trend.data,
            borderColor: '#4F46E5',
            backgroundColor: 'rgba(79,70,229,.08)',
            fill: true,
            tension: .4,
            pointBackgroundColor: '#4F46E5',
            pointRadius: 5,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: tooltipStyle },
          scales: {
            y: { min: 0, max: 100, grid: { color: gridColor }, ticks: { font: baseFont } },
            x: { grid: { display: false }, ticks: { font: baseFont } }
          }
        }
      });

      // Topic-wise performance horizontal bar
      new Chart(document.getElementById('topicPerfChart'), {
        type: 'bar',
        data: {
          labels: data.topic_perf.labels,
          datasets: [{
            label: 'Average Score (%)',
            data: data.topic_perf.data,
            backgroundColor: data.topic_perf.data.map(v =>
              v >= 70 ? 'rgba(16,185,129,.8)' : v >= 55 ? 'rgba(245,158,11,.8)' : 'rgba(239,68,68,.8)'
            ),
            borderRadius: 6,
            borderSkipped: false,
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: tooltipStyle },
          scales: {
            x: { min: 0, max: 100, grid: { color: gridColor }, ticks: { font: baseFont } },
            y: { grid: { display: false }, ticks: { font: baseFont } }
          }
        }
      });

      // Engagement trend area chart
      new Chart(document.getElementById('engagementChart'), {
        type: 'line',
        data: {
          labels: data.engagement.labels,
          datasets: [{
            label: 'Engagement (1-5)',
            data: data.engagement.data,
            borderColor: '#7C3AED',
            backgroundColor: 'rgba(124,58,237,.1)',
            fill: true,
            tension: .4,
            pointBackgroundColor: '#7C3AED',
            pointRadius: 4,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: tooltipStyle },
          scales: {
            y: { min: 0, max: 5, grid: { color: gridColor }, ticks: { font: baseFont } },
            x: { grid: { display: false }, ticks: { font: baseFont } }
          }
        }
      });
    })
    .catch(err => console.warn('Chart data fetch error:', err));
}

// ─── Mini drift chart on dashboard ───────────────────────────────────
if (document.getElementById('miniScoreChart')) {
  fetch('/api/chart-data')
    .then(r => r.json())
    .then(data => {
      new Chart(document.getElementById('miniScoreChart'), {
        type: 'line',
        data: {
          labels: data.quiz_trend.labels.slice(-6),
          datasets: [{
            data: data.quiz_trend.data.slice(-6),
            borderColor: '#4F46E5',
            backgroundColor: 'rgba(79,70,229,.08)',
            fill: true,
            tension: .4,
            pointRadius: 3,
            borderWidth: 2,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: { x: { display: false }, y: { display: false } }
        }
      });
    })
    .catch(() => {});
}

// ─── Password show/hide toggle ────────────────────────────────────────
document.querySelectorAll('.password-toggle').forEach(btn => {
  btn.addEventListener('click', () => {
    const input = btn.closest('.password-wrapper')?.querySelector('input');
    if (!input) return;
    if (input.type === 'password') {
      input.type = 'text';
      btn.innerHTML = '🙈';
    } else {
      input.type = 'password';
      btn.innerHTML = '👁️';
    }
  });
});

// ─── Scroll-based fade-in animation ──────────────────────────────────
const fadeEls = document.querySelectorAll('.fade-in');
if (fadeEls.length) {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        io.unobserve(e.target);
      }
    });
  }, { threshold: .1 });
  fadeEls.forEach(el => io.observe(el));
}
