'use strict';

const BACKEND = 'http://localhost:5000';

/* ── Chart / episode state ──────────────── */
let socket        = null;
let rewardChart   = null;
let avgChart      = null;
let rewardHistory = [];
let avgHistory    = [];
let epLabels      = [];
let bestScore     = 0;
let totalEps      = 0;

/* ── Canvas ─────────────────────────────── */
let canvas, ctx;
const CANVAS_PX = 400;

/* ════════════════════════════════════════
   SNAKE ANIMATION ENGINE
   ─────────────────────────────────────
   • All incoming snake_state frames go into snakeQueue.
   • requestAnimationFrame drains the queue at a steady,
     user-controlled pace (BASE_MS / speedMult ms per step).
   • Every frame transition lerps each body segment so the
     snake glides instead of jumping.
   • Score display is synced to the FRAME being rendered,
     so the viewer sees food eaten at the right moment.
   • Food-eating frames are preserved when trimming overflow.
════════════════════════════════════════ */
const BASE_MS   = 150;   // ms per displayed step at 1× speed
const MAX_QUEUE = 80;    // buffer cap before trimming

let speedMult  = 1.0;    // controlled by slider
let snakeQueue = [];     // incoming frame buffer
let fromFrame  = null;   // frame we're animating FROM
let toFrame    = null;   // frame we're animating TO
let animT      = 1.0;    // 0→1 transition progress (1 = ready for next)
let lastTs     = 0;
let rafId      = null;

/* ════════════════════════════════════════
   BOOT
════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
    canvas        = document.getElementById('snake-canvas');
    canvas.width  = CANVAS_PX;
    canvas.height = CANVAS_PX;
    ctx           = canvas.getContext('2d');

    initCharts();
    bindForm();
    bindButtons();
    connectSocket();
    drawIdleGrid();
});

/* ════════════════════════════════════════
   SOCKET.IO
════════════════════════════════════════ */
function connectSocket() {
    socket = io(BACKEND, { transports: ['websocket', 'polling'] });

    socket.on('connect',           ()  => console.log('[socket] connected'));
    socket.on('disconnect',        ()  => console.log('[socket] disconnected'));
    socket.on('snake_state',       onSnakeState);
    socket.on('episode_update',    onEpisodeUpdate);
    socket.on('training_complete', onComplete);
    socket.on('training_stopped',  onStopped);
    socket.on('training_error',    onError);
}

/* ════════════════════════════════════════
   FORM & BUTTONS
════════════════════════════════════════ */
function bindForm() {
    document.getElementById('config-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const errBox = document.getElementById('connect-error');
        errBox.style.display = 'none';

        const config = {
            gamma:        parseFloat(document.getElementById('gamma').value),
            alpha:        parseFloat(document.getElementById('alpha').value),
            num_episodes: parseInt(document.getElementById('num_episodes').value, 10),
        };
        totalEps = config.num_episodes;

        try {
            const res = await fetch(`${BACKEND}/api/start`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(config),
            });
            if (res.ok) {
                showTraining();
            } else {
                const data = await res.json();
                alert(data.error || 'Failed to start training');
            }
        } catch (_) {
            errBox.style.display = 'block';
        }
    });
}

function bindButtons() {
    document.getElementById('stop-btn').addEventListener('click', async () => {
        await fetch(`${BACKEND}/api/stop`, { method: 'POST' }).catch(() => {});
        setStatus('Stopping…', 'stopping');
    });

    document.getElementById('back-btn').addEventListener('click', showWelcome);

    /* Speed slider → update speedMult */
    const slider = document.getElementById('speed-slider');
    const valEl  = document.getElementById('speed-val');
    slider.addEventListener('input', () => {
        speedMult            = parseFloat(slider.value);
        valEl.textContent    = `${speedMult.toFixed(1)}x`;
    });
}

/* ════════════════════════════════════════
   SCREEN TRANSITIONS
════════════════════════════════════════ */
function showTraining() {
    document.getElementById('welcome-screen').classList.remove('active');
    document.getElementById('training-screen').classList.add('active');

    stopPlayback();
    bestScore     = 0;
    rewardHistory = [];
    avgHistory    = [];
    epLabels      = [];
    resetCharts();

    document.getElementById('m-best').textContent        = '0';
    document.getElementById('m-avg').textContent         = '—';
    document.getElementById('m-pct').textContent         = '0%';
    document.getElementById('m-pct-sub').textContent     = `0 / ${totalEps} episodes`;
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('h-episode').textContent     = `— / ${totalEps}`;
    document.getElementById('h-elapsed').textContent     = '0s';
    document.getElementById('h-epsilon').textContent     = '1.000';
    document.getElementById('gi-score').textContent      = '0';
    document.getElementById('gi-episode').textContent    = '—';
    document.getElementById('gi-reward').textContent     = '—';
    document.getElementById('badge-reward').textContent  = '—';
    document.getElementById('badge-avg').textContent     = '—';
    document.getElementById('canvas-overlay').style.display = 'flex';

    setStatus('Training', 'running');
    drawIdleGrid();
}

function showWelcome() {
    document.getElementById('training-screen').classList.remove('active');
    document.getElementById('welcome-screen').classList.add('active');
}

/* ════════════════════════════════════════
   INCOMING FRAME HANDLER
════════════════════════════════════════ */
function onSnakeState(data) {
    document.getElementById('canvas-overlay').style.display = 'none';
    snakeQueue.push(data);
    if (snakeQueue.length > MAX_QUEUE) trimQueue();
    if (!rafId) { lastTs = 0; rafId = requestAnimationFrame(animLoop); }
}

/*
 * Drop oldest non-food frames when the buffer overflows.
 * A "food frame" is one where the snake's score increased —
 * these are preserved so the viewer always sees food being eaten.
 */
function trimQueue() {
    const target = Math.floor(MAX_QUEUE * 0.55);

    // Mark every frame where the score rose vs the previous frame
    const isFoodFrame = new Array(snakeQueue.length).fill(false);
    isFoodFrame[snakeQueue.length - 1] = true; // always keep most recent
    let prevScore = snakeQueue[0].score;
    for (let i = 1; i < snakeQueue.length; i++) {
        if (snakeQueue[i].score > prevScore) isFoodFrame[i] = true;
        prevScore = snakeQueue[i].score;
    }

    // Remove oldest non-food frames until we reach target size
    const toRemove = snakeQueue.length - target;
    const result   = [];
    let   removed  = 0;
    for (let i = 0; i < snakeQueue.length; i++) {
        if (!isFoodFrame[i] && removed < toRemove) { removed++; continue; }
        result.push(snakeQueue[i]);
    }
    snakeQueue = result;
}

/* ════════════════════════════════════════
   ANIMATION LOOP
════════════════════════════════════════ */
function stopPlayback() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId      = null;
    snakeQueue = [];
    fromFrame  = null;
    toFrame    = null;
    animT      = 1.0;
    lastTs     = 0;
}

function animLoop(ts) {
    rafId = requestAnimationFrame(animLoop);

    // Cap dt so a tab-focus wake-up can't blast through the queue
    const dt     = lastTs ? Math.min(ts - lastTs, 100) : 0;
    lastTs       = ts;
    const stepMs = BASE_MS / speedMult;

    if (animT < 1) {
        /* ── Mid-transition: advance t and redraw ── */
        animT = Math.min(1, animT + dt / stepMs);
        drawInterp(fromFrame, toFrame, easeInOut(animT));

        /* Score syncs to the frame only when fully visible (t = 1) */
        if (animT >= 1) syncFrameStats(toFrame);

    } else if (snakeQueue.length > 0) {
        /* ── Ready for next frame ── */
        fromFrame = toFrame || snakeQueue[0];
        toFrame   = snakeQueue.shift();
        animT     = 0;

    } else if (toFrame) {
        /* ── Queue empty: hold last frame, no flicker ── */
        drawInterp(toFrame, toFrame, 1);
    }
}

/* Smooth ease-in-out curve */
function easeInOut(t) { return t < 0.5 ? 2*t*t : -1 + (4 - 2*t)*t; }

/* Update the live-game stats to match the rendered frame */
function syncFrameStats(frame) {
    if (!frame) return;
    document.getElementById('gi-score').textContent   = frame.score;
    document.getElementById('gi-episode').textContent = frame.episode;
}

/* ════════════════════════════════════════
   CANVAS RENDERING
════════════════════════════════════════ */

/*
 * drawInterp — render a frame interpolated between `from` and `to`.
 *
 * Food handling: during t=0→1 we show FROM's food so the viewer
 * sees the snake glide toward the food cell; at t=1 the score
 * updates and food snaps to the new position.
 *
 * Snake segments: each body part lerps from its old row/col to
 * its new row/col, so movement is pixel-smooth regardless of how
 * many steps the backend skipped between emissions.
 */
function drawInterp(from, to, t) {
    if (!to) return;

    /* Episode boundary or no previous state → snap, skip lerp */
    if (!from || from.episode !== to.episode) {
        drawSnap(to);
        return;
    }

    const G    = to.grid_size;
    const cell = CANVAS_PX / G;

    /* Background */
    ctx.fillStyle = '#080c18';
    ctx.fillRect(0, 0, CANVAS_PX, CANVAS_PX);

    /* Grid lines */
    ctx.strokeStyle = '#111827';
    ctx.lineWidth   = 0.5;
    for (let i = 0; i <= G; i++) {
        ctx.beginPath(); ctx.moveTo(i * cell, 0);       ctx.lineTo(i * cell, CANVAS_PX); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0,       i * cell); ctx.lineTo(CANVAS_PX, i * cell); ctx.stroke();
    }

    /* Food — show from-food during the glide so score & food are in sync */
    const food = t < 1 ? from.food : to.food;
    ctx.save();
    ctx.shadowBlur  = 14;
    ctx.shadowColor = '#ff3355';
    ctx.fillStyle   = '#ff3355';
    roundRect(ctx, food[1] * cell + 2, food[0] * cell + 2, cell - 4, cell - 4, 3);
    ctx.fill();
    ctx.restore();

    /* Snake — lerp every segment */
    const len = to.snake.length;
    to.snake.forEach(([tr, tc], idx) => {
        const fp  = from.snake[idx];
        const lc  = fp ? fp[1] + (tc - fp[1]) * t : tc;  // interpolated col
        const lr  = fp ? fp[0] + (tr - fp[0]) * t : tr;  // interpolated row
        const px  = lc * cell;
        const py  = lr * cell;
        const isHead = idx === 0;

        ctx.save();
        if (isHead) {
            ctx.shadowBlur  = 14;
            ctx.shadowColor = '#00e87a';
            ctx.fillStyle   = '#00e87a';
        } else {
            ctx.fillStyle = `rgba(0,200,100,${Math.max(0.25, 1 - (idx / len) * 0.75)})`;
        }
        roundRect(ctx, px + 1, py + 1, cell - 2, cell - 2, isHead ? 4 : 2);
        ctx.fill();
        ctx.restore();
    });
}

/* Snap draw — no interpolation, used for episode resets and initial frame */
function drawSnap(state) {
    if (!state) return;
    const G    = state.grid_size;
    const cell = CANVAS_PX / G;

    ctx.fillStyle = '#080c18';
    ctx.fillRect(0, 0, CANVAS_PX, CANVAS_PX);

    ctx.strokeStyle = '#111827';
    ctx.lineWidth   = 0.5;
    for (let i = 0; i <= G; i++) {
        ctx.beginPath(); ctx.moveTo(i * cell, 0);       ctx.lineTo(i * cell, CANVAS_PX); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0,       i * cell); ctx.lineTo(CANVAS_PX, i * cell); ctx.stroke();
    }

    const [fr, fc] = state.food;
    ctx.save();
    ctx.shadowBlur  = 14;
    ctx.shadowColor = '#ff3355';
    ctx.fillStyle   = '#ff3355';
    roundRect(ctx, fc * cell + 2, fr * cell + 2, cell - 4, cell - 4, 3);
    ctx.fill();
    ctx.restore();

    const len = state.snake.length;
    state.snake.forEach(([r, c], idx) => {
        const isHead = idx === 0;
        ctx.save();
        if (isHead) {
            ctx.shadowBlur  = 14;
            ctx.shadowColor = '#00e87a';
            ctx.fillStyle   = '#00e87a';
        } else {
            ctx.fillStyle = `rgba(0,200,100,${Math.max(0.25, 1 - (idx / len) * 0.75)})`;
        }
        roundRect(ctx, c * cell + 1, r * cell + 1, cell - 2, cell - 2, isHead ? 4 : 2);
        ctx.fill();
        ctx.restore();
    });

    syncFrameStats(state);
}

function drawIdleGrid() {
    const cell = CANVAS_PX / 20;
    ctx.fillStyle = '#080c18';
    ctx.fillRect(0, 0, CANVAS_PX, CANVAS_PX);
    ctx.strokeStyle = '#111827';
    ctx.lineWidth   = 0.5;
    for (let i = 0; i <= 20; i++) {
        ctx.beginPath(); ctx.moveTo(i * cell, 0);       ctx.lineTo(i * cell, CANVAS_PX); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0,       i * cell); ctx.lineTo(CANVAS_PX, i * cell); ctx.stroke();
    }
}

/* Polyfill for ctx.roundRect on older browsers */
function roundRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, r);
    } else {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y,     x + w, y + r,     r);
        ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h); ctx.arcTo(x,     y + h, x,     y + h - r, r);
        ctx.lineTo(x, y + r);     ctx.arcTo(x,     y,     x + r, y,         r);
        ctx.closePath();
    }
}

/* ════════════════════════════════════════
   EPISODE UPDATE (charts + header stats)
   Score/episode display is handled by syncFrameStats above,
   not here, so it stays synced to the rendered frame.
════════════════════════════════════════ */
function onEpisodeUpdate(data) {
    const { episode, total_episodes, reward, avg_reward, epsilon, elapsed, best_score } = data;

    document.getElementById('h-episode').textContent = `${episode} / ${total_episodes}`;
    document.getElementById('h-elapsed').textContent = fmtTime(elapsed);
    document.getElementById('h-epsilon').textContent = epsilon.toFixed(3);
    document.getElementById('gi-reward').textContent = reward.toFixed(1);
    document.getElementById('m-avg').textContent     = avg_reward.toFixed(2);

    if (best_score > bestScore) {
        bestScore = best_score;
        document.getElementById('m-best').textContent = bestScore;
    }

    const pct = Math.round((episode / total_episodes) * 100);
    document.getElementById('m-pct').textContent         = `${pct}%`;
    document.getElementById('m-pct-sub').textContent     = `${episode} / ${total_episodes} episodes`;
    document.getElementById('progress-fill').style.width = `${pct}%`;

    document.getElementById('badge-reward').textContent = reward.toFixed(1);
    document.getElementById('badge-avg').textContent    = avg_reward.toFixed(2);

    epLabels.push(episode);
    rewardHistory.push(reward);
    avgHistory.push(avg_reward);
    if (epLabels.length > 400) { epLabels.shift(); rewardHistory.shift(); avgHistory.shift(); }
    updateCharts();
}

function onComplete(data) {
    setStatus('Solved ✓', 'solved');
    showToast(`Solved in ${data.episode} episodes! Model saved as snake_model.keras`, 'green');
}

function onStopped() { setStatus('Stopped', 'stopped'); }

function onError(data) {
    setStatus('Error', 'error');
    console.error('Training error:', data.message);
    showToast('Training error — check console for details.', 'red');
}

/* ════════════════════════════════════════
   CHARTS
════════════════════════════════════════ */
function makeChartOpts() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#111827',
                borderColor: '#243060',
                borderWidth: 1,
                titleColor: '#e4ecff',
                bodyColor: '#8899bb',
                padding: 10,
                callbacks: { title: (items) => `Episode ${items[0].label}` },
            },
        },
        scales: {
            x: { grid: { color: '#111827' }, ticks: { maxTicksLimit: 6, color: '#4a5878', font: { size: 11 } } },
            y: { grid: { color: '#111827' }, ticks: { maxTicksLimit: 5, color: '#4a5878', font: { size: 11 } } },
        },
    };
}

function initCharts() {
    Chart.defaults.color       = '#4a5878';
    Chart.defaults.borderColor = '#1c2540';

    rewardChart = new Chart(document.getElementById('reward-chart'), {
        type: 'line',
        data: { labels: [], datasets: [{ data: [], fill: true, tension: 0.25, borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.06)', borderWidth: 1.5, pointRadius: 0 }] },
        options: makeChartOpts(),
    });

    avgChart = new Chart(document.getElementById('avg-chart'), {
        type: 'line',
        data: { labels: [], datasets: [{ data: [], fill: true, tension: 0.4, borderColor: '#00e87a', backgroundColor: 'rgba(0,232,122,0.07)', borderWidth: 2, pointRadius: 0 }] },
        options: makeChartOpts(),
    });
}

function updateCharts() {
    rewardChart.data.labels           = epLabels;
    rewardChart.data.datasets[0].data = rewardHistory;
    rewardChart.update('none');
    avgChart.data.labels              = epLabels;
    avgChart.data.datasets[0].data    = avgHistory;
    avgChart.update('none');
}

function resetCharts() {
    rewardChart.data.labels = []; rewardChart.data.datasets[0].data = [];
    avgChart.data.labels    = []; avgChart.data.datasets[0].data    = [];
    rewardChart.update(); avgChart.update();
}

/* ════════════════════════════════════════
   HELPERS
════════════════════════════════════════ */
function setStatus(text, cls) {
    document.getElementById('status-badge').className    = `status-badge ${cls}`;
    document.getElementById('status-text').textContent   = text;
}

function fmtTime(sec) {
    if (sec < 60)   return `${Math.round(sec)}s`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
    return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

function showToast(msg, color) {
    const t = document.createElement('div');
    t.textContent = msg;
    Object.assign(t.style, {
        position: 'fixed', bottom: '24px', left: '50%', transform: 'translateX(-50%)',
        background: color === 'green' ? 'rgba(0,232,122,.12)' : 'rgba(255,51,85,.12)',
        border: `1px solid ${color === 'green' ? 'rgba(0,232,122,.3)' : 'rgba(255,51,85,.3)'}`,
        color: color === 'green' ? '#00e87a' : '#ff3355',
        padding: '12px 22px', borderRadius: '10px',
        fontSize: '13px', fontWeight: '600',
        backdropFilter: 'blur(8px)', zIndex: '9999', transition: 'opacity .4s',
    });
    document.body.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 400); }, 4000);
}
