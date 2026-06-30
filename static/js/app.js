function switchTab(name, btn) {
  document.querySelectorAll('.panel').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.ward-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  btn.classList.add('active');
}

async function safeJson(res) {
  const text = await res.text();
  try { return JSON.parse(text); }
  catch { return { error: `Error del servidor (HTTP ${res.status}). Revisa la consola.` }; }
}

const KM_FEATURE_LABELS = {
  'INDICE_OCUPACIONAL':        'Índice Ocupacional (%)',
  'PROMEDIO_DIAS_ESTADA':      'Promedio Días Estada',
  'INDICE_ROTACION':           'Índice Rotación',
  'NUMERO_EGRESOS':            'Nº Egresos',
  'PROMEDIO_CAMAS_DISPONIBLE': 'Prom. Camas Disponibles',
  'LETALIDAD':                 'Letalidad (%)',
  'MES':                       'Mes'
};

async function predictCluster() {
  document.getElementById('k_error').textContent = '';
  document.getElementById('k_result').style.display = 'none';

  const features = ['INDICE_OCUPACIONAL','PROMEDIO_DIAS_ESTADA','INDICE_ROTACION',
                    'NUMERO_EGRESOS','PROMEDIO_CAMAS_DISPONIBLE','LETALIDAD','MES'];
  const fd = new FormData();
  for (const f of features) fd.append(f, document.getElementById('k_' + f).value);

  const res  = await fetch('/predict/cluster', { method: 'POST', body: fd });
  const data = await safeJson(res);

  if (data.error) { document.getElementById('k_error').textContent = 'ERROR: ' + data.error; return; }

  const box = document.getElementById('k_result');
  box.style.display     = 'block';
  box.style.borderColor = data.color;

  const head = document.getElementById('k_vitals_head');
  head.style.background = data.color + '18';
  head.style.color      = data.color;
  document.getElementById('k_pulse').style.background = data.color;
  document.getElementById('k_status_text').textContent = 'CLUSTER ' + data.cluster + ' ASIGNADO';
  document.getElementById('k_icon').textContent = '● Cluster ' + data.cluster;
  document.getElementById('k_icon').style.color = data.color;
  document.getElementById('k_name').textContent = data.name;
  document.getElementById('k_name').style.color = data.color;
  document.getElementById('k_desc').textContent = data.description;

  const tbody = document.getElementById('k_centroid_body');
  tbody.innerHTML = '';
  for (const f of features) {
    const userVal = parseFloat(document.getElementById('k_' + f).value).toFixed(2);
    const centVal = data.centroid[f] !== undefined ? parseFloat(data.centroid[f]).toFixed(2) : '—';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${KM_FEATURE_LABELS[f]}</td><td><strong>${userVal}</strong></td><td>${centVal}</td>`;
    tbody.appendChild(tr);
  }
  document.getElementById('k_centroid_table').style.display = 'table';
}

// ════════════════════════════════════════════════════════════
//  PANEL 2 — IMPORTAR / EXPORTAR
// ════════════════════════════════════════════════════════════
let lastBatchToken = null;

const dropzone   = document.getElementById('dropzone');
const fileInput  = document.getElementById('file_input');
const dzFilename = document.getElementById('dz_filename');

if (dropzone) {
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected();
    }
  });
  fileInput.addEventListener('change', handleFileSelected);
}

async function handleFileSelected() {
  const file = fileInput.files[0];
  if (!file) return;

  dzFilename.textContent = '📄 ' + file.name;
  document.getElementById('batch_error').textContent = '';
  document.getElementById('batch_results_chart').style.display = 'none';

  const progressWrap = document.getElementById('progress_wrap');
  const progressFill = document.getElementById('progress_fill');
  progressWrap.style.display = 'block';
  progressFill.style.width = '15%';

  const fd = new FormData();
  fd.append('archivo', file);

  try {
    progressFill.style.width = '45%';
    const res  = await fetch('/batch/upload', { method: 'POST', body: fd });
    progressFill.style.width = '85%';
    const data = await safeJson(res);

    if (data.error) {
      document.getElementById('batch_error').textContent = 'ERROR: ' + data.error;
      progressWrap.style.display = 'none';
      return;
    }

    lastBatchToken = data.token;
    renderBatchResults(data);
    progressFill.style.width = '100%';
    setTimeout(() => { progressWrap.style.display = 'none'; progressFill.style.width = '0%'; }, 600);

  } catch (err) {
    document.getElementById('batch_error').textContent = 'ERROR: ' + err.message;
    progressWrap.style.display = 'none';
  }
}

function renderBatchResults(data) {
  document.getElementById('batch_results_chart').style.display = 'block';
  document.getElementById('batch_count_tag').textContent = data.total_rows + ' REGISTROS';

  const summary = document.getElementById('batch_summary');
  summary.innerHTML = '';
  const stats = [
    { label: 'Procesados', value: data.total_rows },
    { label: 'Errores',    value: data.error_count },
  ];
  for (const s of stats) {
    const div = document.createElement('div');
    div.className = 'summary-stat';
    div.innerHTML = `<div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div>`;
    summary.appendChild(div);
  }

  const thead = document.getElementById('batch_table_head');
  const tbody = document.getElementById('batch_table_body');
  thead.innerHTML = '';
  tbody.innerHTML = '';

  if (!data.preview || !data.preview.length) return;

  const cols = Object.keys(data.preview[0]);
  for (const c of cols) {
    const th = document.createElement('th');
    th.textContent = c;
    thead.appendChild(th);
  }

  const CLUSTER_COLORS = {
    0: { bg: '#fff7ed', text: '#9a3412' },
    1: { bg: '#ecfdf3', text: '#15803d' },
    2: { bg: '#eff6ff', text: '#1d4ed8' },
    3: { bg: '#fef2f2', text: '#b91c1c' },
  };

  for (const row of data.preview) {
    const tr = document.createElement('tr');
    for (const c of cols) {
      const td = document.createElement('td');
      const val = row[c];
      if (c === 'CLUSTER' && CLUSTER_COLORS[val]) {
        const chip = document.createElement('span');
        chip.className = 'badge-chip';
        chip.style.background = CLUSTER_COLORS[val].bg;
        chip.style.color = CLUSTER_COLORS[val].text;
        chip.textContent = 'C' + val;
        td.appendChild(chip);
      } else {
        td.textContent = val === null || val === undefined ? '—' : val;
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function exportReport(format) {
  if (!lastBatchToken) {
    document.getElementById('batch_error').textContent = 'Primero importa un archivo para generar el reporte.';
    return;
  }
  window.location.href = `/batch/export/${format}?token=${lastBatchToken}`;
}

// ════════════════════════════════════════════════════════════
//  EJEMPLOS PRECARGADOS — clustering
// ════════════════════════════════════════════════════════════

const CLUSTER_EXAMPLES = [
  {
    _label: 'C2 — Baja presión: Neonatología pequeña, ocupación ~47%',
    INDICE_OCUPACIONAL: 46.7, PROMEDIO_DIAS_ESTADA: 5.7,
    INDICE_ROTACION: 2.33, NUMERO_EGRESOS: 7,
    PROMEDIO_CAMAS_DISPONIBLE: 3.0, LETALIDAD: 0.00, MES: 9,
  },
  {
    _label: 'C1 — Operación típica: Med-Quirúrgico Pediátrico, ocupación ~77%',
    INDICE_OCUPACIONAL: 77.3, PROMEDIO_DIAS_ESTADA: 3.2,
    INDICE_ROTACION: 7.14, NUMERO_EGRESOS: 70,
    PROMEDIO_CAMAS_DISPONIBLE: 9.8, LETALIDAD: 0.00, MES: 12,
  },
  {
    _label: 'C0 — Alta ocupación hospital grande: ocupación ~93%, 617 egresos',
    INDICE_OCUPACIONAL: 92.9, PROMEDIO_DIAS_ESTADA: 8.5,
    INDICE_ROTACION: 3.54, NUMERO_EGRESOS: 617,
    PROMEDIO_CAMAS_DISPONIBLE: 162.2, LETALIDAD: 1.39, MES: 12,
  },
  {
    _label: 'C3 — UCI / Alta complejidad: estada 44 días, letalidad 75%',
    INDICE_OCUPACIONAL: 81.5, PROMEDIO_DIAS_ESTADA: 43.8,
    INDICE_ROTACION: 0.65, NUMERO_EGRESOS: 4,
    PROMEDIO_CAMAS_DISPONIBLE: 6.2, LETALIDAD: 75.00, MES: 2,
  },
];

function loadClusterExample(idx) {
  const ex = CLUSTER_EXAMPLES[idx];
  const fields = [
    'INDICE_OCUPACIONAL','PROMEDIO_DIAS_ESTADA','INDICE_ROTACION',
    'NUMERO_EGRESOS','PROMEDIO_CAMAS_DISPONIBLE','LETALIDAD','MES',
  ];
  for (const f of fields) {
    const el = document.getElementById('k_' + f);
    if (el) el.value = ex[f];
  }
  document.getElementById('k_result').style.display = 'none';
  document.getElementById('k_error').textContent = '';
}
