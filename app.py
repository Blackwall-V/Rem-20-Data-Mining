"""
app.py — REM 20 Predictor + Clustering
Flask web app for Entrega 2 — Minería de Datos / BIY7121

Tabs:
  1. Predicción de Índice Ocupacional (Random Forest regression)
  2. Clasificación de Patrón Operativo (K-Means clustering)

Run:
  pip install flask pandas numpy scikit-learn category_encoders
  python app.py
  → http://localhost:5000
"""

from flask import Flask, request, render_template_string, jsonify
import pickle, json
import numpy as np
import pandas as pd

app = Flask(__name__)

# ── Load all artifacts ────────────────────────────────────────────────────────
def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

rf_model       = load('models/rem20_occupancy_prediction.pkl')
rf_scaler      = load('models/scaler.pkl')
ohe            = load('models/ohe_area.pkl')
be             = load('models/be_establecimiento.pkl')
km_model       = load('models/kmeans_model.pkl')
km_scaler      = load('models/kmeans_scaler.pkl')

winsor_caps    = load_json('models/winsor_caps.json')
feature_cols   = load_json('models/feature_columns.json')
categories     = load_json('models/categories.json')
km_meta        = load_json('models/kmeans_meta.json')

KM_FEATURES = km_meta['features']

# ── Regression pipeline ───────────────────────────────────────────────────────
def preprocess_regression(form):
    raw = pd.DataFrame([{
        'PERIODO':                   int(form['PERIODO']),
        'TIPO_PERTENENCIA':          int(form['TIPO_PERTENENCIA']),
        'MES':                       int(form['MES']),
        'DIAS_CAMAS_OCUPADAS':       float(form['DIAS_CAMAS_OCUPADAS']),
        'DIAS_CAMAS_DISPONIBLES':    float(form['DIAS_CAMAS_DISPONIBLES']),
        'DIAS_ESTADA':               float(form['DIAS_ESTADA']),
        'NUMERO_EGRESOS':            float(form['NUMERO_EGRESOS']),
        'EGRESOS_FALLECIDOS':        float(form['EGRESOS_FALLECIDOS']),
        'TRASLADOS':                 float(form['TRASLADOS']),
        'PROMEDIO_CAMAS_DISPONIBLE': float(form['PROMEDIO_CAMAS_DISPONIBLE']),
        'PROMEDIO_DIAS_ESTADA':      float(form['PROMEDIO_DIAS_ESTADA']),
        'LETALIDAD':                 float(form['LETALIDAD']),
        'INDICE_ROTACION':           float(form['INDICE_ROTACION']),
        'AREA_FUNCIONAL':            form['AREA_FUNCIONAL'],
        'ESTABLECIMIENTO':           form['ESTABLECIMIENTO'],
    }])

    for col, cap in winsor_caps.items():
        if col in raw.columns:
            raw[col] = raw[col].clip(upper=cap)

    area_enc = ohe.transform(raw[['AREA_FUNCIONAL']])
    area_df  = pd.DataFrame(area_enc,
                             columns=ohe.get_feature_names_out(['AREA_FUNCIONAL']),
                             index=raw.index)
    estab_df = be.transform(raw[['ESTABLECIMIENTO']])

    final = pd.concat([raw.drop(columns=['AREA_FUNCIONAL', 'ESTABLECIMIENTO']),
                       area_df, estab_df], axis=1)
    final = final.reindex(columns=feature_cols, fill_value=0)
    return rf_scaler.transform(final)


def alert_level(value):
    if value < 70:   return ('Normal',          '#dcfce7', '#166534', '🟢')
    elif value < 85: return ('Alerta',           '#fef9c3', '#854d0e', '🟡')
    else:            return ('Crítico',          '#fee2e2', '#991b1b', '🔴')


# ── K-Means pipeline ──────────────────────────────────────────────────────────
def preprocess_kmeans(form):
    row = {f: float(form.get(f, 0)) for f in KM_FEATURES}
    # Winsorize the two extra columns not in original winsor_caps
    extra_caps = {'NUMERO_EGRESOS': 1700.0, 'PROMEDIO_CAMAS_DISPONIBLE': 800.0}
    for col, cap in {**winsor_caps, **extra_caps}.items():
        if col in row:
            row[col] = min(row[col], cap)
    X = pd.DataFrame([row])[KM_FEATURES]
    return km_scaler.transform(X)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
                                  categories=categories,
                                  km_meta=km_meta)


@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    try:
        X = preprocess_regression(request.form)
        pred = float(rf_model.predict(X)[0])
        label, bg, text, icon = alert_level(pred)
        return jsonify({'prediction': round(pred, 2),
                        'alert_level': label,
                        'bg_color': bg, 'text_color': text, 'icon': icon})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/cluster', methods=['POST'])
def predict_cluster():
    try:
        X = preprocess_kmeans(request.form)
        cluster = int(km_model.predict(X)[0])
        return jsonify({
            'cluster':      cluster,
            'name':         km_meta['cluster_names'][str(cluster)],
            'color':        km_meta['cluster_colors'][str(cluster)],
            'description':  km_meta.get('cluster_desc', {}).get(str(cluster), ''),
            'size':         km_meta['cluster_sizes'][str(cluster)],
            'centroid':     km_meta['centroids'][cluster],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ── HTML template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>REM 20 — Predictor y Clustering Hospitalario</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f1f5f9;
      color: #1e293b;
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
      color: white;
      padding: 20px 32px;
    }
    header h1 { font-size: 22px; font-weight: 700; }
    header p  { font-size: 13px; opacity: 0.8; margin-top: 4px; }

    /* ── Tabs ── */
    .tabs {
      display: flex;
      background: white;
      border-bottom: 2px solid #e2e8f0;
      padding: 0 32px;
    }
    .tab-btn {
      padding: 14px 24px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      color: #64748b;
      border: none;
      background: none;
      border-bottom: 3px solid transparent;
      margin-bottom: -2px;
      transition: all 0.2s;
    }
    .tab-btn:hover  { color: #2563eb; }
    .tab-btn.active { color: #2563eb; border-bottom-color: #2563eb; }

    /* ── Content ── */
    .tab-content { display: none; padding: 28px 32px; max-width: 1000px; }
    .tab-content.active { display: block; }

    /* ── Card ── */
    .card {
      background: white;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
      margin-bottom: 20px;
    }
    .card h2 {
      font-size: 15px; font-weight: 700; color: #1e3a5f;
      margin-bottom: 16px; padding-bottom: 8px;
      border-bottom: 1px solid #e2e8f0;
    }

    /* ── Form grid ── */
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .field { display: flex; flex-direction: column; gap: 5px; }
    .field.full { grid-column: 1 / -1; }
    label { font-size: 12px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: .03em; }
    input, select {
      padding: 9px 11px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 14px;
      background: #f8fafc;
      transition: border-color .15s;
    }
    input:focus, select:focus { outline: none; border-color: #2563eb; background: white; }

    .section-divider {
      grid-column: 1 / -1;
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      color: #94a3b8; letter-spacing: .06em;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 4px; margin-top: 6px;
    }

    /* ── Button ── */
    .btn {
      margin-top: 18px; width: 100%; padding: 13px;
      background: #2563eb; color: white;
      border: none; border-radius: 10px;
      font-size: 15px; font-weight: 700;
      cursor: pointer; transition: background .2s;
    }
    .btn:hover { background: #1d4ed8; }
    .btn.green { background: #16a34a; }
    .btn.green:hover { background: #15803d; }

    /* ── Result boxes ── */
    .result-box {
      display: none;
      margin-top: 20px; padding: 22px;
      border-radius: 12px; text-align: center;
      border: 1px solid rgba(0,0,0,.06);
    }
    .result-box .value  { font-size: 42px; font-weight: 800; margin-bottom: 4px; }
    .result-box .label  { font-size: 16px; font-weight: 600; }
    .result-box .detail { font-size: 13px; margin-top: 8px; opacity: .75; }

    /* ── Cluster badges ── */
    .cluster-badge {
      display: inline-block;
      padding: 3px 10px; border-radius: 20px;
      font-size: 12px; font-weight: 700;
      margin: 4px 2px;
    }

    /* ── Centroid table ── */
    .centroid-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }
    .centroid-table th { background: #f1f5f9; padding: 7px 10px; text-align: left; font-weight: 600; }
    .centroid-table td { padding: 6px 10px; border-top: 1px solid #f1f5f9; }

    /* ── Cluster legend ── */
    .legend { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 0; }
    .legend-item {
      padding: 12px 14px; border-radius: 10px;
      border: 1px solid rgba(0,0,0,.06);
      font-size: 13px;
    }
    .legend-item strong { display: block; margin-bottom: 3px; }

    .error-msg { color: #dc2626; font-size: 13px; margin-top: 10px; text-align: center; }
  </style>
</head>
<body>

<header>
  <h1>🏥 REM 20 — Análisis Predictivo Hospitalario</h1>
  <p>Ministerio de Salud de Chile · Minería de Datos BIY7121</p>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('regression', this)">
    📈 Predicción de Ocupación
  </button>
  <button class="tab-btn" onclick="switchTab('cluster', this)">
    🔵 Clasificación de Patrón Operativo
  </button>
</div>

<!-- ══════════════════════════════════════════════════════
     TAB 1 — REGRESSION
══════════════════════════════════════════════════════ -->
<div id="tab-regression" class="tab-content active">

  <div class="card">
    <h2>Predictor de Índice Ocupacional — Random Forest</h2>
    <div class="grid">

      <div class="section-divider">Identificación</div>

      <div class="field full">
        <label>Área Funcional</label>
        <select id="r_AREA_FUNCIONAL">
          {% for a in categories.AREA_FUNCIONAL %}
            <option value="{{ a }}">{{ a }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="field full">
        <label>Establecimiento</label>
        <select id="r_ESTABLECIMIENTO">
          {% for e in categories.ESTABLECIMIENTO %}
            <option value="{{ e }}">{{ e }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="field">
        <label>Período (Año)</label>
        <input type="number" id="r_PERIODO" value="2024" min="2014" max="2030">
      </div>

      <div class="field">
        <label>Mes</label>
        <select id="r_MES">
          {% for m in range(1,13) %}
            <option value="{{ m }}">{{ m }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="field">
        <label>Tipo Pertenencia</label>
        <select id="r_TIPO_PERTENENCIA">
          <option value="1">1 — Perteneciente SNSS</option>
          <option value="2">2 — No perteneciente</option>
        </select>
      </div>

      <div class="section-divider">Camas</div>

      <div class="field">
        <label>Días Camas Ocupadas</label>
        <input type="number" id="r_DIAS_CAMAS_OCUPADAS" value="620">
      </div>
      <div class="field">
        <label>Días Camas Disponibles</label>
        <input type="number" id="r_DIAS_CAMAS_DISPONIBLES" value="744">
      </div>
      <div class="field">
        <label>Promedio Camas Disponibles</label>
        <input type="number" step="0.1" id="r_PROMEDIO_CAMAS_DISPONIBLE" value="24">
      </div>

      <div class="section-divider">Pacientes</div>

      <div class="field">
        <label>Días de Estada</label>
        <input type="number" id="r_DIAS_ESTADA" value="640">
      </div>
      <div class="field">
        <label>Promedio Días de Estada</label>
        <input type="number" step="0.1" id="r_PROMEDIO_DIAS_ESTADA" value="8.5">
      </div>
      <div class="field">
        <label>Número de Egresos</label>
        <input type="number" id="r_NUMERO_EGRESOS" value="75">
      </div>
      <div class="field">
        <label>Egresos Fallecidos</label>
        <input type="number" id="r_EGRESOS_FALLECIDOS" value="2">
      </div>
      <div class="field">
        <label>Traslados</label>
        <input type="number" id="r_TRASLADOS" value="5">
      </div>
      <div class="field">
        <label>Letalidad (%)</label>
        <input type="number" step="0.01" id="r_LETALIDAD" value="2.67">
      </div>
      <div class="field">
        <label>Índice de Rotación</label>
        <input type="number" step="0.01" id="r_INDICE_ROTACION" value="3.1">
      </div>
    </div>

    <button class="btn" onclick="predictRegression()">Predecir Índice Ocupacional</button>
    <div class="error-msg" id="r_error"></div>
  </div>

  <div class="result-box" id="r_result">
    <div class="value" id="r_value"></div>
    <div class="label" id="r_label"></div>
    <div class="detail">Índice Ocupacional Predicho — Random Forest Regressor</div>
  </div>

</div><!-- /tab-regression -->


<!-- ══════════════════════════════════════════════════════
     TAB 2 — CLUSTERING
══════════════════════════════════════════════════════ -->
<div id="tab-cluster" class="tab-content">

  <div class="card">
    <h2>Clasificador de Patrón Operativo — K-Means (K=4)</h2>
    <div class="grid">

      <div class="section-divider">Indicadores Operativos</div>

      <div class="field">
        <label>Índice Ocupacional (%)</label>
        <input type="number" step="0.1" id="k_INDICE_OCUPACIONAL" value="77">
      </div>
      <div class="field">
        <label>Promedio Días de Estada</label>
        <input type="number" step="0.1" id="k_PROMEDIO_DIAS_ESTADA" value="7">
      </div>
      <div class="field">
        <label>Índice de Rotación</label>
        <input type="number" step="0.01" id="k_INDICE_ROTACION" value="3.2">
      </div>
      <div class="field">
        <label>Número de Egresos</label>
        <input type="number" id="k_NUMERO_EGRESOS" value="70">
      </div>
      <div class="field">
        <label>Promedio Camas Disponibles</label>
        <input type="number" step="0.1" id="k_PROMEDIO_CAMAS_DISPONIBLE" value="22">
      </div>
      <div class="field">
        <label>Letalidad (%)</label>
        <input type="number" step="0.01" id="k_LETALIDAD" value="2.5">
      </div>
      <div class="field">
        <label>Mes</label>
        <select id="k_MES">
          {% for m in range(1,13) %}
            <option value="{{ m }}">{{ m }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <button class="btn green" onclick="predictCluster()">Clasificar Patrón Operativo</button>
    <div class="error-msg" id="k_error"></div>
  </div>

  <div class="result-box" id="k_result">
    <div class="value" id="k_icon"></div>
    <div class="label" id="k_name"></div>
    <div class="detail" id="k_desc"></div>
    <table class="centroid-table" id="k_centroid_table">
      <thead>
        <tr>
          <th>Métrica</th>
          <th>Tu registro</th>
          <th>Centroide del cluster</th>
        </tr>
      </thead>
      <tbody id="k_centroid_body"></tbody>
    </table>
  </div>

  <!-- Cluster legend always visible -->
  <div class="card">
    <h2>Referencia de Clusters</h2>
    <div class="legend">
      {% for c in ['0','1','2','3'] %}
      <div class="legend-item" style="background:{{ km_meta.cluster_colors[c] }}22; border-color: {{ km_meta.cluster_colors[c] }}44">
        <strong style="color:{{ km_meta.cluster_colors[c] }}">
          Cluster {{ c }} — {{ km_meta.cluster_names[c] }}
        </strong>
        {{ km_meta.get('cluster_desc', {}).get(c, '') }}
        <br><small style="color:#64748b">{{ km_meta.cluster_sizes[c] }} registros en el dataset</small>
      </div>
      {% endfor %}
    </div>
    <p style="font-size:12px; color:#94a3b8; margin-top:12px;">
      Silhouette Score: <strong>{{ km_meta.silhouette_score }}</strong> &nbsp;·&nbsp;
      Davies-Bouldin: <strong>{{ km_meta.get('davies_bouldin', 'N/A') }}</strong> &nbsp;·&nbsp;
      Calinski-Harabasz: <strong>{{ km_meta.get('calinski_harabasz', 'N/A') }}</strong>
    </p>
  </div>

</div><!-- /tab-cluster -->


<script>
  // ── Tab switching ──────────────────────────────────────────
  function switchTab(name, btn) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }

  // ── Regression prediction ──────────────────────────────────
  async function predictRegression() {
    document.getElementById('r_error').textContent = '';
    document.getElementById('r_result').style.display = 'none';

    const fields = [
      'AREA_FUNCIONAL','ESTABLECIMIENTO','PERIODO','MES','TIPO_PERTENENCIA',
      'DIAS_CAMAS_OCUPADAS','DIAS_CAMAS_DISPONIBLES','DIAS_ESTADA',
      'NUMERO_EGRESOS','EGRESOS_FALLECIDOS','TRASLADOS',
      'PROMEDIO_CAMAS_DISPONIBLE','PROMEDIO_DIAS_ESTADA',
      'LETALIDAD','INDICE_ROTACION'
    ];
    const fd = new FormData();
    for (const f of fields) fd.append(f, document.getElementById('r_' + f).value);

    const res  = await fetch('/predict/regression', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) { document.getElementById('r_error').textContent = 'Error: ' + data.error; return; }

    const box = document.getElementById('r_result');
    box.style.display    = 'block';
    box.style.background = data.bg_color;
    box.style.color      = data.text_color;
    document.getElementById('r_value').textContent = data.icon + ' ' + data.prediction + '%';
    document.getElementById('r_label').textContent = 'Nivel de Alerta: ' + data.alert_level;
  }

  // ── Clustering prediction ──────────────────────────────────
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
    const data = await res.json();

    if (data.error) { document.getElementById('k_error').textContent = 'Error: ' + data.error; return; }

    const box = document.getElementById('k_result');
    box.style.display    = 'block';
    box.style.background = data.color + '22';
    box.style.borderColor = data.color + '55';
    box.style.color      = '#1e293b';

    document.getElementById('k_icon').textContent = 'Cluster ' + data.cluster;
    document.getElementById('k_icon').style.color = data.color;
    document.getElementById('k_name').textContent = data.name;
    document.getElementById('k_name').style.color = data.color;
    document.getElementById('k_desc').textContent = data.description;

    // Centroid comparison table
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
</script>

</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
