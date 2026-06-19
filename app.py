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
  <title>REM 20 — Estación de Análisis Clínico</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      /* ── Paleta: quirófano + expediente clínico ── */
      --ink:          #0d3b3e;   /* verde quirófano oscuro — texto principal, header */
      --ink-soft:     #2d5558;
      --teal:         #1a6b6f;   /* verde médico medio — acentos, botones */
      --teal-light:   #5fa8a3;
      --paper:        #f7f5f0;   /* blanco hospital / papel de expediente */
      --paper-line:   #e8e3d8;   /* líneas de cuaderno clínico */
      --paper-card:   #ffffff;
      --steel:        #8a9a9b;   /* gris acero de equipo médico */
      --steel-dark:   #5c6b6c;

      /* Triage — semántica real de urgencias */
      --triage-green:  #16a34a;
      --triage-amber:  #d97706;
      --triage-red:    #dc2626;
      --triage-green-bg: #ecfdf3;
      --triage-amber-bg: #fffbeb;
      --triage-red-bg:   #fef2f2;

      --font-display: 'Inter', sans-serif;
      --font-mono:    'JetBrains Mono', monospace;
    }

    * , *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font-display);
      background: var(--paper);
      color: var(--ink);
      min-height: 100vh;
      background-image:
        linear-gradient(var(--paper-line) 1px, transparent 1px);
      background-size: 100% 32px;
      background-attachment: local;
    }

    /* ══════════════════════════════════════════
       HEADER — placa de identificación + monitor ECG
    ══════════════════════════════════════════ */
    header {
      background: var(--ink);
      color: var(--paper);
      padding: 0;
      position: relative;
      overflow: hidden;
      border-bottom: 4px solid var(--teal);
    }
    .header-inner {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px 28px 18px;
      position: relative;
      z-index: 2;
    }
    .header-top {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }
    .header-id {
      font-family: var(--font-mono);
      font-size: 11px;
      letter-spacing: .12em;
      color: var(--teal-light);
      text-transform: uppercase;
    }
    header h1 {
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -.01em;
      margin-top: 4px;
    }
    header h1 .rx { color: var(--teal-light); font-weight: 500; }
    header p.sub {
      font-size: 12.5px;
      color: var(--steel);
      margin-top: 3px;
      font-family: var(--font-mono);
    }

    /* ECG line — animated signature element */
    .ecg-wrap {
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 34px;
      z-index: 1;
      opacity: .55;
    }
    .ecg-wrap svg { width: 200%; height: 100%; }
    .ecg-line {
      fill: none;
      stroke: var(--teal-light);
      stroke-width: 1.6;
      stroke-linecap: round;
      stroke-linejoin: round;
      animation: ecg-scroll 7s linear infinite;
    }
    @keyframes ecg-scroll {
      from { transform: translateX(0); }
      to   { transform: translateX(-50%); }
    }
    @media (prefers-reduced-motion: reduce) {
      .ecg-line { animation: none; }
    }

    /* ══════════════════════════════════════════
       LAYOUT — panel lateral tipo admisión + contenido
    ══════════════════════════════════════════ */
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 0;
      min-height: calc(100vh - 110px);
    }

    nav.ward {
      border-right: 1px solid var(--paper-line);
      padding: 28px 0;
    }
    .ward-label {
      font-family: var(--font-mono);
      font-size: 10px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--steel-dark);
      padding: 0 24px 12px;
    }
    .ward-btn {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      text-align: left;
      padding: 13px 24px;
      border: none;
      background: none;
      cursor: pointer;
      font-family: var(--font-display);
      font-size: 13.5px;
      font-weight: 600;
      color: var(--steel-dark);
      border-left: 3px solid transparent;
      transition: all .15s;
    }
    .ward-btn .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--steel);
      flex-shrink: 0;
      transition: all .15s;
    }
    .ward-btn:hover { background: var(--paper-card); color: var(--ink); }
    .ward-btn.active {
      background: var(--paper-card);
      color: var(--ink);
      border-left-color: var(--teal);
    }
    .ward-btn.active .dot { background: var(--teal); box-shadow: 0 0 0 3px rgba(26,107,111,.15); }

    .ward-meta {
      padding: 18px 24px 0;
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--steel);
      line-height: 1.7;
      border-top: 1px solid var(--paper-line);
      margin-top: 16px;
    }
    .ward-meta strong { color: var(--ink-soft); }

    main { padding: 28px 32px 60px; }
    .panel { display: none; }
    .panel.active { display: block; animation: fade-in .25s ease; }
    @keyframes fade-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

    /* ══════════════════════════════════════════
       CARDS — expediente clínico
    ══════════════════════════════════════════ */
    .chart {
      background: var(--paper-card);
      border: 1px solid var(--paper-line);
      border-radius: 4px;
      margin-bottom: 22px;
      box-shadow: 0 1px 2px rgba(13,59,62,.04);
    }
    .chart-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 20px;
      border-bottom: 1px solid var(--paper-line);
      background: linear-gradient(180deg, #fafaf7, var(--paper-card));
    }
    .chart-head h2 {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: var(--ink);
    }
    .chart-head .tag {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--teal);
      background: rgba(26,107,111,.08);
      padding: 3px 8px;
      border-radius: 3px;
      letter-spacing: .03em;
    }
    .chart-body { padding: 20px; }

    /* ── Form grid ── */
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 16px; }
    .field { display: flex; flex-direction: column; gap: 5px; }
    .field.full { grid-column: 1 / -1; }
    label {
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 600;
      color: var(--steel-dark);
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    input, select {
      padding: 9px 11px;
      border: 1px solid var(--paper-line);
      border-radius: 3px;
      font-size: 13.5px;
      font-family: var(--font-display);
      background: #fbfaf7;
      color: var(--ink);
      transition: border-color .15s, background .15s;
    }
    input:focus, select:focus {
      outline: none;
      border-color: var(--teal);
      background: white;
      box-shadow: 0 0 0 3px rgba(26,107,111,.1);
    }

    .section-divider {
      grid-column: 1 / -1;
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--teal);
      letter-spacing: .08em;
      padding-bottom: 5px;
      margin-top: 8px;
      border-bottom: 1px dashed var(--paper-line);
      display: flex; align-items: center; gap: 6px;
    }
    .section-divider::before { content: '+'; font-size: 12px; }

    /* ── Button ── */
    .btn {
      margin-top: 18px;
      width: 100%;
      padding: 13px;
      background: var(--ink);
      color: var(--paper);
      border: none;
      border-radius: 3px;
      font-family: var(--font-mono);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
      cursor: pointer;
      transition: background .15s;
    }
    .btn:hover { background: var(--teal); }
    .btn::before { content: '▸ '; }

    /* ══════════════════════════════════════════
       RESULTADOS — monitor de signos vitales
    ══════════════════════════════════════════ */
    .vitals-box {
      display: none;
      margin-top: 18px;
      border-radius: 4px;
      border: 1px solid;
      overflow: hidden;
    }
    .vitals-head {
      padding: 9px 18px;
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
      display: flex; align-items: center; gap: 8px;
    }
    .vitals-head .pulse {
      width: 8px; height: 8px; border-radius: 50%;
      animation: pulse 1.4s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
    .vitals-body { padding: 22px; text-align: center; background: white; }
    .vitals-body .value {
      font-family: var(--font-mono);
      font-size: 46px;
      font-weight: 800;
      line-height: 1;
      letter-spacing: -.02em;
    }
    .vitals-body .label { font-size: 15px; font-weight: 700; margin-top: 6px; }
    .vitals-body .detail {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--steel-dark);
      margin-top: 10px;
    }

    /* ── Cluster centroid table ── */
    .centroid-table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 16px; text-align: left; }
    .centroid-table th {
      font-family: var(--font-mono);
      background: #fafaf7;
      padding: 8px 12px;
      font-weight: 600;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .03em;
      color: var(--steel-dark);
      border-bottom: 1px solid var(--paper-line);
    }
    .centroid-table td {
      padding: 7px 12px;
      border-bottom: 1px solid var(--paper-line);
      font-family: var(--font-mono);
    }
    .centroid-table tr:last-child td { border-bottom: none; }

    /* ── Cluster legend ── */
    .legend { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .legend-item {
      padding: 14px 16px;
      border-radius: 3px;
      border: 1px solid var(--paper-line);
      border-left-width: 4px;
      font-size: 12.5px;
      background: #fafaf7;
    }
    .legend-item strong {
      display: block; margin-bottom: 4px; font-size: 13px;
    }
    .legend-item .count {
      display: block; margin-top: 6px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--steel-dark);
    }

    .metric-strip {
      display: flex; gap: 22px; flex-wrap: wrap;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--steel-dark);
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px dashed var(--paper-line);
    }
    .metric-strip strong { color: var(--ink); }

    .error-msg {
      color: var(--triage-red);
      font-family: var(--font-mono);
      font-size: 12px;
      margin-top: 10px;
      text-align: center;
    }

    @media (max-width: 760px) {
      .shell { grid-template-columns: 1fr; }
      nav.ward { display: flex; overflow-x: auto; padding: 14px 16px; border-right: none; border-bottom: 1px solid var(--paper-line); gap: 4px; }
      .ward-label, .ward-meta { display: none; }
      .ward-btn { white-space: nowrap; border-left: none; border-bottom: 3px solid transparent; padding: 10px 16px; }
      .ward-btn.active { border-left: none; border-bottom-color: var(--teal); }
      .grid { grid-template-columns: 1fr; }
      main { padding: 20px 16px 40px; }
    }
  </style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="header-top">
      <span class="header-id">FICHA N° REM-20 · SISTEMA DE APOYO A LA DECISIÓN</span>
    </div>
    <h1>Estación de Análisis <span class="rx">℞</span> Hospitalario</h1>
    <p class="sub">MINISTERIO DE SALUD DE CHILE — RED ASISTENCIAL · MINERÍA DE DATOS BIY7121</p>
  </div>
  <div class="ecg-wrap">
    <svg viewBox="0 0 1000 60" preserveAspectRatio="none">
      <path class="ecg-line" d="M0,30 L60,30 L75,30 L85,10 L95,50 L105,30 L160,30 L175,30 L185,15 L195,45 L205,30 L260,30
                                 L500,30 L560,30 L575,30 L585,10 L595,50 L605,30 L660,30 L675,30 L685,15 L695,45 L705,30 L760,30
                                 L1000,30" />
    </svg>
  </div>
</header>

<div class="shell">

  <nav class="ward">
    <div class="ward-label">Módulos</div>
    <button class="ward-btn active" onclick="switchTab('regression', this)">
      <span class="dot"></span> Predicción de Ocupación
    </button>
    <button class="ward-btn" onclick="switchTab('cluster', this)">
      <span class="dot"></span> Patrón Operativo
    </button>
    <div class="ward-meta">
      MODELO 1<br><strong>Random Forest</strong><br>Regresión continua<br><br>
      MODELO 2<br><strong>K-Means (K=4)</strong><br>Clustering no supervisado
    </div>
  </nav>

  <main>

    <!-- ══════════════ PANEL 1 — REGRESIÓN ══════════════ -->
    <div id="panel-regression" class="panel active">

      <div class="chart">
        <div class="chart-head">
          <h2>Predictor de Índice Ocupacional</h2>
          <span class="tag">RANDOM FOREST</span>
        </div>
        <div class="chart-body">
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

          <button class="btn" onclick="predictRegression()">Ejecutar Predicción</button>
          <div class="error-msg" id="r_error"></div>

          <div class="vitals-box" id="r_result">
            <div class="vitals-head" id="r_vitals_head">
              <span class="pulse" id="r_pulse"></span>
              <span id="r_status_text">RESULTADO DEL ANÁLISIS</span>
            </div>
            <div class="vitals-body">
              <div class="value" id="r_value"></div>
              <div class="label" id="r_label"></div>
              <div class="detail">ÍNDICE OCUPACIONAL PREDICHO · RANDOM FOREST REGRESSOR</div>
            </div>
          </div>
        </div>
      </div>

    </div><!-- /panel-regression -->


    <!-- ══════════════ PANEL 2 — CLUSTERING ══════════════ -->
    <div id="panel-cluster" class="panel">

      <div class="chart">
        <div class="chart-head">
          <h2>Clasificador de Patrón Operativo</h2>
          <span class="tag">K-MEANS · K=4</span>
        </div>
        <div class="chart-body">
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

          <button class="btn" onclick="predictCluster()">Clasificar Patrón</button>
          <div class="error-msg" id="k_error"></div>

          <div class="vitals-box" id="k_result">
            <div class="vitals-head" id="k_vitals_head">
              <span class="pulse" id="k_pulse"></span>
              <span id="k_status_text">CLASIFICACIÓN ASIGNADA</span>
            </div>
            <div class="vitals-body">
              <div class="value" id="k_icon" style="font-size: 28px;"></div>
              <div class="label" id="k_name"></div>
              <div class="detail" id="k_desc" style="text-transform:none; font-family: var(--font-display); font-size: 13px; margin-top: 8px;"></div>
              <table class="centroid-table" id="k_centroid_table" style="display:none;">
                <thead>
                  <tr><th>Métrica</th><th>Tu registro</th><th>Centroide</th></tr>
                </thead>
                <tbody id="k_centroid_body"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div class="chart">
        <div class="chart-head">
          <h2>Referencia de Clusters</h2>
          <span class="tag">PERFILES</span>
        </div>
        <div class="chart-body">
          <div class="legend">
            {% for c in ['0','1','2','3'] %}
            <div class="legend-item" style="border-left-color: {{ km_meta.cluster_colors[c] }}">
              <strong style="color:{{ km_meta.cluster_colors[c] }}">
                Cluster {{ c }} — {{ km_meta.cluster_names[c] }}
              </strong>
              {{ km_meta.get('cluster_desc', {}).get(c, '') }}
              <span class="count">{{ km_meta.cluster_sizes[c] }} REGISTROS EN EL DATASET</span>
            </div>
            {% endfor %}
          </div>
          <div class="metric-strip">
            <span>SILHOUETTE: <strong>{{ km_meta.silhouette_score }}</strong></span>
            <span>DAVIES-BOULDIN: <strong>{{ km_meta.get('davies_bouldin', 'N/A') }}</strong></span>
            <span>CALINSKI-HARABASZ: <strong>{{ km_meta.get('calinski_harabasz', 'N/A') }}</strong></span>
          </div>
        </div>
      </div>

    </div><!-- /panel-cluster -->

  </main>
</div>


<script>
  function switchTab(name, btn) {
    document.querySelectorAll('.panel').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.ward-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
    btn.classList.add('active');
  }

  // ── Triage color mapping (consistent with backend alert_level) ──
  const TRIAGE = {
    'Normal':   { bg: '#ecfdf3', border: '#16a34a', text: '#15803d' },
    'Alerta':   { bg: '#fffbeb', border: '#d97706', text: '#92610a' },
    'Crítico':  { bg: '#fef2f2', border: '#dc2626', text: '#b91c1c' },
  };

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

    if (data.error) { document.getElementById('r_error').textContent = 'ERROR: ' + data.error; return; }

    const t = TRIAGE[data.alert_level] || TRIAGE['Normal'];
    const box = document.getElementById('r_result');
    box.style.display = 'block';
    box.style.borderColor = t.border;

    const head = document.getElementById('r_vitals_head');
    head.style.background = t.bg;
    head.style.color = t.text;
    document.getElementById('r_pulse').style.background = t.border;
    document.getElementById('r_status_text').textContent = 'NIVEL DE ALERTA: ' + data.alert_level.toUpperCase();

    document.getElementById('r_value').style.color = t.border;
    document.getElementById('r_value').textContent = data.prediction + '%';
    document.getElementById('r_label').textContent = data.icon + ' ' + data.alert_level;
    document.getElementById('r_label').style.color = t.text;
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
    const data = await res.json();

    if (data.error) { document.getElementById('k_error').textContent = 'ERROR: ' + data.error; return; }

    const box = document.getElementById('k_result');
    box.style.display = 'block';
    box.style.borderColor = data.color;

    const head = document.getElementById('k_vitals_head');
    head.style.background = data.color + '18';
    head.style.color = data.color;
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
</script>

</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
