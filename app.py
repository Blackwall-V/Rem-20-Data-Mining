"""
app.py — REM 20 · Capa de rutas Flask
=====================================================
Este archivo SOLO contiene rutas HTTP. Toda la lógica de:
  - Carga y predicción de modelos       → ml_models.py
  - Lectura de CSV/Excel y export       → data_io.py
  - Presentación (HTML/CSS/JS)          → templates/ + static/

Así cada pieza se puede editar, testear o reemplazar sin tocar
las demás.

Run:
  pip install -r requirements-prod.txt
  python app.py
  → http://localhost:5000
"""

import uuid
from flask import Flask, request, render_template, jsonify, send_file

from ml_models import RemModels
from data_io import (
    read_uploaded_table, validate_columns, dataframe_to_rows,
    build_excel_report, build_pdf_report,
)

app = Flask(__name__)

# Carga todos los modelos UNA vez al iniciar el servidor
models = RemModels()

# Cache en memoria de los últimos lotes procesados, para que
# /batch/export pueda generar el archivo sin re-subir el CSV.
# token -> DataFrame de resultados
_batch_cache: dict[str, "object"] = {}
_BATCH_CACHE_MAX = 20  # evita crecimiento indefinido en sesiones largas


# ════════════════════════════════════════════════════════════
#  PÁGINA PRINCIPAL
# ════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template(
        'index.html',
        categories=models.categories,
        km_meta=models.km_meta,
    )


# ════════════════════════════════════════════════════════════
#  PREDICCIÓN INDIVIDUAL
# ════════════════════════════════════════════════════════════
@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    try:
        row = {f: request.form.get(f) for f in RemModels.REGRESSION_INPUT_FIELDS}
        result = models.predict_occupancy(row)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/cluster', methods=['POST'])
def predict_cluster():
    try:
        row = {f: request.form.get(f) for f in models.km_features}
        result = models.predict_cluster(row)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ════════════════════════════════════════════════════════════
#  IMPORTACIÓN MASIVA (CSV / Excel)
# ════════════════════════════════════════════════════════════
@app.route('/batch/upload', methods=['POST'])
def batch_upload():
    try:
        file = request.files.get('archivo')
        if file is None or file.filename == '':
            return jsonify({'error': 'No se recibió ningún archivo.'}), 400

        df_raw = read_uploaded_table(file)
        validation = validate_columns(df_raw)
        if not validation['valid']:
            return jsonify({
                'error': 'Faltan columnas requeridas: ' + ', '.join(validation['missing_columns'])
            }), 400

        rows = dataframe_to_rows(df_raw)
        df_results = models.predict_full_batch(rows)

        # Cachear resultados completos para exportación posterior
        token = uuid.uuid4().hex[:12]
        if len(_batch_cache) >= _BATCH_CACHE_MAX:
            _batch_cache.pop(next(iter(_batch_cache)))
        _batch_cache[token] = df_results

        valid = df_results[df_results['_error'].isna()] if '_error' in df_results.columns else df_results
        alert_counts = (
            valid['NIVEL_ALERTA'].value_counts().to_dict()
            if 'NIVEL_ALERTA' in valid.columns else {}
        )
        avg_occ = (
            round(float(valid['PREDICCION_OCUPACIONAL'].mean()), 2)
            if 'PREDICCION_OCUPACIONAL' in valid.columns and len(valid) else None
        )
        error_count = int(df_results['_error'].notna().sum()) if '_error' in df_results.columns else 0

        preview_cols = [c for c in df_results.columns if not c.startswith('_')]
        priority = ['ESTABLECIMIENTO', 'AREA_FUNCIONAL', 'PREDICCION_OCUPACIONAL', 'NIVEL_ALERTA', 'CLUSTER', 'PATRON_OPERATIVO']
        ordered_cols = [c for c in priority if c in preview_cols] + [c for c in preview_cols if c not in priority]
        preview = df_results[ordered_cols].head(200).fillna('').to_dict(orient='records')

        return jsonify({
            'token': token,
            'total_rows': len(df_results),
            'error_count': error_count,
            'alert_counts': alert_counts,
            'avg_occupancy': avg_occ,
            'preview': preview,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE REPORTES (Excel / PDF)
# ════════════════════════════════════════════════════════════
@app.route('/batch/export/<fmt>')
def batch_export(fmt):
    token = request.args.get('token')
    df_results = _batch_cache.get(token)
    if df_results is None:
        return jsonify({'error': 'Lote no encontrado. Vuelve a importar el archivo.'}), 404

    if fmt == 'xlsx':
        content = build_excel_report(df_results, models.km_meta)
        return send_file(
            io_bytes(content),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='reporte_rem20.xlsx',
        )
    elif fmt == 'pdf':
        content = build_pdf_report(df_results, models.km_meta)
        return send_file(
            io_bytes(content),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='reporte_rem20.pdf',
        )
    else:
        return jsonify({'error': 'Formato no soportado. Usa xlsx o pdf.'}), 400


def io_bytes(content: bytes):
    import io
    buf = io.BytesIO(content)
    buf.seek(0)
    return buf


if __name__ == '__main__':
    app.run(debug=True)
