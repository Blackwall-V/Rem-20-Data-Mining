import os
import uuid
from flask import Flask, request, render_template, jsonify, send_file

from ml_models import RemModels
from data_io import (
    read_uploaded_table, validate_columns, dataframe_to_rows,
    build_excel_report, build_pdf_report,
)

app = Flask(__name__)
models = RemModels()


_batch_cache: dict[str, "object"] = {}
_BATCH_CACHE_MAX = 20

@app.route('/')
def index():
    return render_template(
        'index.html',
        km_meta=models.km_meta,
    )


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
        df_results = models.predict_cluster_batch(rows)

        token = uuid.uuid4().hex[:12]
        if len(_batch_cache) >= _BATCH_CACHE_MAX:
            _batch_cache.pop(next(iter(_batch_cache)))
        _batch_cache[token] = df_results

        valid = df_results[df_results['_error'].isna()] if '_error' in df_results.columns else df_results
        error_count = int(df_results['_error'].notna().sum()) if '_error' in df_results.columns else 0

        preview_cols = [c for c in df_results.columns if not c.startswith('_')]
        priority = ['CLUSTER', 'PATRON_OPERATIVO']
        ordered_cols = [c for c in priority if c in preview_cols] + [c for c in preview_cols if c not in priority]
        preview = df_results[ordered_cols].head(200).fillna('').to_dict(orient='records')

        return jsonify({
            'token': token,
            'total_rows': len(df_results),
            'error_count': error_count,
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
