"""
ml_models.py — Capa de modelos de Minería de Datos
=====================================================
Encapsula TODA la lógica de carga de artefactos .pkl/.json y las
funciones de preprocesamiento + predicción. No conoce nada de Flask,
HTTP, ni HTML — solo recibe diccionarios/listas de Python y devuelve
diccionarios de resultado.

Esto permite:
  - Reusar esta lógica en un script de batch, un notebook, un test, etc.
  - Probar las predicciones sin levantar un servidor web
  - Que app.py (rutas) y este archivo (algoritmos) evolucionen por separado
"""

import pickle
import json
import os
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _path(filename):
    return os.path.join(MODELS_DIR, filename)


def _load_pickle(filename):
    with open(_path(filename), 'rb') as f:
        return pickle.load(f)


def _load_json(filename):
    with open(_path(filename), encoding='utf-8') as f:
        return json.load(f)


class RemModels:
    """
    Carga todos los artefactos entrenados una sola vez y expone
    métodos de predicción de alto nivel.
    """

    # Columnas crudas que el formulario de regresión debe enviar
    REGRESSION_INPUT_FIELDS = [
        'PERIODO', 'TIPO_PERTENENCIA', 'MES',
        'DIAS_CAMAS_OCUPADAS', 'DIAS_CAMAS_DISPONIBLES', 'DIAS_ESTADA',
        'NUMERO_EGRESOS', 'EGRESOS_FALLECIDOS', 'TRASLADOS',
        'PROMEDIO_CAMAS_DISPONIBLE', 'PROMEDIO_DIAS_ESTADA',
        'LETALIDAD', 'INDICE_ROTACION', 'AREA_FUNCIONAL', 'ESTABLECIMIENTO',
    ]

    def __init__(self):
        # ── Modelo de regresión y su pipeline de preprocesamiento ──
        self.rf_model     = _load_pickle('rem20_occupancy_prediction.pkl')
        self.rf_scaler    = _load_pickle('scaler.pkl')
        self.ohe          = _load_pickle('ohe_area.pkl')
        self.be           = _load_pickle('be_establecimiento.pkl')
        self.winsor_caps  = _load_json('winsor_caps.json')
        self.feature_cols = _load_json('feature_columns.json')
        self.categories   = _load_json('categories.json')

        # ── Modelo de clustering K-Means ──
        self.km_model  = _load_pickle('kmeans_model.pkl')
        self.km_scaler = _load_pickle('kmeans_scaler.pkl')
        self.km_meta   = _load_json('kmeans_meta.json')
        self.km_features = self.km_meta['features']

        # Caps adicionales para clustering (no estaban en winsor_caps original)
        self._km_extra_caps = {
            'NUMERO_EGRESOS': 1700.0,
            'PROMEDIO_CAMAS_DISPONIBLE': 800.0,
        }

    # ──────────────────────────────────────────────────────────
    #  REGRESIÓN — Índice Ocupacional (Random Forest)
    # ──────────────────────────────────────────────────────────
    def _preprocess_regression(self, row: dict) -> np.ndarray:
        """Replica el pipeline del notebook: winsorize -> encode -> scale."""
        dias_ocupadas    = float(row['DIAS_CAMAS_OCUPADAS'])
        dias_disponibles = float(row['DIAS_CAMAS_DISPONIBLES'])

        # INDICE_OCUPACIONAL fue incluido como feature en el notebook.
        # Como no es un input del formulario (es lo que se predice), lo
        # reconstruimos desde los campos disponibles para que el modelo
        # no reciba siempre 0 en esa columna.
        indice_ocup = (dias_ocupadas / dias_disponibles * 100) if dias_disponibles > 0 else 0.0
        indice_ocup = min(indice_ocup, 100.0)

        raw = pd.DataFrame([{
            'PERIODO':                   float(row['PERIODO']),
            'TIPO_PERTENENCIA':          float(row['TIPO_PERTENENCIA']),
            'MES':                       float(row['MES']),
            'DIAS_CAMAS_OCUPADAS':       dias_ocupadas,
            'DIAS_CAMAS_DISPONIBLES':    dias_disponibles,
            'DIAS_ESTADA':               float(row['DIAS_ESTADA']),
            'NUMERO_EGRESOS':            float(row['NUMERO_EGRESOS']),
            'EGRESOS_FALLECIDOS':        float(row['EGRESOS_FALLECIDOS']),
            'TRASLADOS':                 float(row['TRASLADOS']),
            'PROMEDIO_CAMAS_DISPONIBLE': float(row['PROMEDIO_CAMAS_DISPONIBLE']),
            'PROMEDIO_DIAS_ESTADA':      float(row['PROMEDIO_DIAS_ESTADA']),
            'LETALIDAD':                 float(row['LETALIDAD']),
            'INDICE_ROTACION':           float(row['INDICE_ROTACION']),
            'INDICE_OCUPACIONAL':        indice_ocup,
            'AREA_FUNCIONAL':            str(row['AREA_FUNCIONAL']),
            'ESTABLECIMIENTO':           str(row['ESTABLECIMIENTO']),
        }])

        for col, cap in self.winsor_caps.items():
            if col in raw.columns:
                raw[col] = raw[col].clip(upper=cap)

        area_enc = self.ohe.transform(raw[['AREA_FUNCIONAL']])
        area_df  = pd.DataFrame(
            area_enc,
            columns=self.ohe.get_feature_names_out(['AREA_FUNCIONAL']),
            index=raw.index,
        )
        estab_df = self.be.transform(raw[['ESTABLECIMIENTO']])

        final = pd.concat(
            [raw.drop(columns=['AREA_FUNCIONAL', 'ESTABLECIMIENTO']), area_df, estab_df],
            axis=1,
        )
        final = final.reindex(columns=self.feature_cols, fill_value=0)
        return self.rf_scaler.transform(final)

    @staticmethod
    def alert_level(value: float) -> dict:
        """Clasifica un índice ocupacional en nivel de alerta (triage)."""
        if value < 70:
            return {'label': 'Normal',  'bg': '#ecfdf3', 'text': '#15803d', 'border': '#16a34a', 'icon': '🟢'}
        elif value < 85:
            return {'label': 'Alerta',  'bg': '#fffbeb', 'text': '#92610a', 'border': '#d97706', 'icon': '🟡'}
        else:
            return {'label': 'Crítico', 'bg': '#fef2f2', 'text': '#b91c1c', 'border': '#dc2626', 'icon': '🔴'}

    def predict_occupancy(self, row: dict) -> dict:
        """Predice el índice ocupacional para UN registro (dict de inputs)."""
        X = self._preprocess_regression(row)
        pred = float(self.rf_model.predict(X)[0])
        alert = self.alert_level(pred)
        return {
            'prediction': round(pred, 2),
            'alert_level': alert['label'],
            'bg_color': alert['bg'],
            'text_color': alert['text'],
            'border_color': alert['border'],
            'icon': alert['icon'],
        }

    def predict_occupancy_batch(self, rows: list[dict]) -> pd.DataFrame:
        """Predice el índice ocupacional para una lista de registros (CSV/Excel importado)."""
        results = []
        for i, row in enumerate(rows):
            try:
                res = self.predict_occupancy(row)
                results.append({**row, **res, '_row': i, '_error': None})
            except Exception as e:
                results.append({**row, '_row': i, '_error': str(e)})
        return pd.DataFrame(results)

    # ──────────────────────────────────────────────────────────
    #  CLUSTERING — Patrón Operativo (K-Means)
    # ──────────────────────────────────────────────────────────
    def _preprocess_cluster(self, row: dict) -> np.ndarray:
        values = {f: float(row.get(f, 0)) for f in self.km_features}
        caps = {**self.winsor_caps, **self._km_extra_caps}
        for col, cap in caps.items():
            if col in values:
                values[col] = min(values[col], cap)
        X = pd.DataFrame([values])[self.km_features]
        return self.km_scaler.transform(X)

    def predict_cluster(self, row: dict) -> dict:
        """Clasifica UN registro en uno de los 4 clusters K-Means."""
        X = self._preprocess_cluster(row)
        cluster = int(self.km_model.predict(X)[0])
        c = str(cluster)
        return {
            'cluster': cluster,
            'name': self.km_meta['cluster_names'][c],
            'color': self.km_meta['cluster_colors'][c],
            'description': self.km_meta.get('cluster_desc', {}).get(c, ''),
            'size': self.km_meta['cluster_sizes'][c],
            'centroid': self.km_meta['centroids'][cluster],
        }

    def predict_cluster_batch(self, rows: list[dict]) -> pd.DataFrame:
        """Clasifica una lista de registros en sus clusters correspondientes."""
        results = []
        for i, row in enumerate(rows):
            try:
                res = self.predict_cluster(row)
                res.pop('centroid', None)  # no aporta valor por fila en tabla batch
                results.append({**row, **res, '_row': i, '_error': None})
            except Exception as e:
                results.append({**row, '_row': i, '_error': str(e)})
        return pd.DataFrame(results)

    # ──────────────────────────────────────────────────────────
    #  COMBINADO — regresión + clustering en un solo paso
    #  (usado por el flujo de importación de CSV/Excel)
    # ──────────────────────────────────────────────────────────
    def predict_full_batch(self, rows: list[dict]) -> pd.DataFrame:
        """
        Corre AMBOS modelos sobre cada fila importada y devuelve un único
        DataFrame con: datos originales + predicción de ocupación +
        nivel de alerta + cluster asignado. Es la base del reporte
        exportable a Excel/PDF para apoyo a la toma de decisiones.
        """
        rows_with_defaults = []
        for row in rows:
            r = dict(row)
            r.setdefault('TRASLADOS', 0)
            r.setdefault('EGRESOS_FALLECIDOS', 0)
            r.setdefault('TIPO_PERTENENCIA', 1)
            rows_with_defaults.append(r)

        out_rows = []
        for i, row in enumerate(rows_with_defaults):
            record = dict(row)
            record['_row'] = i
            record['_error'] = None
            try:
                reg = self.predict_occupancy(row)
                record['PREDICCION_OCUPACIONAL'] = reg['prediction']
                record['NIVEL_ALERTA'] = reg['alert_level']
            except Exception as e:
                record['_error'] = f"Regresión: {e}"

            try:
                clu = self.predict_cluster(row)
                record['CLUSTER'] = clu['cluster']
                record['PATRON_OPERATIVO'] = clu['name']
            except Exception as e:
                prev = record.get('_error')
                record['_error'] = (prev + ' | ' if prev else '') + f"Clustering: {e}"

            out_rows.append(record)

        return pd.DataFrame(out_rows)
