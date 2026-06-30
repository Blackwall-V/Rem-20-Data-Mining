"""
ml_models.py — Capa de modelos de Minería de Datos
=====================================================
Encapsula la carga de artefactos .pkl/.json del K-Means
y las funciones de preprocesamiento + predicción.
No conoce nada de Flask, HTTP ni HTML — solo recibe
diccionarios/listas de Python y devuelve resultados.
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

    def __init__(self):
        self.km_model  = _load_pickle('kmeans_model.pkl')
        self.km_scaler = _load_pickle('kmeans_scaler.pkl')
        self.km_meta   = _load_json('kmeans_meta.json')
        self.km_features = self.km_meta['features']
        self.winsor_caps = _load_json('winsor_caps.json')

        self._km_extra_caps = {
            'NUMERO_EGRESOS': 1700.0,
            'PROMEDIO_CAMAS_DISPONIBLE': 800.0,
        }

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
                res.pop('centroid', None)
                results.append({**row, **res, '_row': i, '_error': None})
            except Exception as e:
                results.append({**row, '_row': i, '_error': str(e)})
        return pd.DataFrame(results)
