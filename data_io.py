"""
data_io.py — Importación y exportación de datos
=====================================================
Toda la lógica de:
  - Leer un CSV/Excel subido por el usuario y convertirlo en filas
    que ml_models.py pueda consumir
  - Exportar resultados (predicciones de regresión + clustering)
    a un Excel formateado y a un PDF de reporte ejecutivo

Separado de app.py (rutas) y de ml_models.py (algoritmos) para que
cada pieza tenga una sola responsabilidad.
"""

import io
import pandas as pd
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, PieChart, Reference

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors as rl_colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# Columnas requeridas para que un registro pueda pasar por ambos modelos
REQUIRED_COLUMNS = [
    'PERIODO', 'MES', 'AREA_FUNCIONAL', 'ESTABLECIMIENTO',
    'DIAS_CAMAS_OCUPADAS', 'DIAS_CAMAS_DISPONIBLES', 'DIAS_ESTADA',
    'NUMERO_EGRESOS', 'PROMEDIO_CAMAS_DISPONIBLE', 'PROMEDIO_DIAS_ESTADA',
    'LETALIDAD', 'INDICE_ROTACION',
]


# ════════════════════════════════════════════════════════════
#  IMPORTACIÓN
# ════════════════════════════════════════════════════════════
def read_uploaded_table(file_storage) -> pd.DataFrame:
    """
    Lee un archivo subido (CSV o Excel) y devuelve un DataFrame crudo.
    file_storage: objeto de Flask request.files['archivo']
    """
    filename = file_storage.filename.lower()

    if filename.endswith('.csv'):
        raw_bytes = file_storage.read()
        # Intenta separador ; (formato chileno/REM20) y si falla, ,
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=';')
            if df.shape[1] == 1:  # no se separó bien, probablemente es coma
                df = pd.read_csv(io.BytesIO(raw_bytes), sep=',')
        except Exception:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=',')
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_storage)
    else:
        raise ValueError('Formato no soportado. Usa .csv, .xlsx o .xls')

    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame) -> dict:
    """Verifica que el archivo tenga las columnas mínimas necesarias."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return {
        'valid': len(missing) == 0,
        'missing_columns': missing,
        'found_columns': list(df.columns),
        'row_count': len(df),
    }


def dataframe_to_rows(df: pd.DataFrame) -> list[dict]:
    """Convierte el DataFrame importado en lista de dicts para ml_models."""
    df = df.fillna(0)
    return df.to_dict(orient='records')


# ════════════════════════════════════════════════════════════
#  EXPORTACIÓN — EXCEL
# ════════════════════════════════════════════════════════════
HEADER_FILL  = PatternFill('solid', start_color='0D3B3E', end_color='0D3B3E')
HEADER_FONT  = Font(color='FFFFFF', bold=True, size=10, name='Arial')
TITLE_FONT   = Font(color='0D3B3E', bold=True, size=16, name='Arial')
SUB_FONT     = Font(color='5C6B6C', size=10, italic=True, name='Arial')
BORDER_THIN  = Border(*(Side(style='thin', color='E8E3D8') for _ in range(4)))

ALERT_FILLS = {
    'Normal':  PatternFill('solid', start_color='ECFDF3', end_color='ECFDF3'),
    'Alerta':  PatternFill('solid', start_color='FFFBEB', end_color='FFFBEB'),
    'Crítico': PatternFill('solid', start_color='FEF2F2', end_color='FEF2F2'),
}
ALERT_FONTS = {
    'Normal':  Font(color='15803D', bold=True, name='Arial'),
    'Alerta':  Font(color='92610A', bold=True, name='Arial'),
    'Crítico': Font(color='B91C1C', bold=True, name='Arial'),
}


def build_excel_report(df_results: pd.DataFrame, km_meta: dict) -> bytes:
    """
    Construye un Excel de 3 hojas:
      1. Resumen ejecutivo (métricas agregadas + gráficos)
      2. Detalle de predicciones (tabla completa)
      3. Perfil de clusters (referencia)
    Devuelve los bytes del archivo, listos para enviar como descarga.
    """
    wb = Workbook()

    # ── Hoja 1: Resumen ejecutivo ──────────────────────────────
    ws = wb.active
    ws.title = 'Resumen Ejecutivo'

    ws['B2'] = 'REM 20 — Reporte de Apoyo a la Decisión'
    ws['B2'].font = TITLE_FONT
    ws['B3'] = f"Generado el {datetime.now().strftime('%d-%m-%Y %H:%M')} · {len(df_results)} registros analizados"
    ws['B3'].font = SUB_FONT
    ws.merge_cells('B2:F2')
    ws.merge_cells('B3:F3')

    valid = df_results[df_results['_error'].isna()] if '_error' in df_results.columns else df_results

    # KPIs
    kpi_row = 6
    kpis = [
        ('Registros Procesados', len(df_results)),
        ('Predicción Promedio (%)', round(valid['PREDICCION_OCUPACIONAL'].mean(), 2) if 'PREDICCION_OCUPACIONAL' in valid else 'N/A'),
        ('En Nivel Crítico', int((valid['NIVEL_ALERTA'] == 'Crítico').sum()) if 'NIVEL_ALERTA' in valid else 'N/A'),
        ('En Nivel Normal', int((valid['NIVEL_ALERTA'] == 'Normal').sum()) if 'NIVEL_ALERTA' in valid else 'N/A'),
    ]
    for i, (label, value) in enumerate(kpis):
        col = get_column_letter(2 + i)
        ws[f'{col}{kpi_row}'] = label
        ws[f'{col}{kpi_row}'].font = Font(bold=True, size=9, color='5C6B6C', name='Arial')
        ws[f'{col}{kpi_row+1}'] = value
        ws[f'{col}{kpi_row+1}'].font = Font(bold=True, size=18, color='0D3B3E', name='Arial')

    # Tabla de distribución por nivel de alerta (fuente del gráfico)
    chart_row = kpi_row + 4
    ws[f'B{chart_row}'] = 'Distribución por Nivel de Alerta'
    ws[f'B{chart_row}'].font = Font(bold=True, size=11, color='0D3B3E', name='Arial')

    if 'NIVEL_ALERTA' in valid.columns:
        alert_counts = valid['NIVEL_ALERTA'].value_counts()
        data_start = chart_row + 1
        ws[f'B{data_start}'] = 'Nivel'
        ws[f'C{data_start}'] = 'Cantidad'
        ws[f'B{data_start}'].font = HEADER_FONT
        ws[f'C{data_start}'].font = HEADER_FONT
        ws[f'B{data_start}'].fill = HEADER_FILL
        ws[f'C{data_start}'].fill = HEADER_FILL

        for i, level in enumerate(['Normal', 'Alerta', 'Crítico']):
            r = data_start + 1 + i
            ws[f'B{r}'] = level
            ws[f'C{r}'] = int(alert_counts.get(level, 0))

        chart = BarChart()
        chart.title = 'Registros por Nivel de Alerta'
        chart.y_axis.title = 'Cantidad'
        chart.style = 10
        data_ref = Reference(ws, min_col=3, min_row=data_start, max_row=data_start + 3)
        cats_ref = Reference(ws, min_col=2, min_row=data_start + 1, max_row=data_start + 3)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.width, chart.height = 14, 8
        ws.add_chart(chart, f'E{chart_row}')

    # Distribución por cluster
    if 'CLUSTER' in valid.columns:
        cl_row = data_start + 6 if 'NIVEL_ALERTA' in valid.columns else chart_row + 1
        ws[f'B{cl_row}'] = 'Distribución por Patrón Operativo (Cluster)'
        ws[f'B{cl_row}'].font = Font(bold=True, size=11, color='0D3B3E', name='Arial')

        cluster_counts = valid['CLUSTER'].value_counts().sort_index()
        cl_data_start = cl_row + 1
        ws[f'B{cl_data_start}'] = 'Cluster'
        ws[f'C{cl_data_start}'] = 'Cantidad'
        ws[f'B{cl_data_start}'].font = HEADER_FONT
        ws[f'C{cl_data_start}'].font = HEADER_FONT
        ws[f'B{cl_data_start}'].fill = HEADER_FILL
        ws[f'C{cl_data_start}'].fill = HEADER_FILL

        for i, (cluster_id, count) in enumerate(cluster_counts.items()):
            r = cl_data_start + 1 + i
            name = km_meta['cluster_names'].get(str(cluster_id), f'Cluster {cluster_id}')
            ws[f'B{r}'] = f"C{cluster_id} — {name}"
            ws[f'C{r}'] = int(count)

        pie = PieChart()
        pie.title = 'Distribución de Patrones Operativos'
        n = len(cluster_counts)
        data_ref = Reference(ws, min_col=3, min_row=cl_data_start, max_row=cl_data_start + n)
        cats_ref = Reference(ws, min_col=2, min_row=cl_data_start + 1, max_row=cl_data_start + n)
        pie.add_data(data_ref, titles_from_data=True)
        pie.set_categories(cats_ref)
        pie.width, pie.height = 14, 8
        ws.add_chart(pie, f'E{cl_row}')

    for col, width in zip('ABCDEF', [3, 24, 16, 14, 14, 14]):
        ws.column_dimensions[col].width = width

    # ── Hoja 2: Detalle de predicciones ────────────────────────
    ws2 = wb.create_sheet('Detalle de Predicciones')

    export_cols = [c for c in df_results.columns if not c.startswith('_')]
    # Reordenar para que las columnas de resultado queden al principio
    priority = ['PREDICCION_OCUPACIONAL', 'NIVEL_ALERTA', 'CLUSTER', 'PATRON_OPERATIVO']
    ordered = [c for c in priority if c in export_cols] + [c for c in export_cols if c not in priority]

    for j, col_name in enumerate(ordered, start=1):
        cell = ws2.cell(row=1, column=j, value=col_name)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center')

    for i, (_, row) in enumerate(df_results.iterrows(), start=2):
        for j, col_name in enumerate(ordered, start=1):
            val = row.get(col_name, '')
            cell = ws2.cell(row=i, column=j, value=val)
            cell.border = BORDER_THIN
            if col_name == 'NIVEL_ALERTA' and val in ALERT_FILLS:
                cell.fill = ALERT_FILLS[val]
                cell.font = ALERT_FONTS[val]

    for j, col_name in enumerate(ordered, start=1):
        ws2.column_dimensions[get_column_letter(j)].width = max(14, len(col_name) + 2)
    ws2.freeze_panes = 'A2'

    # ── Hoja 3: Perfil de clusters (referencia) ────────────────
    ws3 = wb.create_sheet('Perfil de Clusters')
    ws3['B2'] = 'Referencia de Patrones Operativos — K-Means (K=4)'
    ws3['B2'].font = Font(bold=True, size=13, color='0D3B3E', name='Arial')
    ws3.merge_cells('B2:F2')

    headers = ['Cluster', 'Nombre', 'Registros en Dataset', 'Descripción']
    for j, h in enumerate(headers, start=2):
        cell = ws3.cell(row=4, column=j, value=h)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL

    for i, c in enumerate(['0', '1', '2', '3']):
        r = 5 + i
        ws3.cell(row=r, column=2, value=f"Cluster {c}")
        ws3.cell(row=r, column=3, value=km_meta['cluster_names'].get(c, ''))
        ws3.cell(row=r, column=4, value=km_meta['cluster_sizes'].get(c, ''))
        ws3.cell(row=r, column=5, value=km_meta.get('cluster_desc', {}).get(c, ''))
        for col in range(2, 6):
            ws3.cell(row=r, column=col).border = BORDER_THIN

    ws3.column_dimensions['B'].width = 12
    ws3.column_dimensions['C'].width = 32
    ws3.column_dimensions['D'].width = 18
    ws3.column_dimensions['E'].width = 50

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════
#  EXPORTACIÓN — PDF
# ════════════════════════════════════════════════════════════
def build_pdf_report(df_results: pd.DataFrame, km_meta: dict) -> bytes:
    """
    Construye un PDF de reporte ejecutivo: portada con KPIs,
    distribución de niveles de alerta, distribución de clusters,
    y tabla resumen de los registros más críticos.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()

    ink = rl_colors.HexColor('#0D3B3E')
    teal = rl_colors.HexColor('#1A6B6F')
    steel = rl_colors.HexColor('#5C6B6C')

    title_style = ParagraphStyle('TitleClin', parent=styles['Title'], textColor=ink, fontSize=20, spaceAfter=4)
    sub_style = ParagraphStyle('SubClin', parent=styles['Normal'], textColor=steel, fontSize=10, fontName='Helvetica-Oblique')
    h2_style = ParagraphStyle('H2Clin', parent=styles['Heading2'], textColor=ink, fontSize=13, spaceBefore=14, spaceAfter=8)
    body_style = ParagraphStyle('BodyClin', parent=styles['Normal'], fontSize=9.5, leading=14)

    story = []
    story.append(Paragraph('Reporte de Apoyo a la Toma de Decisiones', title_style))
    story.append(Paragraph('Sistema REM 20 — Análisis Predictivo Hospitalario', sub_style))
    story.append(Paragraph(f"Generado el {datetime.now().strftime('%d-%m-%Y a las %H:%M')}", sub_style))
    story.append(Spacer(1, 16))

    valid = df_results[df_results['_error'].isna()] if '_error' in df_results.columns else df_results

    # ── KPIs ──
    story.append(Paragraph('Resumen General', h2_style))
    kpi_data = [['Métrica', 'Valor']]
    kpi_data.append(['Registros procesados', str(len(df_results))])
    if 'PREDICCION_OCUPACIONAL' in valid.columns and len(valid):
        kpi_data.append(['Índice ocupacional promedio', f"{valid['PREDICCION_OCUPACIONAL'].mean():.2f}%"])
        kpi_data.append(['Índice ocupacional máximo', f"{valid['PREDICCION_OCUPACIONAL'].max():.2f}%"])
        kpi_data.append(['Índice ocupacional mínimo', f"{valid['PREDICCION_OCUPACIONAL'].min():.2f}%"])
    if 'NIVEL_ALERTA' in valid.columns:
        kpi_data.append(['Registros en nivel Crítico', str(int((valid['NIVEL_ALERTA'] == 'Crítico').sum()))])
        kpi_data.append(['Registros en nivel Alerta', str(int((valid['NIVEL_ALERTA'] == 'Alerta').sum()))])
        kpi_data.append(['Registros en nivel Normal', str(int((valid['NIVEL_ALERTA'] == 'Normal').sum()))])

    t = Table(kpi_data, colWidths=[9 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ink),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#FAFAF7')]),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#E8E3D8')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t)

    # ── Distribución de clusters ──
    if 'CLUSTER' in valid.columns and len(valid):
        story.append(Paragraph('Distribución de Patrones Operativos', h2_style))
        cluster_counts = valid['CLUSTER'].value_counts().sort_index()
        cl_data = [['Cluster', 'Patrón', 'Registros', '% del Total']]
        for cid, count in cluster_counts.items():
            name = km_meta['cluster_names'].get(str(cid), f'Cluster {cid}')
            pct = count / len(valid) * 100
            cl_data.append([f"C{cid}", name, str(int(count)), f"{pct:.1f}%"])

        t2 = Table(cl_data, colWidths=[1.8 * cm, 7.5 * cm, 2.7 * cm, 3 * cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), teal),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#FAFAF7')]),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#E8E3D8')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t2)

    # ── Top registros críticos ──
    if 'NIVEL_ALERTA' in valid.columns:
        criticos = valid[valid['NIVEL_ALERTA'] == 'Crítico'].copy()
        if len(criticos):
            story.append(Paragraph(
                f'Registros en Nivel Crítico ({len(criticos)} de {len(valid)})', h2_style
            ))
            criticos = criticos.sort_values('PREDICCION_OCUPACIONAL', ascending=False).head(15)
            cols_show = [c for c in ['ESTABLECIMIENTO', 'AREA_FUNCIONAL', 'PREDICCION_OCUPACIONAL', 'CLUSTER'] if c in criticos.columns]
            crit_data = [cols_show]
            for _, row in criticos.iterrows():
                crit_data.append([str(row[c])[:35] for c in cols_show])

            col_widths = [5.5 * cm, 5.5 * cm, 2.5 * cm, 1.5 * cm][:len(cols_show)]
            t3 = Table(crit_data, colWidths=col_widths, repeatRows=1)
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#DC2626')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#FEF2F2')]),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#E8E3D8')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t3)
        else:
            story.append(Paragraph('No se encontraron registros en nivel Crítico.', body_style))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        'Reporte generado automáticamente por el sistema de apoyo a la decisión REM 20. '
        'Modelos: Random Forest (regresión de ocupación) y K-Means K=4 (clustering de patrones operativos). '
        'Proyecto de Minería de Datos — BIY7121.',
        ParagraphStyle('Footer', parent=body_style, fontSize=7.5, textColor=steel),
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
