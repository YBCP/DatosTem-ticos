import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import os
import random
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Seguimiento de Datos Temáticos", layout="wide")

# Título del dashboard
st.title("Dashboard de Datos Temáticos")

# Cargar datos con mejor manejo de errores
@st.cache_data
def load_data():
    try:
        # Verificar si el archivo existe
        filename = "Prueba seguimiento.csv"
        if not os.path.exists(filename):
            st.error(f"Archivo no encontrado: {filename}")
            return pd.DataFrame()
        
        # Intentar cargar el archivo
        try:
            df = pd.read_csv(filename, sep=";", encoding="utf-8")
            st.success(f"Datos cargados correctamente. {len(df)} filas encontradas.")
            return df
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {str(e)}")
            
            # Intentar con diferentes codificaciones
            for encoding in ["latin1", "ISO-8859-1", "cp1252"]:
                try:
                    df = pd.read_csv(filename, sep=";", encoding=encoding)
                    st.success(f"Datos cargados con codificación {encoding}. {len(df)} filas encontradas.")
                    return df
                except:
                    pass
                    
            # Intentar con diferentes separadores
            for sep in [",", "\t"]:
                try:
                    df = pd.read_csv(filename, sep=sep, encoding="utf-8")
                    st.success(f"Datos cargados con separador '{sep}'. {len(df)} filas encontradas.")
                    return df
                except:
                    pass
                    
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error desconocido: {str(e)}")
        return pd.DataFrame()

# Función para descargar datos
def download_dataframe(df):
    csv = df.to_csv(sep=";", index=False, encoding="utf-8")
    return csv

# Función para convertir fechas en formato correcto
def parse_date(date_str):
    if pd.isna(date_str) or date_str == "" or date_str == "-":
        return None
    
    try:
        # Intentar diferentes formatos de fecha
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
                
        # Si es solo un año
        if len(str(date_str).strip()) == 4 and str(date_str).strip().isdigit():
            return datetime(int(str(date_str).strip()), 1, 1)
        
        return None
    except:
        return None

# Función para calcular el porcentaje de avance
def calculate_progress(row):
    # Contar columnas que representan actividades
    activity_columns = [col for col in row.index if col not in ['Funcionario', 'Entidad', 'Nive de información', 'Estado', 'Programado', 'Estado_Vencimiento', 'Porcentaje de Avance']]
    
    # Contar actividades completadas (que tienen fecha)
    completed_activities = sum(1 for col in activity_columns if pd.notna(row[col]) and row[col] != "" and row[col] != "-")
    
    # Calcular porcentaje
    if len(activity_columns) > 0:
        return round((completed_activities / len(activity_columns)) * 100, 1)
    else:
        return 0.0

# Función para generar colores de entidades
def generate_entity_colors(entities):
    # Paleta de colores visualmente distintos
    color_palette = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 
        'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 
        'rgb(23, 190, 207)', 'rgb(174, 199, 232)', 'rgb(255, 187, 120)',
        'rgb(152, 223, 138)', 'rgb(255, 152, 150)', 'rgb(197, 176, 213)'
    ]
    
    # Asignar colores a entidades
    colors = {}
    for i, entity in enumerate(entities):
        colors[entity] = color_palette[i % len(color_palette)]
    
    return colors

# Función para crear diagrama de Gantt con mejoras
def create_gantt(df):
    # Filtrar filas con datos suficientes
    df_gantt = df.copy()
    
    # Preparar datos para diagrama de Gantt
    gantt_data = []
    
    # Obtener colores para las entidades
    unique_entities = df_gantt['Entidad'].dropna().unique()
    entity_colors = generate_entity_colors(unique_entities)
    
    for _, row in df_gantt.iterrows():
        if pd.notna(row.get('Funcionario')) and pd.notna(row.get('Entidad')) and pd.notna(row.get('Nive de información')):
            # Definir inicio y fin
            start_date = None
            end_date = None
            
            # Buscar fecha de inicio (primera fecha no vacía en la fila)
            for col in ['Actas de acercamiento y manifestación de interés', 'Suscripción acuerdo de compromiso']:
                if col in row and pd.notna(row[col]) and row[col] != "" and row[col] != "-":
                    start_date = parse_date(row[col])
                    if start_date:
                        break
            
            # Buscar fecha de fin (usar Programado si está disponible)
            if 'Programado' in row and pd.notna(row['Programado']) and row['Programado'] != "" and row['Programado'] != "-":
                end_date = parse_date(row['Programado'])
            else:
                # Si no hay fecha programada, buscar la última fecha de actividad
                for col in reversed(df_gantt.columns):
                    if isinstance(col, str) and (col.endswith('/2025') or col.endswith('/2024')):
                        if pd.notna(row.get(col)) and row.get(col) != "" and row.get(col) != "-":
                            end_date = parse_date(row[col])
                            if end_date:
                                break
            
            # Si no se encontró una fecha de fin, establecer un plazo predeterminado
            if start_date and not end_date:
                end_date = start_date + timedelta(days=90)  # 3 meses por defecto
            
            # Si tenemos fechas válidas, agregar a los datos del gantt
            if start_date and end_date:
                task_name = f"{row['Nive de información']} - {row['Porcentaje de Avance']}%"
                
                gantt_data.append({
                    'Task': task_name,
                    'Start': start_date,
                    'Finish': end_date,
                    'Funcionario': row['Funcionario'],
                    'Entidad': row['Entidad'],
                    'Nivel': row['Nive de información'],
                    'Avance': row['Porcentaje de Avance']
                })
    
    if len(gantt_data) > 0:
        df_gantt_chart = pd.DataFrame(gantt_data)
        
        # Crear figura con ff.create_gantt
        fig = ff.create_gantt(df_gantt_chart, index_col='Entidad', show_colorbar=True, group_tasks=True)
        
        # Actualizar colores según la entidad
        for i, task in enumerate(fig['data']):
            entity = getattr(task, 'name', None) or getattr(task, 'legendgroup', None)
            if entity in entity_colors:
                fig['data'][i]['marker']['color'] = entity_colors[entity]

        # Añadir línea vertical para la fecha actual
        today = datetime.now()
        fig.add_shape(
            type="line",
            x0=today,
            y0=0,
            x1=today,
            y1=len(gantt_data) + 1,
            line=dict(color="Red", width=2, dash="dash"),
        )
        
        # Añadir anotación para la fecha actual
        fig.add_annotation(
            x=today,
            y=0,
            text="HOY",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="red", size=12),
        )
        
        # Actualizar diseño
        fig.update_layout(
            title="Diagrama de Gantt - Seguimiento de Indicadores",
            xaxis_title="Fecha",
            yaxis_title="Nivel de Información",
            height=700,
            margin=dict(l=150),
            legend_title="Entidades"
        )
        
        return fig, df_gantt_chart
    
    return None, pd.DataFrame()

# Función para verificar el estado de vencimiento
def check_status(row):
    today = datetime.now()
    end_date = None
    
    # Buscar fecha de fin (usar Programado si está disponible)
    if 'Programado' in row and pd.notna(row['Programado']) and row['Programado'] != "" and row['Programado'] != "-":
        end_date = parse_date(row['Programado'])
    else:
        # Si no hay fecha programada, buscar la última fecha de actividad
        for col in reversed(row.index):
            if isinstance(col, str) and (col.endswith('/2025') or col.endswith('/2024')):
                if pd.notna(row.get(col)) and row.get(col) != "" and row.get(col) != "-":
                    end_date = parse_date(row[col])
                    if end_date:
                        break
    
    if end_date:
        if end_date < today:
            return "Vencido"
        elif end_date < today + timedelta(days=30):  # Próximo a vencer (30 días)
            return "Próximo a vencer"
        else:
            return "En tiempo"
    
    return "Sin fecha"

# Función para aplicar estilos condicionales a la tabla
def highlight_status(df):
    df_styled = df.copy()
    df_styled['Estado_Vencimiento'] = df.apply(check_status, axis=1)
    
    # Calcular porcentaje de avance
    df_styled['Porcentaje de Avance'] = df.apply(calculate_progress, axis=1)
    
    # Aplicar estilos condicionales
    def apply_color(row):
        if row['Estado_Vencimiento'] == 'Vencido':
            return ['background-color: #ffcccc'] * len(row)
        elif row['Estado_Vencimiento'] == 'Próximo a vencer':
            return ['background-color: #ffffcc'] * len(row)
        elif row['Estado_Vencimiento'] == 'En tiempo':
            return ['background-color: #ccffcc'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = df_styled.style.apply(apply_color, axis=1)
    return styled_df, df_styled

# Nuevas funciones para el Panel de Control

# Función para clasificar el estado del proyecto
def classify_project_status(progress):
    if progress == 100:
        return "Completo"
    elif progress > 0:
        return "En proceso"
    else:
        return "Pendiente"

# Función para identificar etapas del proceso
def identify_process_stage(column_name):
    # Clasificar columnas en etapas del proceso
    if any(term in column_name.lower() for term in ['acta', 'acercamiento', 'interés', 'suscripción', 'compromiso']):
        return "Acuerdos iniciales"
    elif any(term in column_name.lower() for term in ['análisis', 'analisis', 'información', 'cronograma', 'plan']):
        return "Implementación"
    elif any(term in column_name.lower() for term in ['verificación', 'verificacion', 'validación', 'validacion', 'datos', 'servicio']):
        return "Verificación"
    else:
        return "Otras actividades"

# Función para crear gráfico de barras de completitud por entidad
def create_completeness_by_entity(df):
    if df.empty or 'Entidad' not in df.columns:
        return None
    
    # Agrupar por entidad y calcular promedios de avance
    avg_progress = df.groupby('Entidad')['Porcentaje de Avance'].mean().reset_index()
    avg_progress['Porcentaje de Avance'] = avg_progress['Porcentaje de Avance'].round(1)
    
    # Ordenar de mayor a menor
    avg_progress = avg_progress.sort_values(by='Porcentaje de Avance', ascending=False)
    
    # Crear gráfico
    fig = px.bar(
        avg_progress, 
        x='Entidad', 
        y='Porcentaje de Avance',
        title='Porcentaje de Completitud por Entidad',
        text='Porcentaje de Avance',
        color='Porcentaje de Avance',
        color_continuous_scale='Blues',
        labels={'Porcentaje de Avance': 'Avance (%)'}
    )
    
    fig.update_layout(
        xaxis_title="Entidad",
        yaxis_title="Porcentaje de Avance",
        yaxis=dict(range=[0, 100]),
        height=500
    )
    
    return fig

# Función para crear gráfico de estado de proyectos
def create_project_status_chart(df):
    if df.empty:
        return None
    
    # Clasificar estado de proyectos
    df['Estado del Proyecto'] = df['Porcentaje de Avance'].apply(classify_project_status)
    
    # Contar proyectos por estado
    status_counts = df['Estado del Proyecto'].value_counts().reset_index()
    status_counts.columns = ['Estado', 'Cantidad']
    
    # Definir colores para cada estado
    colors = {'Completo': '#4CAF50', 'En proceso': '#2196F3', 'Pendiente': '#F44336'}
    
    # Crear gráfico de pie
    fig = px.pie(
        status_counts, 
        values='Cantidad', 
        names='Estado',
        title='Estado de los Proyectos',
        color='Estado',
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

# Función para crear gráfico de fechas programadas vs avance actual
def create_schedule_vs_progress(df):
    if df.empty or 'Programado' not in df.columns:
        return None
    
    # Crear datos para el gráfico
    plot_data = []
    
    for _, row in df.iterrows():
        if pd.notna(row.get('Programado')) and row.get('Programado') != "" and row.get('Programado') != "-":
            end_date = parse_date(row['Programado'])
            if end_date:
                plot_data.append({
                    'Entidad': row['Entidad'],
                    'Nivel': row['Nive de información'],
                    'Fecha Programada': end_date,
                    'Avance Actual': row['Porcentaje de Avance'],
                    'Días Restantes': (end_date - datetime.now()).days
                })
    
    if not plot_data:
        return None
        
    df_plot = pd.DataFrame(plot_data)
    
    # Crear gráfico de dispersión
    fig = px.scatter(
        df_plot,
        x='Fecha Programada',
        y='Avance Actual',
        color='Entidad',
        size='Avance Actual',
        hover_name='Nivel',
        title='Fechas Programadas vs. Avance Actual',
        labels={'Avance Actual': 'Avance (%)', 'Fecha Programada': 'Fecha Programada de Entrega'},
        size_max=30
    )
    
    # Añadir línea de referencia para avance esperado
    today = datetime.now()
    fig.add_trace(
        go.Scatter(
            x=[today, today.replace(year=today.year + 1)],
            y=[0, 100],
            mode='lines',
            name='Avance Esperado',
            line=dict(color='red', dash='dash'),
            opacity=0.7
        )
    )
    
    fig.update_layout(
        xaxis_title="Fecha Programada",
        yaxis_title="Porcentaje de Avance",
        yaxis=dict(range=[0, 100]),
        height=500
    )
    
    return fig

# Función para crear gráfico de barras apiladas del progreso por etapas
def create_stacked_progress_chart(df):
    if df.empty:
        return None
    
    # Clasificar las columnas por etapas
    stages = {}
    for col in df.columns:
        if col not in ['Funcionario', 'Entidad', 'Nive de información', 'Estado', 'Programado', 'Estado_Vencimiento', 'Porcentaje de Avance']:
            stage = identify_process_stage(col)
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(col)
    
    # Calcular progreso por entidad y etapa
    entities = df['Entidad'].unique()
    progress_data = []
    
    for entity in entities:
        entity_df = df[df['Entidad'] == entity]
        
        for stage, columns in stages.items():
            if not columns:  # Skip if no columns for this stage
                continue
                
            # Calcular avance promedio para esta etapa
            completed = 0
            total = 0
            
            for _, row in entity_df.iterrows():
                for col in columns:
                    if col in row:
                        total += 1
                        if pd.notna(row[col]) and row[col] != "" and row[col] != "-":
                            completed += 1
            
            progress = (completed / total * 100) if total > 0 else 0
            
            progress_data.append({
                'Entidad': entity,
                'Etapa': stage,
                'Progreso': round(progress, 1)
            })
    
    if not progress_data:
        return None
        
    df_progress = pd.DataFrame(progress_data)
    
    # Crear gráfico de barras apiladas
    fig = px.bar(
        df_progress,
        x='Entidad',
        y='Progreso',
        color='Etapa',
        title='Progreso de Entidades por Etapa del Proceso',
        labels={'Progreso': 'Avance (%)', 'Entidad': 'Entidad', 'Etapa': 'Etapa del Proceso'},
        category_orders={"Etapa": ["Acuerdos iniciales", "Implementación", "Verificación", "Otras actividades"]},
        color_discrete_map={
            'Acuerdos iniciales': '#1976D2',
            'Implementación': '#388E3C',
            'Verificación': '#FBC02D',
            'Otras actividades': '#7B1FA2'
        }
    )
    
    fig.update_layout(
        xaxis_title="Entidad",
        yaxis_title="Porcentaje de Avance",
        yaxis=dict(range=[0, 100]),
        height=500,
        barmode='stack'
    )
    
    return fig

# Función para crear gráfico de Gantt por funcionario

def create_gantt_by_staff(df, selected_staff=None):
    if not selected_staff:
        return None, None

    df_filtered = df[df['Funcionario'] == selected_staff].copy()

    # Calcular el porcentaje de avance si no existe
    if 'Porcentaje de Avance' not in df_filtered.columns:
        df_filtered['Porcentaje de Avance'] = df_filtered.apply(calculate_progress, axis=1)

    return create_gantt(df_filtered)

# Función para crear mapa de calor de etapas con demoras
def create_delay_heatmap(df):
    if df.empty:
        return None
    
    # Identificar etapas del proceso para cada columna
    stages = {}
    activity_columns = []
    
    for col in df.columns:
        if col not in ['Funcionario', 'Entidad', 'Nive de información', 'Estado', 'Programado', 'Estado_Vencimiento', 'Porcentaje de Avance']:
            stage = identify_process_stage(col)
            stages[col] = stage
            activity_columns.append(col)
    
    if not activity_columns:
        return None
    
    # Calcular retrasos para cada actividad y entidad
    delay_data = {}
    entities = df['Entidad'].unique()
    
    for col in activity_columns:
        delays = []
        for entity in entities:
            entity_df = df[df['Entidad'] == entity]
            
            # Contar cuántas actividades tienen fecha y cuántas no
            total = len(entity_df)
            with_date = sum(1 for _, row in entity_df.iterrows() 
                         if pd.notna(row.get(col)) and row[col] != "" and row[col] != "-")
            
            # Calcular tasa de completitud (inversa de retraso)
            completion_rate = with_date / total if total > 0 else 0
            delay_rate = 1 - completion_rate  # Convertir a tasa de retraso
            
            delays.append(delay_rate)
        
        # Promedio de retrasos para esta actividad
        delay_data[col] = sum(delays) / len(delays) if delays else 0
    
    # Organizar datos para el mapa de calor
    heatmap_data = []
    for col, delay in delay_data.items():
        stage = stages[col]
        heatmap_data.append({
            'Actividad': col[:30] + '...' if len(col) > 30 else col,  # Truncar nombres largos
            'Etapa': stage,
            'Índice de Demora': round(delay * 100, 1)  # Convertir a porcentaje
        })
    
    df_heatmap = pd.DataFrame(heatmap_data)
    
    # Ordenar por etapa y luego por índice de demora
    df_heatmap = df_heatmap.sort_values(by=['Etapa', 'Índice de Demora'], ascending=[True, False])
    
    # Crear mapa de calor
    fig = px.density_heatmap(
        df_heatmap,
        x='Etapa',
        y='Actividad',
        z='Índice de Demora',
        title='Mapa de Calor de Demoras por Etapa y Actividad',
        labels={'Índice de Demora': 'Demora (%)', 'Etapa': 'Etapa del Proceso', 'Actividad': 'Actividad'},
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis_title="Etapa del Proceso",
        yaxis_title="Actividad",
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Función para estimar frecuencia de actualización
def estimate_update_frequency(df):
    if df.empty:
        return None
    
    activity_columns = [col for col in df.columns 
                       if col not in ['Funcionario', 'Entidad', 'Nive de información', 'Estado', 'Programado', 'Estado_Vencimiento', 'Porcentaje de Avance']]
    
    # Obtener fechas para actividades
    dates_by_indicator = {}
    
    for _, row in df.iterrows():
        indicator = row.get('Nive de información', 'Sin nombre')
        
        if indicator not in dates_by_indicator:
            dates_by_indicator[indicator] = []
        
        for col in activity_columns:
            if pd.notna(row.get(col)) and row[col] != "" and row[col] != "-":
                date = parse_date(row[col])
                if date:
                    dates_by_indicator[indicator].append(date)
    
    # Calcular frecuencia promedio para cada indicador
    frequency_data = []
    
    for indicator, dates in dates_by_indicator.items():
        if len(dates) >= 2:
            # Ordenar fechas
            sorted_dates = sorted(dates)
            
            # Calcular diferencias en días
            diff_days = [(sorted_dates[i+1] - sorted_dates[i]).days 
                         for i in range(len(sorted_dates)-1)]
            
            if diff_days:
                avg_days = sum(diff_days) / len(diff_days)
                
                # Clasificar frecuencia
                if avg_days <= 60:  # 2 meses
                    frequency = "Bimensual"
                elif avg_days <= 90:  # 3 meses
                    frequency = "Trimestral"
                elif avg_days <= 180:  # 6 meses
                    frequency = "Semestral"
                else:
                    frequency = "Anual"
                    
                frequency_data.append({
                    'Indicador': indicator,
                    'Frecuencia': frequency,
                    'Promedio de días': round(avg_days, 1)
                })
    
    if not frequency_data:
        return None
        
    df_frequency = pd.DataFrame(frequency_data)
    
    # Contar indicadores por frecuencia
    freq_counts = df_frequency['Frecuencia'].value_counts().reset_index()
    freq_counts.columns = ['Frecuencia', 'Cantidad']
    
    # Ordenar por frecuencia
    order = {'Bimensual': 1, 'Trimestral': 2, 'Semestral': 3, 'Anual': 4}
    freq_counts['Orden'] = freq_counts['Frecuencia'].map(order)
    freq_counts = freq_counts.sort_values('Orden').drop('Orden', axis=1)
    
    # Crear gráfico de barras
    fig = px.bar(
        freq_counts,
        x='Frecuencia',
        y='Cantidad',
        title='Indicadores Agrupados por Frecuencia de Actualización',
        labels={'Cantidad': 'Número de Indicadores', 'Frecuencia': 'Frecuencia de Actualización'},
        color='Frecuencia',
        color_discrete_map={
            'Bimensual': '#1E88E5',
            'Trimestral': '#43A047',
            'Semestral': '#FB8C00',
            'Anual': '#E53935'
        }
    )
    
    fig.update_layout(
        xaxis_title="Frecuencia de Actualización",
        yaxis_title="Número de Indicadores",
        height=400
    )
    
    return fig

# Cargar datos
df = load_data()

# Interfaz principal
if not df.empty:
    # Sidebar
    st.sidebar.header("Filtros")
    
    # Obtener valores únicos para filtros
    funcionarios = ['Todos'] + sorted(df['Funcionario'].dropna().unique().tolist()) if 'Funcionario' in df.columns else ['Todos']
    entidades = ['Todas'] + sorted(df['Entidad'].dropna().unique().tolist()) if 'Entidad' in df.columns else ['Todas']
    
    # Selectores de filtro
    funcionario_seleccionado = st.sidebar.selectbox("Funcionario", funcionarios)
    entidad_seleccionada = st.sidebar.selectbox("Entidad", entidades)
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    if funcionario_seleccionado != 'Todos' and 'Funcionario' in df.columns:
        df_filtrado = df_filtrado[df_filtrado['Funcionario'] == funcionario_seleccionado]
    
    if entidad_seleccionada != 'Todas' and 'Entidad' in df.columns:
        df_filtrado = df_filtrado[df_filtrado['Entidad'] == entidad_seleccionada]
    
    # Calcular porcentaje de avance antes de todo
    df_filtrado['Porcentaje de Avance'] = df_filtrado.apply(calculate_progress, axis=1)
    
    # Pestaña principal - Diagrama de Gantt y tabla de datos
    tab1, tab2, tab3, tab4 = st.tabs(["Seguimiento con Gantt", "Tabla de Datos", "Editar Datos", "Panel de Control"])
    
    with tab1:
        st.header("Diagrama de Gantt")
        
        # Crear diagrama de Gantt mejorado
        fig_gantt, df_gantt = create_gantt(df_filtrado)
        
        if fig_gantt:
            st.plotly_chart(fig_gantt, use_container_width=True)
            
            # Mostrar leyenda de entidades
            st.subheader("Leyenda de Entidades")
            entities = df_filtrado['Entidad'].dropna().unique()
            entity_colors = generate_entity_colors(entities)
            
            # Mostrar leyenda en columnas
            cols = st.columns(4)
            for i, entity in enumerate(entities):
                color = entity_colors[entity]
                cols[i % 4].markdown(f'<div style="background-color: {color}; padding: 5px; border-radius: 5px; margin: 2px; color: white;">{entity}</div>', unsafe_allow_html=True)
        else:
            st.warning("No hay datos suficientes para crear el diagrama de Gantt con los filtros actuales.")
    
    with tab2:
        st.header("Tabla de Datos")
        
        # Aplicar estilos condicionales y mostrar la tabla
        styled_df, df_with_status = highlight_status(df_filtrado)
        
        # Mostrar leyenda de colores para estados
        st.subheader("Estados de Vencimiento")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div style="background-color: #ffcccc; padding: 10px; border-radius: 5px;">Vencido</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="background-color: #ffffcc; padding: 10px; border-radius: 5px;">Próximo a vencer (30 días)</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div style="background-color: #ccffcc; padding: 10px; border-radius: 5px;">En tiempo</div>', unsafe_allow_html=True)
        
        # Mostrar datos filtrados con estilos
        st.dataframe(styled_df, use_container_width=True)
        
        # Mostrar resumen de progreso
        st.subheader("Resumen de Progreso")
        promedio_avance = df_with_status['Porcentaje de Avance'].mean()
        tareas_completadas = len(df_with_status[df_with_status['Porcentaje de Avance'] == 100])
        tareas_en_progreso = len(df_with_status[(df_with_status['Porcentaje de Avance'] > 0) & (df_with_status['Porcentaje de Avance'] < 100)])
        tareas_no_iniciadas = len(df_with_status[df_with_status['Porcentaje de Avance'] == 0])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Promedio de Avance", f"{promedio_avance:.1f}%")
        col2.metric("Tareas Completadas", tareas_completadas)
        col3.metric("Tareas en Progreso", tareas_en_progreso)
        col4.metric("Tareas No Iniciadas", tareas_no_iniciadas)
        
        # Botón para descargar datos
        csv = download_dataframe(df_with_status)
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name="seguimiento_indicadores.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.header("Editar Datos")
        
        # Crear una copia editable de los datos
        edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        
        # Botón para guardar cambios
        if st.button("Guardar Cambios"):
            # Guardar datos en un archivo temporal
            edited_csv = download_dataframe(edited_df)
            with open("Prueba seguimiento.csv", "w", encoding="utf-8") as f:
                f.write(edited_csv)
            
            st.success("Cambios guardados correctamente.")
            st.rerun()  # Recargar la aplicación para reflejar los cambios
    
    with tab4:
        st.header("Panel de Control General")
        
        # Asegurarse que el dataframe tiene los datos necesarios
        df_pc = df_filtrado.copy()
        df_pc['Porcentaje de Avance'] = df_pc.apply(calculate_progress, axis=1)
        df_pc['Estado_Vencimiento'] = df_pc.apply(check_status, axis=1)
        
        # Primera fila de gráficos - Completitud y Estado
        st.subheader("Indicadores Generales")
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico 1: Porcentaje de completitud por entidad
            fig_completeness = create_completeness_by_entity(df_pc)
            if fig_completeness:
                st.plotly_chart(fig_completeness, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar la completitud por entidad.")
        
        with col2:
            # Gráfico 2: Estado de los proyectos
            fig_status = create_project_status_chart(df_pc)
            if fig_status:
                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para mostrar el estado de los proyectos.")
        
        # Segunda fila - Fechas programadas vs avance y barras apiladas
        st.subheader("Seguimiento de Avance por Etapas")
        
        # Gráfico 3: Fechas programadas vs avance actual
        fig_schedule = create_schedule_vs_progress(df_pc)
        if fig_schedule:
            st.plotly_chart(fig_schedule, use_container_width=True)
        else:
            st.warning("No hay fechas programadas suficientes para mostrar la relación de avance.")
            
        # Gráfico 4: Gráfico de barras apiladas
        fig_stacked = create_stacked_progress_chart(df_pc)
        if fig_stacked:
            st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para mostrar el progreso por etapas.")
        
        # Tercera fila - Gantt por funcionario y mapa de calor
        st.subheader("Análisis Detallado")
        
        # Gráfico 5: Gantt por funcionario
        st.markdown("#### Diagrama de Gantt por Funcionario")
        # Selector de funcionario
        func_options = ['Seleccione un funcionario'] + sorted(df['Funcionario'].dropna().unique().tolist())
        selected_func = st.selectbox("Seleccione un funcionario para ver su diagrama de Gantt", func_options)
        
        if selected_func != 'Seleccione un funcionario':
            fig_gantt_func, _ = create_gantt_by_staff(df, selected_func)
            if fig_gantt_func:
                st.plotly_chart(fig_gantt_func, use_container_width=True)
            else:
                st.warning(f"No hay datos suficientes para mostrar el diagrama de Gantt para {selected_func}.")
        
        # Gráfico 6: Mapa de calor de etapas con demoras
        st.markdown("#### Mapa de Calor de Demoras por Etapa")
        fig_heatmap = create_delay_heatmap(df_pc)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para mostrar el mapa de calor de demoras.")
        
        # Cuarta fila - Frecuencia de actualización
        st.subheader("Frecuencia de Actualización")
        
        # Gráfico 7: Gráfico de barras de frecuencia
        fig_frequency = estimate_update_frequency(df_pc)
        if fig_frequency:
            st.plotly_chart(fig_frequency, use_container_width=True)
        else:
            st.warning("No hay suficientes fechas para estimar la frecuencia de actualización.")
        
        # Sección de métricas clave
        st.subheader("Métricas Clave")
        
        # Crear métricas clave
        col1, col2, col3, col4 = st.columns(4)
        
        # Métrica 1: Promedio de avance
        avg_progress = df_pc['Porcentaje de Avance'].mean()
        col1.metric("Avance Promedio", f"{avg_progress:.1f}%")
        
        # Métrica 2: Porcentaje de indicadores vencidos
        vencidos = len(df_pc[df_pc['Estado_Vencimiento'] == 'Vencido'])
        porcentaje_vencidos = (vencidos / len(df_pc) * 100) if len(df_pc) > 0 else 0
        col2.metric("Indicadores Vencidos", f"{porcentaje_vencidos:.1f}%")
        
        # Métrica 3: Indicadores completados
        completados = len(df_pc[df_pc['Porcentaje de Avance'] == 100])
        porcentaje_completados = (completados / len(df_pc) * 100) if len(df_pc) > 0 else 0
        col3.metric("Indicadores Completados", f"{porcentaje_completados:.1f}%")
        
        # Métrica 4: Indicadores sin iniciar
        sin_iniciar = len(df_pc[df_pc['Porcentaje de Avance'] == 0])
        porcentaje_sin_iniciar = (sin_iniciar / len(df_pc) * 100) if len(df_pc) > 0 else 0
        col4.metric("Indicadores Sin Iniciar", f"{porcentaje_sin_iniciar:.1f}%")
        
        # Botón para descargar informe
        st.markdown("#### Descargar Informe")
        
        # Preparar informe básico
        buffer = io.StringIO()
        buffer.write("# INFORME DE SEGUIMIENTO DE INDICADORES\n\n")
        buffer.write(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y')}\n\n")
        buffer.write(f"## Resumen General\n")
        buffer.write(f"- Avance promedio: {avg_progress:.1f}%\n")
        buffer.write(f"- Indicadores completados: {completados} ({porcentaje_completados:.1f}%)\n")
        buffer.write(f"- Indicadores en progreso: {len(df_pc[(df_pc['Porcentaje de Avance'] > 0) & (df_pc['Porcentaje de Avance'] < 100)])}\n")
        buffer.write(f"- Indicadores sin iniciar: {sin_iniciar} ({porcentaje_sin_iniciar:.1f}%)\n")
        buffer.write(f"- Indicadores vencidos: {vencidos} ({porcentaje_vencidos:.1f}%)\n\n")
        
        # Agregar detalle por entidad
        buffer.write("## Detalle por Entidad\n")
        entidades = df_pc['Entidad'].unique()
        for entidad in entidades:
            df_ent = df_pc[df_pc['Entidad'] == entidad]
            avg_ent = df_ent['Porcentaje de Avance'].mean()
            vencidos_ent = len(df_ent[df_ent['Estado_Vencimiento'] == 'Vencido'])
            buffer.write(f"### {entidad}\n")
            buffer.write(f"- Avance promedio: {avg_ent:.1f}%\n")
            buffer.write(f"- Indicadores vencidos: {vencidos_ent}\n")
            buffer.write(f"- Total indicadores: {len(df_ent)}\n\n")
        
        # Botón de descarga
        st.download_button(
            label="Descargar Informe en Markdown",
            data=buffer.getvalue(),
            file_name="informe_seguimiento.md",
            mime="text/markdown"
        )
