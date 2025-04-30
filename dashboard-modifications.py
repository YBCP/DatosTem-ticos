import pandas as pd
import dash
from dash import dcc, html, dash_table, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Cargar y limpiar los datos
def cargar_datos():
    try:
        # Intentar cargar directamente desde el archivo CSV
        df = pd.read_csv('Tematicos.csv', sep=';', encoding='utf-8')
    except:
        # Si falla, usar el método alternativo
        print("Usando método alternativo para cargar datos")
        data = []
        with open('Tematicos.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            headers = lines[0].strip().split(';')
            for line in lines[1:]:
                if line.strip():  # Verificar que la línea no esté vacía
                    values = line.strip().split(';')
                    if len(values) >= 5:  # Verificar que haya suficientes valores
                        # Rellenar con None si faltan valores
                        while len(values) < len(headers):
                            values.append(None)
                        data.append(dict(zip(headers, values)))
        
        df = pd.DataFrame(data)
    
    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    
    # Convertir fechas al formato adecuado
    date_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['fecha', 'programado', 'suscripción', 'entrega'])]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%d/%m/%Y')
    
    # Calcular estado de avance numérico para visualizaciones
    df['Avance_Porcentaje'] = calcular_avance(df)
    
    # Determinar el estado basado en las columnas existentes (derivado)
    df['Estado'] = determinar_estado(df)
    
    return df

def determinar_estado(df):
    """Determina el estado (Completo, Proceso, Pendiente) basado en las columnas del dataframe"""
    estados = []
    
    for _, row in df.iterrows():
        # Verificar si está en la columna "Observaciones"
        observacion = str(row.get('Observaciones', '')).lower() if pd.notna(row.get('Observaciones')) else ''
        
        if 'completo' in observacion:
            estados.append('Completo')
        elif 'proceso' in observacion:
            estados.append('Proceso')
        # Si tiene valor en la columna "Oficios de cierre"
        elif pd.notna(row.get('Oficios de cierre')) and row['Oficios de cierre'] == 'Si':
            estados.append('Completo')
        # Si el porcentaje de avance es 100%
        elif row['Avance_Porcentaje'] == 100:
            estados.append('Completo')
        # Si el porcentaje de avance es mayor que 0%
        elif row['Avance_Porcentaje'] > 0:
            estados.append('Proceso')
        else:
            estados.append('Pendiente')
    
    return estados

def calcular_avance(df):
    """Calcula el porcentaje de avance basado en las columnas completadas"""
    # Columnas de proceso que pueden ser "Completo" o "Si"
    columnas_proceso = ['Actas de acercamiento y manifestación de interés', 
                       'Suscripción acuerdo de compromiso', 
                       'Entrega acuerdo de compromiso',
                       'Acuerdo de compromiso', 
                       'Gestion acceso a los datos y documentos requeridos',
                       'Análisis de información', 
                       'Cronograma Concertado', 
                       'Seguimiento a los acuerdos definidos',
                       'Registro', 'ET', 'CO', 'DD', 'REC', 'SERVICIO', 
                       'Resultados orientación técnica',
                       'Verificación del servicio web geográfico',
                       'Verificar aprobar resultados',
                       'Revisar y validar los datos cargados en la base de datos',
                       'Aprobación resultados orientación',
                       'Disponer datos temáticos',
                       'Catálogo de recursos\ngeográficos',
                       'Oficios de cierre']
    
    # Contar cuántas de estas columnas están marcadas como "Completo" o "Si"
    # y calcular el porcentaje sobre el total de columnas
    def calcular_fila(row):
        completados = 0
        total = 0
        for col in columnas_proceso:
            if col in row and pd.notna(row[col]):
                total += 1
                if row[col] in ['Completo', 'Si']:
                    completados += 1
        
        if total == 0:
            return 0
        return (completados / total) * 100
    
    return df.apply(calcular_fila, axis=1)

# Inicializar la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Definir el layout con filtros horizontales
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1('Dashboard de Seguimiento de Datos Temáticos',
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px', 'marginTop': '20px'}),
    
    # Sección de filtros en una fila horizontal
    html.Div([
        html.H3('Filtros', style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
        html.Div([
            # Columna para el filtro de Funcionario
            html.Div([
                html.Label('Funcionario:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-funcionario',
                    options=[],  # Se llenará en el callback
                    multi=True,
                    placeholder='Seleccione funcionario(s)'
                ),
            ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
            
            # Columna para el filtro de Entidad
            html.Div([
                html.Label('Entidad:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-entidad',
                    options=[],  # Se llenará en el callback
                    multi=True,
                    placeholder='Seleccione entidad(es)'
                ),
            ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
            
            # Columna para el filtro de Estado
            html.Div([
                html.Label('Estado:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-estado',
                    options=[],  # Se llenará en el callback
                    multi=True,
                    placeholder='Seleccione estado(s)'
                ),
            ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
            
            # Columna para el filtro de Tipo de dato
            html.Div([
                html.Label('Tipo de dato:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='dropdown-tipo-dato',
                    options=[],  # Se llenará en el callback
                    multi=True,
                    placeholder='Seleccione tipo(s) de dato'
                ),
            ], style={'width': '24%', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '10px'})
    ], style={'padding': '20px', 'margin': '0 0 20px 0', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 
              'borderRadius': '5px', 'backgroundColor': 'white'}),
    
    # Sección de Resumen
    html.Div([
        html.H3('Resumen', style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
        html.Div([
            html.Div(id='indicador-total', className='indicador'),
            html.Div(id='indicador-completados', className='indicador'),
            html.Div(id='indicador-en-proceso', className='indicador'),
            html.Div(id='indicador-pendientes', className='indicador'),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
        
        html.Div([
            dcc.Graph(id='grafico-avance-entidades')
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            dcc.Graph(id='grafico-estado-por-entidad')
        ]),
    ], style={'padding': '20px', 'margin': '0 0 20px 0', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 
              'borderRadius': '5px', 'backgroundColor': 'white'}),
    
    html.Div([
        html.H3('Próximas Fechas Programadas', style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
        html.Div(id='proximas-fechas')
    ], style={'padding': '20px', 'margin': '20px 0', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '5px',
              'backgroundColor': 'white'}),
    
    html.Div([
        html.H3('Cronograma de Actividades', style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
        dcc.Graph(id='grafico-cronograma'),
    ], style={'padding': '20px', 'margin': '20px 0', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '5px',
              'backgroundColor': 'white'}),
    
    html.Div([
        html.H3('Detalle de Indicadores', style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
        dash_table.DataTable(

            id='tabla-indicadores',
            columns=[],  # Se llenará en el callback
            data=[],     # Se llenará en el callback
            hidden_columns=['fecha_vencida'],  # Hide this column by default
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            page_action='native',
            page_size=15,
            style_table={'overflowX': 'auto'},
            style_cell={
                'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'textAlign': 'left'
            },
            style_header={
                'backgroundColor': '#34495e',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f2f2f2'
                },
                {
                    'if': {'filter_query': '{Estado} = "Completo"'},
                    'backgroundColor': '#d4efdf',
                },
                {
                    'if': {'filter_query': '{Estado} = "Proceso"'},
                    'backgroundColor': '#fef9e7',
                },
                {
                    'if': {'filter_query': '{Estado} = "Pendiente"'},
                    'backgroundColor': '#f9ebea',
                },
                # Nuevas condiciones para colorear fechas programadas
                {
                    'if': {
                        'filter_query': '{fecha_vencida} = "vencida"',
                        'column_id': 'Programado'
                    },
                    'backgroundColor': '#ff6b6b',  # Rojo para fechas vencidas
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': '{fecha_vencida} = "proxima"',
                        'column_id': 'Programado'
                    },
                    'backgroundColor': '#feca57',  # Amarillo para fechas próximas a vencer
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': '{fecha_vencida} = "futura"',
                        'column_id': 'Programado'
                    },
                    'backgroundColor': '#1dd1a1',  # Verde para fechas futuras
                    'color': 'white'
                }
            ],
            tooltip_duration=None,
            tooltip_data=[],  # Se llenará en el callback
        )
    ], style={'padding': '20px', 'margin': '20px 0', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'borderRadius': '5px',
              'backgroundColor': 'white'}),
])

# Callback para actualizar todos los componentes basados en los filtros
@app.callback(
    [
        Output('dropdown-funcionario', 'options'),
        Output('dropdown-entidad', 'options'),
        Output('dropdown-estado', 'options'),
        Output('dropdown-tipo-dato', 'options'),
        Output('indicador-total', 'children'),
        Output('indicador-completados', 'children'),
        Output('indicador-en-proceso', 'children'),
        Output('indicador-pendientes', 'children'),
        Output('grafico-avance-entidades', 'figure'),
        Output('grafico-estado-por-entidad', 'figure'),
        Output('grafico-cronograma', 'figure'),
        Output('proximas-fechas', 'children'),
        Output('tabla-indicadores', 'columns'),
        Output('tabla-indicadores', 'data'),
        Output('tabla-indicadores', 'tooltip_data'),
        Output('tabla-indicadores', 'hidden_columns')  # Add this line
    ],
    [
        Input('dropdown-funcionario', 'value'),
        Input('dropdown-entidad', 'value'),
        Input('dropdown-estado', 'value'),
        Input('dropdown-tipo-dato', 'value')
    ]
)
def actualizar_dashboard(funcionarios_seleccionados, entidades_seleccionadas, estados_seleccionados,tipos_dato_seleccionados):
    # Cargar datos
    df = cargar_datos()
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    if funcionarios_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Funcionario'].isin(funcionarios_seleccionados)]
    
    if entidades_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['Entidad'].isin(entidades_seleccionadas)]
    
    if estados_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Estado'].isin(estados_seleccionados)]
    
    if tipos_dato_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Tipo dato'].isin(tipos_dato_seleccionados)]
    
    # Preparar opciones para dropdowns
    opciones_funcionario = [{'label': f, 'value': f} for f in sorted(df['Funcionario'].dropna().unique())]
    opciones_entidad = [{'label': e, 'value': e} for e in sorted(df['Entidad'].dropna().unique())]
    opciones_estado = [{'label': e, 'value': e} for e in sorted(df['Estado'].unique())]
    opciones_tipo_dato = [{'label': t, 'value': t} for t in sorted(df['Tipo dato'].dropna().unique())]
    
    # Preparar indicadores de resumen
    total_indicadores = len(df_filtrado)
    completados = len(df_filtrado[df_filtrado['Estado'] == 'Completo'])
    en_proceso = len(df_filtrado[df_filtrado['Estado'] == 'Proceso'])
    pendientes = len(df_filtrado[df_filtrado['Estado'] == 'Pendiente'])
    
    indicador_total = html.Div([
        html.H4('Total Indicadores'),
        html.P(f"{total_indicadores}", style={'fontSize': '24px', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'borderRadius': '5px',
              'boxShadow': '0 2px 4px 0 rgba(0,0,0,0.2)'})
    
    indicador_completados = html.Div([
        html.H4('Completados'),
        html.P(f"{completados}", style={'fontSize': '24px', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 'borderRadius': '5px',
              'boxShadow': '0 2px 4px 0 rgba(0,0,0,0.2)'})
    
    indicador_en_proceso = html.Div([
        html.H4('En Proceso'),
        html.P(f"{en_proceso}", style={'fontSize': '24px', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f39c12', 'color': 'white', 'borderRadius': '5px',
              'boxShadow': '0 2px 4px 0 rgba(0,0,0,0.2)'})
    
    indicador_pendientes = html.Div([
        html.H4('Pendientes'),
        html.P(f"{pendientes}", style={'fontSize': '24px', 'fontWeight': 'bold'})
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#e74c3c', 'color': 'white', 'borderRadius': '5px',
              'boxShadow': '0 2px 4px 0 rgba(0,0,0,0.2)'})
    
    # Preparar gráfico de avance por entidades
    df_avance_entidades = df_filtrado.groupby('Entidad')['Avance_Porcentaje'].mean().reset_index()
    df_avance_entidades = df_avance_entidades.sort_values('Avance_Porcentaje', ascending=False)
    
    figura_avance_entidades = px.bar(
        df_avance_entidades,
        x='Entidad',
        y='Avance_Porcentaje',
        title='Porcentaje de Avance por Entidad',
        labels={'Avance_Porcentaje': 'Avance (%)', 'Entidad': 'Entidad'},
        color='Avance_Porcentaje',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    figura_avance_entidades.update_layout(
        xaxis={'categoryorder': 'total descending', 'title': 'Entidad'},
        yaxis={'range': [0, 100], 'title': 'Avance (%)'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Preparar gráfico de estado por entidad
    conteo_estado_entidad = df_filtrado.groupby(['Entidad', 'Estado']).size().reset_index(name='Conteo')
    
    figura_estado_entidad = px.bar(
        conteo_estado_entidad,
        x='Entidad',
        y='Conteo',
        title='Estado de Indicadores por Entidad',
        color='Estado',
        barmode='stack',
        color_discrete_map={
            'Completo': '#2ecc71',
            'Proceso': '#f39c12',
            'Pendiente': '#e74c3c'
        }
    )
    
    figura_estado_entidad.update_layout(
        xaxis={'title': 'Entidad'},
        yaxis={'title': 'Cantidad de Indicadores'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Próximas fechas programadas (ordenadas)
    hoy = pd.Timestamp(datetime.now())
    proximas_fechas_df = df_filtrado[df_filtrado['Programado'] >= hoy].sort_values('Programado')
    
    proximas_fechas_html = []
    for _, row in proximas_fechas_df.head(5).iterrows():
        fecha = row['Programado']
        entidad = row['Entidad']
        nivel_info = row['Nive de información']
        dias_restantes = (fecha - hoy).days
        
        # Determinar color según proximidad
        if dias_restantes <= 7:
            color = '#e74c3c'  # Rojo - Urgente
        elif dias_restantes <= 30:
            color = '#f39c12'  # Amarillo - Próximo
        else:
            color = '#2ecc71'  # Verde - Con tiempo
        
        proximas_fechas_html.append(
            html.Div([
                html.Div([
                    html.H4(f"{fecha.strftime('%d/%m/%Y')}"),
                    html.P(f"Días restantes: {dias_restantes}", style={'fontWeight': 'bold'})
                ], style={'width': '30%'}),
                html.Div([
                    html.H4(f"{entidad}"),
                    html.P(f"{nivel_info}")
                ], style={'width': '70%'})
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'padding': '10px',
                'margin': '10px 0',
                'borderLeft': f'5px solid {color}',
                'boxShadow': '0 2px 4px 0 rgba(0,0,0,0.1)',
                'borderRadius': '5px'
            })
        )
    
    if not proximas_fechas_html:
        proximas_fechas_html = [
            html.Div("No hay fechas programadas próximamente", 
                     style={'textAlign': 'center', 'padding': '20px', 'color': '#7f8c8d'})
        ]
    
    # Preparar gráfico de cronograma
    # Ordenar por fecha programada
    df_cronograma = df_filtrado.copy()
    
    # Asegurar que todas las filas tengan una fecha de inicio y fin
    fecha_inicial = pd.Timestamp('2024-01-01')
    df_cronograma['fecha_inicio'] = df_cronograma['Suscripción acuerdo de compromiso'].fillna(fecha_inicial)
    df_cronograma['fecha_fin'] = df_cronograma['Programado'].fillna(hoy + pd.Timedelta(days=30))
    
    # Crear gráfico de Gantt
    figura_cronograma = px.timeline(
        df_cronograma,
        x_start='fecha_inicio',
        x_end='fecha_fin',
        y='Nive de información',
        color='Estado',
        title='Cronograma de Actividades por Indicador',
        color_discrete_map={
            'Completo': '#2ecc71',
            'Proceso': '#f39c12',
            'Pendiente': '#e74c3c'
        },
        hover_data=['Entidad', 'Funcionario', 'Avance_Porcentaje']
    )
    
    # Agregar línea vertical para la fecha actual
    figura_cronograma.add_vline(x=hoy, line_width=3, line_dash="dash", line_color="red")
    
    # Agregar anotación para la fecha actual
    figura_cronograma.add_annotation(
        x=hoy,
        y=0,
        text="Hoy",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=30
    )
    
    figura_cronograma.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Indicador",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis={'autorange': 'reversed'}
    )
    
    # Preparar tabla de detalles con color para fechas programadas

    columnas_tabla = [
        {'name': 'Cod', 'id': 'Cod'},
        {'name': 'Funcionario', 'id': 'Funcionario'},
        {'name': 'Entidad', 'id': 'Entidad'},
        {'name': 'Nivel de información', 'id': 'Nive de información'},
        {'name': 'Frecuencia', 'id': 'Frecuencia actualizacion'},
        {'name': 'Tipo dato', 'id': 'Tipo dato'},
        {'name': 'Estado', 'id': 'Estado'},
        {'name': 'Avance (%)', 'id': 'Avance_Porcentaje', 'type': 'numeric', 'format': {'specifier': '.0f'}},
        {'name': 'Programado', 'id': 'Programado', 'type': 'datetime'},
        {'name': 'Observaciones', 'id': 'Observaciones'},
        # Changed approach for hidden column
        {'name': 'fecha_vencida', 'id': 'fecha_vencida'}
    ]
    
    # Formatear las fechas para la tabla y añadir la clasificación de vencimiento
    df_para_tabla = df_filtrado.copy()
    
    # Agregar columna para determinar si la fecha está vencida, próxima a vencer o en el futuro
    df_para_tabla['fecha_vencida'] = 'futura'  # Por defecto todas son futuras
    
    if 'Programado' in df_para_tabla.columns and pd.api.types.is_datetime64_any_dtype(df_para_tabla['Programado']):
        # Marcar fechas vencidas (antes de hoy)
        df_para_tabla.loc[df_para_tabla['Programado'] < hoy, 'fecha_vencida'] = 'vencida'
        
        # Marcar fechas próximas a vencer (en los próximos 15 días)
        fecha_limite = hoy + timedelta(days=15)
        df_para_tabla.loc[(df_para_tabla['Programado'] >= hoy) & 
                           (df_para_tabla['Programado'] <= fecha_limite), 
                           'fecha_vencida'] = 'proxima'
        
        # Formatear fechas para mostrar en la tabla
        df_para_tabla['Programado'] = df_para_tabla['Programado'].dt.strftime('%d/%m/%Y')
    
    datos_tabla = df_para_tabla.to_dict('records')
    
    # Preparar datos para tooltips en la tabla
    tooltip_data = []
    for row in datos_tabla:
        tooltip_row = {}
        for col in ['Nive de información', 'Observaciones']:
            if col in row and pd.notna(row[col]):
                tooltip_row[col] = {'value': str(row[col]), 'type': 'markdown'}
        
        # Añadir tooltip informativo para las fechas según su estado
        if 'fecha_vencida' in row:
            if row['fecha_vencida'] == 'vencida':
                tooltip_row['Programado'] = {'value': 'Fecha vencida', 'type': 'markdown'}
            elif row['fecha_vencida'] == 'proxima':
                tooltip_row['Programado'] = {'value': 'Fecha próxima a vencer', 'type': 'markdown'}
        
        tooltip_data.append(tooltip_row)
    
    return (
        opciones_funcionario,
        opciones_entidad,
        opciones_estado,
        opciones_tipo_dato,
        indicador_total,
        indicador_completados,
        indicador_en_proceso,
        indicador_pendientes,
        figura_avance_entidades,
        figura_estado_entidad,
        figura_cronograma,
        proximas_fechas_html,
        columnas_tabla,
        datos_tabla,
        tooltip_data,
        ['fecha_vencida']
    )

if __name__ == '__main__':
    app.run(debug=True)
