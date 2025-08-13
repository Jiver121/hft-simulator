import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import threading
import time
import sys
sys.path.append('..')
from utils.performance_monitor import get_performance_metrics

# Dashboard app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('Performance Metrics Dashboard'),
    dcc.Interval(id='interval-update', interval=1000, n_intervals=0),  # 1 second
    dcc.Dropdown(id='section-dropdown', options=[], placeholder="Select section(s)", multi=True),
    dcc.Graph(id='perf-graph'),
    html.Div(id='summary-metrics', style={'marginTop': 20})
])

@app.callback(
    Output('section-dropdown', 'options'),
    Output('section-dropdown', 'value'),
    Input('interval-update', 'n_intervals'),
    Input('section-dropdown', 'value')
)
def update_dropdown(_, selected):
    metrics = get_performance_metrics()
    all_sections = list(metrics.keys())
    return [{'label': s, 'value': s} for s in all_sections], selected or all_sections

@app.callback(
    Output('perf-graph', 'figure'),
    Output('summary-metrics', 'children'),
    Input('interval-update', 'n_intervals'),
    Input('section-dropdown', 'value'),
)
def update_graph(_, sections):
    metrics = get_performance_metrics()
    if not sections:
        sections = list(metrics.keys())
    fig = go.Figure()
    summary = []
    for sec in sections:
        data = metrics.get(sec, [])
        if data:
            times = [d['time'] for d in data]
            mems = [d['memory_profiled'] for d in data]
            peaks = [d['mem_peak_kb'] for d in data]
            fig.add_trace(go.Scatter(y=times, mode='lines+markers', name=f'{sec} - Time/s'))
            fig.add_trace(go.Scatter(y=mems, mode='lines+markers', name=f'{sec} - ΔMem/MB'))
            fig.add_trace(go.Scatter(y=peaks, mode='lines+markers', name=f'{sec} - PeakMem/KB'))
            summary.append(html.Div([
                html.H4(f'Section: {sec}'),
                html.P(f"Runs: {len(data)} | Mean time: {sum(times)/len(times):.6f}s | "+
                       f"Mean peak mem: {sum(peaks)/len(peaks):.2f}KB | Mean Δmem: {sum(mems)/len(mems):.2f}MB")
            ]))
    fig.update_layout(title='Function Performance Metrics', xaxis_title='Run', yaxis_title='Value')
    return fig, summary

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)

