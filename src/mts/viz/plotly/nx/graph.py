import networkx as nx

from mts.viz.plotly.figure import create_new_figure
from plotly import graph_objects as go


@create_new_figure
def plot_nodes(
    graph: nx.Graph,
    pos: dict,
    fig: go.Figure = None,
) -> go.Figure:
    node_x = []
    node_y = []
    node_text = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=20, line_width=2),
    )
    fig.add_trace(node_trace)
    return fig


@create_new_figure
def plot_edges(
    graph: nx.Graph,
    pos: dict,
    fig: go.Figure = None,
) -> go.Figure:
    edge_x = []
    edge_y = []
    edge_text = []

    middle_pos_x = []
    middle_pos_y = []

    for u, v, d in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        middle_pos_x.append((x1 + x0) / 2)
        middle_pos_y.append((y1 + y0) / 2)
        edge_text.append(d.get("weight", ""))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        text=edge_text,
        line=dict(width=2),
        hoverinfo="text",
        mode="lines+text",
        marker=dict(symbol="x")
    )

    node_trace = go.Scatter(
        x=middle_pos_x,
        y=middle_pos_y,
        mode="markers",
        text=edge_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=20, line_width=2),
    )
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    return fig


@create_new_figure
def plot_graph(
    graph: nx.Graph,
    title: str = "graph-viz",
    fig: go.Figure = None,
) -> go.Figure:
    pos = nx.spring_layout(graph, seed=42)
    plot_nodes(graph, pos, fig=fig)
    plot_edges(graph, pos, fig=fig)

    return fig
