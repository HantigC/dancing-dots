import trimesh
import plotly.graph_objects as go

from mts.viz.plotly.figure import create_new_figure


def mesh_trace(
    mesh: trimesh.Trimesh,
    name: str = "",
    color: str = "blue",
    opacity: float = 0.4,
) -> go.Mesh3d:
    v, f = mesh.vertices, mesh.faces
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=True,
    )


@create_new_figure
def show_meshes(
    meshes: list[tuple[trimesh.Trimesh, str, str]],
    fig: go.Figure | None = None,
) -> go.Figure:
    """meshes: list of (mesh, name, color)"""
    for mesh, name, color in meshes:
        fig.add_trace(mesh_trace(mesh, name=name, color=color))
    fig.update_layout(scene=dict(aspectmode="data"))
    return fig
