import plotly.graph_objects as go
from Singlelayer import get_model
import torch.nn as nn

def extract_layer_sizes(model):
    sizes = []
    last = None
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            if last is None:
                sizes.append(layer.in_features)
            sizes.append(layer.out_features)
            last = layer.out_features
    return sizes

def visualize_2d(layers):
    fig = go.Figure()
    max_neurons = max(layers)
    spacing = 200
    for i, num in enumerate(layers):
        for j in range(num):
            x = i * spacing
            y = (max_neurons - num) * 20 + j * 40
            fig.add_shape(type="circle", x0=x-10, y0=y-10, x1=x+10, y1=y+10, line=dict(color="blue"))
            fig.add_annotation(x=x, y=y, text=f"{j+1}", showarrow=False)

    for i in range(len(layers)-1):
        for a in range(layers[i]):
            for b in range(layers[i+1]):
                x0, x1 = i * spacing, (i+1) * spacing
                y0 = (max_neurons - layers[i]) * 20 + a * 40
                y1 = (max_neurons - layers[i+1]) * 20 + b * 40
                fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="gray", width=1))

    fig.update_layout(title="Neural Network - 2D View", xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.show()

def visualize_3d(layers):
    fig = go.Figure()
    for i, num in enumerate(layers):
        for j in range(num):
            fig.add_trace(go.Scatter3d(
                x=[i * 100], y=[j * 100], z=[0],
                mode='markers+text',
                marker=dict(size=6),
                text=[f"{j+1}"],
                showlegend=False
            ))

    for i in range(len(layers)-1):
        for a in range(layers[i]):
            for b in range(layers[i+1]):
                fig.add_trace(go.Scatter3d(
                    x=[i * 100, (i+1) * 100],
                    y=[a * 100, b * 100],
                    z=[0, 0],
                    mode='lines',
                    line=dict(color='gray'),
                    showlegend=False
                ))

    fig.update_layout(title="Neural Network - 3D View",
                      scene=dict(xaxis_title='Layer', yaxis_title='Neuron', zaxis_title='Depth'),
                      margin=dict(l=0, r=0, b=0, t=30))
    fig.show()

if __name__ == "__main__":
    model = get_model()
    layer_sizes = extract_layer_sizes(model)
    print(f"📐 Detected Layers: {layer_sizes}")
    mode = input("Visualize model in 2D or 3D? (type '2D' or '3D'): ").strip().lower()
    if mode == "2d":
        visualize_2d(layer_sizes)
    else:
        visualize_3d(layer_sizes)
