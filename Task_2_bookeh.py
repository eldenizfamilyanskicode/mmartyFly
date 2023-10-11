import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure, output_notebook, show, output_file, save
from bokeh.models import ColumnDataSource, CustomJS, Slider, Label, Legend, CategoricalColorMapper, HoverTool
from bokeh.layouts import column

# Ensure the 'outcomes' folder exists; if not, create it
if not os.path.exists('outcomes'):
    os.makedirs('outcomes')

# Load data
X = np.load("release_gene_analysis_data/data/p1/X.npy")
y = np.load("release_gene_analysis_data/data/p1/y.npy")

# Apply log transformation
X_log = np.log2(X + 1)

# Perform PCA for 100 principal components
pca_100 = PCA(n_components=100)
X_pca_100 = pca_100.fit_transform(X_log)

# Run T-SNE for different learning rates and number of iterations
# learning_rates = [10, 500, 1000]
learning_rates = []

for lr in range(10, 1001, 50):
    learning_rates.append(lr)
    

# n_iters = [250, 1000, 2000]
n_iters = []

for n_iter in range(250, 2001, 250):
    n_iters.append(n_iter)

tsne_results = {}
kmeans_results = {}

# Run T-SNE and K-means experiments
for lr in learning_rates:
    for n_iter in n_iters:
        tsne = TSNE(n_components=2, learning_rate=lr, n_iter=n_iter)
        X_tsne = tsne.fit_transform(X_pca_100)
        
        # Store T-SNE results
        key = f"lr_{lr}_n_iter_{n_iter}"
        tsne_results[key] = X_tsne.tolist()
        
        # Run K-means clustering
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(X_tsne)
        
        # Store K-means results
        kmeans_results[key] = kmeans.labels_.tolist()

# Initialize Bokeh plot
source = ColumnDataSource(data=dict(x=tsne_results['lr_10_n_iter_250'], y=tsne_results['lr_10_n_iter_250'], label=kmeans_results['lr_10_n_iter_250']))

# Create a new column in source data to hold colors based on cluster label
colors = [Category10[5][i] for i in kmeans_results['lr_10_n_iter_250']]
source.add(colors, 'color')

# Initialize plot
p = figure(title="T-SNE with Variable Hyperparameters", tools="pan,box_zoom,reset,hover")
hover = p.select(dict(type=HoverTool))
hover.tooltips = [("Cluster", "@label"), ("(x,y)", "($x, $y)")]

# Add scatter plot
scatter = p.scatter('x', 'y', legend_field='label', source=source, fill_alpha=0.6, size=7, fill_color='color', line_color='color')

# Add axis labels
p.xaxis.axis_label = "T-SNE Component 1"
p.yaxis.axis_label = "T-SNE Component 2"

# JavaScript callback to update plot based on slider values
callback = CustomJS(args=dict(source=source, tsne_data=tsne_results, kmeans_data=kmeans_results, colors=Category10[5]), code="""
    const data = source.data;
    const selected_lr = lr_slider.value;
    const selected_n_iter = n_iter_slider.value;
    const key = 'lr_' + selected_lr + '_n_iter_' + selected_n_iter;
    data['x'] = tsne_data[key].map(d => d[0]);
    data['y'] = tsne_data[key].map(d => d[1]);
    data['label'] = kmeans_data[key];
    data['color'] = kmeans_data[key].map(label => colors[label]);
    source.change.emit();
""")

# Create sliders
lr_slider = Slider(start=10, end=1000, value=10, step=50, title="Learning Rate")
n_iter_slider = Slider(start=250, end=2000, value=250, step=250, title="Number of Iterations")

# Attach callback to sliders
lr_slider.js_on_change('value', callback)
n_iter_slider.js_on_change('value', callback)

# Add sliders to callback arguments
callback.args["lr_slider"] = lr_slider
callback.args["n_iter_slider"] = n_iter_slider

# Add sliders to layout and save the plot
layout = column(lr_slider, n_iter_slider, p)
output_file("outcomes/Hyperparameter_Exploration.html")
save(layout)
