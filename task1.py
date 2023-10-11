import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_notebook, show, output_file, save
from bokeh.models import ColumnDataSource, CustomJS, Slider, Label
from bokeh.layouts import column

# Ensure the 'outcomes' folder exists; if not, create it
if not os.path.exists('outcomes'):
    os.makedirs('outcomes')

# Load data
X = np.load("release_gene_analysis_data/data/p1/X.npy")
y = np.load("release_gene_analysis_data/data/p1/y.npy")

# Apply log transformation
X_log = np.log2(X + 1)

# Perform PCA for 500 principal components
pca_500 = PCA(n_components=500)
X_pca_500 = pca_500.fit_transform(X_log)

# Initialize dictionary to store T-SNE results
X_tsne_dict = {}

list_of_n_components = []
for i in range(10, 501, 10):
    list_of_n_components.append(i)

# Run T-SNE for various numbers of principal components
for n_components in list_of_n_components:
    X_reduced = X_pca_500[:, :n_components]
    tsne = TSNE(n_components=2, perplexity=40)
    X_tsne = tsne.fit_transform(X_reduced)
    X_tsne_dict[n_components] = X_tsne.tolist()   # Convert numpy array to list for compatibility with Bokeh

# Prepare data for Bokeh
x_values = [item[0] for item in X_tsne_dict[10]]
y_values = [item[1] for item in X_tsne_dict[10]]
source = ColumnDataSource(data=dict(x=x_values, y=y_values))


# Create figure
p = figure(title="T-SNE using varying number of PCs", tools="pan,box_zoom,reset")
p.scatter('x', 'y', source=source)
label = Label(x=70, y=70, text=str(10), text_font_size='12pt')
p.add_layout(label)

# JavaScript callback to update data source
callback = CustomJS(args=dict(source=source, label=label, tsne_data=X_tsne_dict), code="""
    // Debugging
    console.log('Current n_components:', cb_obj.value);
    console.log('Available tsne_data:', tsne_data);

    const n_components = cb_obj.value;
console.log('Data for current n_components:', tsne_data.get(n_components));

if(tsne_data.has(n_components)) {
    label.text = n_components.toString();
    const new_data = tsne_data.get(n_components);
    source.data['x'] = new_data.map(row => row[0]);
    source.data['y'] = new_data.map(row => row[1]);
    source.change.emit();
} else {
    console.error('Data for current n_components is undefined.');
}

""")


# Create slider
slider = Slider(start=10, end=500, value=10, step=10, title="Number of PCs")
slider.js_on_change('value', callback)

# Display in notebook
output_notebook()
layout = column(slider, p)

# Save the layout
output_file("My_dashboard.html")
save(layout)

# # Display in notebook
# show(layout)
