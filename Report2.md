### Report for the Second Task

## Introduction

In this study, we investigated the impact of three distinct hyperparameters—learning rate and number of iterations for T-SNE, as well as the number of principal components for PCA—on the quality of data clustering and visualization.

## Methodology

Experiments were carried out with varying values for each of these hyperparameters. The outcomes were analyzed both visually and through the use of numerical metrics such as cluster inertia.

## Observations and Results

### T-SNE Learning Rate

- **Impact on Inter-cluster Distance**: Increasing the learning rate leads to an increase in the distance between clusters.
- **Anomalies**: At higher learning rate values, increased clustering is observed, and even alignment along lines occurs.

### T-SNE Number of Iterations

- **Cluster Stability**: Increasing the number of iterations leads to an increase in the stability of the clusters.
- **Outliers**: Increasing the number of iterations does not significantly affect the appearance of outliers.

### Impact of Number of Principal Components on Clustering

- **Inertia**: The inertia value increases with the number of principal components.
- **Visual Changes**: No significant visual changes in clusters were observed.

## Conclusions

1. **Learning Rate**: The optimal learning rate value is around 200 for obtaining the most distinct clusters. Lower values result in a higher number of anomalies.
   
2. **Number of Iterations**: Increasing the number of iterations improves both the stability and density of the clusters. The optimal value depends on the specific case.

3. **Number of Principal Components**: Changing the number of principal components affects the inertia of the clusters but does not lead to significant visual changes.

These findings underscore the importance of careful hyperparameter selection when analyzing data using visualization and clustering methods.

