# Mental Health in Tech – employee segmentation (IU case study DLBDSMLUSL01, Task 1)

Unsupervised analysis of the OSMI "Mental Health in Tech Survey 2016": cleaning, feature engineering,
PCA, k-means / GMM / Ward clustering, validation (silhouette, BIC, bootstrap ARI), 2D views (PCA, MDS, LLE)
and cluster profiles.

## Data
Download `mental-heath-in-tech-2016_20161114.csv` from https://www.kaggle.com/osmi/mental-health-in-tech-2016
(not redistributed here).

## Run
```
pip install -r requirements.txt
python mhtech_clustering.py <path/to/mental-heath-in-tech-2016_20161114.csv> <output_folder> 4
```
The last argument is the number of clusters (k = 4 in the study). The output folder must exist.
Outputs: `fig1..fig4*.png`, `cluster_profiles.csv`, `results.json`. Random seed 42.
