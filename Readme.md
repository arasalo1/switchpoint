# Analysis codes for Measuring mechanical cues for modeling stromal matrix in 3D cell culture


## Requirements

Download the necessary python packages. One option is to create a new conda environment

```
conda create --name switchpoint python=3.10.4
conda activate switchpoint
conda install -c anaconda pip
```


GPU versions are not needed for tensorflow-probability and jax.

```
pip install -r requirements.txt
```

## Data
Data is available at https://osf.io/7s4zn/

You can download the necessary files with 
```
python download_data.py
```

## linear.ipynb

Jupyter notebook to to replicate the results for the phase angle analysis.

NUTS sampler takes around 1 minute on AMD Ryzen 7.

## switchpoint.ipynb

Jupyter notebook to replicate the results for the absolute complex shear modulus analysis.

- This notebook also contains the visualizations for the joint effect sizes and heterogeneity comparisons. These require running the phase angle analysis (linear.ipynb) first and saving the results. These are posterior samples and the predictive heterogeneity results.

NUTS sampler takes around 3 minutes on AMD Ryzen 7.