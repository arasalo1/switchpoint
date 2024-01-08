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

## model.ipynb

Jupyter notebook to to replicate the results.

NUTS sampler takes around 6 minute on AMD Ryzen 7.

> [!NOTE]
> If jax import fails due to jaxlib import error, install CPU version
> `pip install --upgrade "jax[cpu]"`