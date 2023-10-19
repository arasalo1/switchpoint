import pandas as pd
import os

paths = ['https://osf.io/download/wvcxa/','https://osf.io/download/hjbmk/']
names = ['processed_both.csv','processed_macro.csv']

for i,j in zip(paths,names):
    data = pd.read_csv(i,index_col=0).to_csv(os.path.join('data',j))