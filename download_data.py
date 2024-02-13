import pandas as pd
import os

paths = ['https://osf.io/download/shax4/','https://osf.io/download/t62aq/',
         'https://osf.io/download/uwjas/']
names = ['processed_both.csv','processed_macro.csv','revision_macro_collagen2_temperature20.csv']

for i,j in zip(paths,names):
    data = pd.read_csv(i,index_col=0).to_csv(os.path.join('data',j))