import pandas as pd 
import numpy as np 
import os

path_from = 'images/dump/'

li_imgs = os.listdir(path_from)
df = pd.DataFrame(columns=['fruit_name'])
count = 1
for imgs in li_imgs:
	label = imgs.split('_')[0]
	print(label,'    ',count)
	df = df.append({'fruit_name':label},ignore_index=True)
	count += 1

df.to_csv('labels.csv',sep=',')
