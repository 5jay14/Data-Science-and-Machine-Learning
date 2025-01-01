import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\penguins_size.csv")

df = df.dropna()
print(df)
print(df.isnull().sum())
print(df['sex'].unique())
print(df[df['sex'] == '.'])
print(df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose())  # data leans towards female
df = df.at[336, 'sex'] == 'Female'  # reassignment of value, this creates a copy and does not change the original data frame
sns.pairplot(df, hue='species')
plt.show()

sns.catplot(df, x='species',y = 'culmen_length_mm',kind='box', col='sex')
plt.show()