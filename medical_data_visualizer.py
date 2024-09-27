import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = 0
df['height'] = df['height']/100
df['bmi'] = df.weight/(df.height**2)
df.loc[df['bmi'] > 25, 'overweight'] = 1

# 3
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

# 4
def draw_cat_plot(): 
    # 5
    categorical_feats = ['active', 'alco', 'cholesterol', 'gluc',  'overweight', 'smoke']
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=categorical_feats)
    # 6
    df_cat.rename(columns={'value': 'Value'}, inplace=True)
    # 7
    fig, ax = plt.subplots(figsize=(12,6))
    catplot = sns.catplot(x='variable', hue='Value', col='cardio', data=df_cat, kind="count")
    # 8
    catplot.set_axis_labels("variable", "total")
    fig = catplot.fig


    
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df
    df_heat = df_heat[['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']]
    df_heat = df_heat[
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    # 2. Calculate the correlation matrix
    corr = df_heat.corr()

    # 3. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 4. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # 5. Plot the correlation matrix using seaborn's heatmap

    sns.heatmap(corr, mask=mask, annot=True,fmt='.1f', square=True, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig
