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
    categorical_feats = ['cholesterol', 'gluc', 'active', 'alco', 'overweight', 'smoke']
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=categorical_feats)


    # 6

    # 7



    # 8
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind="count")


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df
    df_heat[df_heat['height'] >= df_heat['height'].quantile(0.025)]
    df_heat[df_heat['height'] <= df_heat['height'].quantile(0.975)]
    df_heat[df_heat['weight'] >= df_heat['weight'].quantile(0.025)]
    df_heat[df_heat['weight'] <= df_heat['weight'].quantile(0.975)]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr))



    # 14
    fig = plt.figure(figsize=(12,6))
    # 15
    sns.heatmap(corr, mask=mask, annot=True, square=True)


    # 16
    fig.savefig('heatmap.png')
    return fig
