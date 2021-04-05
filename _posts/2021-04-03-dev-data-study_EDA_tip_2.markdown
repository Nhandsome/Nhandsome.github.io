---
layout: post
title:  "[DataScience] EDAでのデータ可視化②"
subtitle:   "Subtitle - Data visualization tips 2(EDA)"
categories: dev
tags: data
comments: true
---
## Describe
> Nishika、中古マンション価格予測Competitionに参加して行ったEDAの一部を整理、　日本語<br>

## 目次
- [Intro](#jump1)
- [NumericとNumericの関係 : HEAT MAP](#jump2)
- [HEAT MAPの結果を用いたSCATTER PLOT](#jump3)
- [SCATTER PLOTの遊び](#jump4)


<br><br><br>

## <a name="jump1">Intro</a>
- [Nishika　中古マンション価格予測](https://www.nishika.com/competitions/11/summary)のデータを使用

日本のKaggleを作ろうとしているサイトがあると聞き、短い時間だったがCompatitionに参加してみた。<br>EDA整理のついでに使ってみたグラフを紹介する。
<br><br><br>


## <a name="jump2">NumericalとNumericalの関係 : HEAT MAP</a>
  **HEAT MAP**<br>
- 色を使って、２次元空間に、様々な情報を可視化する
- ここでは「マンションの値段・面積・建築年」など、Numericalデータの相関関係(Correlation)を見易くグラフを作る。

```python
## HEAT MAP
f, ax = plt.subplots(figsize=(12, 9))
## train_dfのNumeric Variableに関する相関関係を求める
mat = train_df.corr('pearson')
## Matrixは対称なので、Uppper Triangle部分をマスキングする
mask = np.triu(np.ones_like(mat, dtype=bool))
## 色の設定
cmap = sns.diverging_palette(230, 20, as_cmap=True)
## Heat Map生成
sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
```

![EDA8](/assets/img/dev/eda_8.png)

> 様々な関係が確認できるが、主に「取引価格_log」との関係性に注目したい。「面積・建築年・駅からの距離」がマンションの値段とある程度の相関関係を持っているように見える。<br>実際の分析結果、``面積と建築年はFeature Importanceが高かった。``しかし、**建設年より「建設年ー取引年」の方が値段の予測に役に立つこと**など、単純な判断より深い分析が必要になる。

<br><br>


## <a name="jump3">HEAT MAPの結果を用いたSCATTER PLOT</a>
- [HEAT MAP](jump2)から意味があると思われるVariableの関係を可視化する。
- Scatter Plot・RegPlotを使うと、Numerical Variable間の関係が見やすい。

```python
## TARGET(取引価格)と建築年
plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
sns.scatterplot(data=train_df, x=built_year, y=TARGET, alpha=0.2)
plt.title('Built year vs TARGET',size=15)
## TARGET(取引価格)と面積
plt.subplot(3,1,2)
sns.scatterplot(data=train_df, x=area, y=TARGET, alpha=0.2)
plt.title('Area vs TARGET',size=15)
## TARGET(取引価格)と駅からの距離
plt.subplot(3,1,3)
sns.scatterplot(data=train_df, x=sta_dist, y=TARGET, alpha=0.2)
plt.title('Distance from station vs TARGET',size=15)

plt.show()
```

![EDA10](/assets/img/dev/eda_10.png)

> マンションの取引価格がlogになっているし、その分散が大きいので見辛いこともあるが、傾向性はある。

<br><br>

## <a name="jump4">SCATTER PLOTの遊び</a>
前回[[DataScience] EDAでのデータ可視化](https://nhandsome.github.io/dev/2021/02/19/dev-data-study_EDA_tip_1/)で紹介したものだが、seabornを使う時**hueオプションを上手く利用する**ことで、もう一つのCategorical情報が可視化できる。

- 東京都のマンションは高くないかな
- 東京都のマンションを分けてPLOTしてみよう

```python
## 東京都を表す新たなColumn(is_tokyo)を作る
is_tokyo = train_df['都道府県名']=='東京都'
train_df['is_tokyo'] = is_tokyo
## 軽く動かすために、5000個のRowをサンプリングする。
temp_df = train_df.sample(n=5000)
## hueオプションを用い、「東京都のマンションである」情報を可視化する
plt.figure(figsize=(10,15))
plt.subplot(3,1,1)
sns.scatterplot(data=temp_df, x=built_year, y=TARGET, alpha=0.2, hue='is_tokyo')
plt.title('Built year vs TARGET',size=15)

plt.subplot(3,1,2)
sns.scatterplot(data=temp_df, x=area, y=TARGET, alpha=0.2, hue='is_tokyo')
plt.title('Area vs TARGET',size=15)

plt.subplot(3,1,3)
sns.scatterplot(data=temp_df, x=sta_dist, y=TARGET, alpha=0.2, hue='is_tokyo')
plt.title('Distance from station vs TARGET',size=15)

plt.show()
```
![EDA11](/assets/img/dev/eda_11.png)

> 東京都のマンションは他の地域より高い取引価格を形成している。