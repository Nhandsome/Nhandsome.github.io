---
layout: post
title:  "[DataScience] EDAでのデータ可視化"
subtitle:   "Subtitle - Data visualization tips (EDA)"
categories: dev
tags: data
comments: true
---
## Describe
> データの特徴によるEDA(Exploratory Data Analysis)方法を整理、　日本語<br>

## 目次
- [Intro](#jump1)
- [Numeric and Numeric Variable](#jump2)
  - [季節・日にちなどの変化によるトレンドが知りたい時](#jump3)
  - [二つのContinuous Variableの関係が知りたい時](#jump4)
- [Numeric and Categorical Variable](#jump5)
  - [Category基準によるContinueous分布が知りたい時](#jump6)
  - [Category基準によるContinueous分布が知りたい時(大量のデータ)](#jump7)
- [Categorical and Categorical Variable](#jump8)
  - [Category基準によるCategorical分布が知りたい時](#jump9)


<br><br><br>

## <a name="jump1">Intro</a>
- [KaggleのData Visualizationコース](https://www.kaggle.com/alexisbcook/hello-seaborn)
- [Kaggleのプロジェクト](https://github.com/Nhandsome/for_kaggle)
- 学校のデータサイエンスコース

で様々なデータ分析を行いながら、データの可視化に関して気付いたこと・覚えて置きたいことを整理してみる。 データの特徴によってどのような分析・グラブが必要であるかを主に扱おうとする。
<br><br><br>


## <a name="jump2">Numeric and Numeric Variable</a>
### <a name="jump3">季節・日にちなどの変化によるトレンドが知りたい時</a>
  **LinePlot**<br>
- X(Date)の変化によるY(Data)への影響が見やすい。
```python
plt.figure(figsize=(10,5))
sns.lineplot(data=df,x="Date",y='Data')
```
![EDA1](/assets/img/dev/eda_2.png)
<br><br>


### <a name="jump4">二つのContinuous Variableの関係が知りたい時</a>
  **ScatterPlot**<br>
- [Medical cost dataset](https://www.kaggle.com/mirichoi0218/insurance)
- 喫煙者のX(BMI数値)とY(治療費)の関係が分かりやすい。
```python
plt.figure(figsize=(10,5))
plt.title('BMI - CHARGES SCATTER PLOT',fontsize=20,color='gray')
sns.scatterplot(data=temp,x='bmi',y='charges')
plt.show()
```
![EDA２](/assets/img/dev/eda_1.jpg)
<br>

  **ScatterPlot with Hue**<br>
- 喫煙者・非喫煙者の比較して見ることもできる
- hue parameterは比較の基準になる`Categorical Variable Column`になる。２−３個ぐらいのcategoryに分けれる方が良さそう。
```python
plt.figure(figsize=(10,5))
plt.title('LABELED SCATTER PLOT WITH REGRESSION LINE',fontsize=20,color='w')
##Smokerは Yes or No 二つのCategoryに分けられているColumn
sns.scatterplot(data=df_temp,x='bmi',y='charges',hue='smoker')
```
![EDA3](/assets/img/dev/eda_3.jpg)
<br><br><br>

## <a name="jump5">Numeric and Categorical Variable</a>
### <a name="jump6">Category基準によるContinueous分布が知りたい時</a>
  **SwarmPlot**<br>
- X(喫煙状況)基準によるY(治療費)の分布を比較しやすい。
- [二つのContinuous Variableの関係が知りたい時](#jump4)で扱ったScatterPlotをより明確に確認する。
```python
plt.figure(figsize=(5,10))
plt.title('SWARM PLOT_SMOKER_CHARGES',fontsize=20,color='w')
sns.swarmplot(data=df_temp,x='smoker',y='charges')
```
![EDA4](/assets/img/dev/eda_4.jpg)
<br><br>

### <a name="jump7">Category基準によるContinueous分布が知りたい時（大量のデータ）</a>
  **BoxPlot**<br>
- 一般的に使われる方法。
- [Category基準によるContinueous分布が知りたい時](#jump６)よりグラフ生成が早く、median・分布などが見やすい。
```python
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.boxplot(data=train_df, x=sta_dist, y=TARGET)
plt.subplot(1,2,2)
sns.stripplot(data=train_df, x=sta_dist, y=TARGET)
```
![EDA9](/assets/img/dev/eda_9.png)
<br><br><br>


## <a name="jump8">Categorical and Categorical Variable</a>
### <a name="jump9">Category基準によるCategorical分布が知りたい時</a>
  **BarChart**<br>
- [Titanic Dataset](https://www.kaggle.com/c/titanic)
- X(Survived/Dead)基準によるY(座席のClass)の分布を比較しやすい。
```python
def bar_char(feature):
  ##Survived Category Columnを基準にあるFeature情報を抽出
  temp_s = train[train['Survived']==1][feature].value_counts()
  temp_d = train[train['Survived']==0][feature].value_counts()
  df = pd.DataFrame([temp_s, temp_d])
  df.index=['Survuved', 'Dead']
  ##Stock optionで分布を比較
  df.plot(kind='bar',stacked=True,figsize=(10,5))

  bar_char('Pclass')
```
![EDA4](/assets/img/dev/eda_5.jpg)
<br><br>

  **Tips for BarChart : Groupby**<br>
- Group : Categorical Variable Column Name, X-axisになる比較基準
- Data : Categorical Variable Column Name, 比較したいデーター
- df pandas DataFrameから簡単に``[Group, Data]に関するBarChartを生成``。
```python
  temp_1 = df.groupby(['Group','Data']).Data.count()[1]
  temp_2 = df.groupby(['Group','Data']).Data.count()[2]
  temp_df = pd.DataFrame([temp_1,temp_2])
  temp_df.index = ['Group1','Group2']
  temp_df.fillna(0, inplace=True)

  temp_df.plot(kind='bar',stacked=True,legend=True,figsize=(10,5),title='Relationship between Data1 and Data2',cmap='Pastel2')
  plt.show()
```
![EDA5](/assets/img/dev/eda_6.png)
<br><br>


  **Tips for BarChart : 比率で比較**<br>
- Group1, Gruop2のスケールが違う時、`分布を相対的な比率で比較する`。
  - Group Category中、Data Categoryの全般的な分布がわかる。

```python
  Group_N = 'X_axis_Col_Name'
  Data_N = 'Y_axix_Col_Name'

  temp_1 = df.groupby([Group_N,Data_N]).Data.count()[1]
  temp_2 = df.groupby([Group_N,Data_N]).Data.count()[2]
  temp_df = pd.DataFrame([temp_1,temp_2])
  temp_df.index = ['Group1','Group2']
  temp_df.fillna(0, inplace=True)
  ## Percentage 情報に変える
  temp_df = temp_df.apply(lambda row: row*100/sum(row), axis=1)

  temp_df.plot(kind='bar',stacked=True,legend=True,figsize=(10,5),title='Relationship between Data1 and Data2',cmap='Pastel2')
  plt.show()
```
![EDA7](/assets/img/dev/eda_7.png)
<br><br>

> EDAの基本になるグラフの描き方であるが、確認したいこと・使用するVariableの週類によってその分析方法が変わるのを理解しよう。