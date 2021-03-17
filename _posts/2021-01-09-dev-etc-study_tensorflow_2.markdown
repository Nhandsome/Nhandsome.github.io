---
layout: post
title:  "[Study] Introduction to Tensorflow_2"
subtitle:   "Subtitle - Coursera, Intoroduction to Tensorflow_2"
categories: dev
tags: nlp
comments: true
published: false
---
## Describe
> Summarizing Coursera lectures, [`Introduction to Tensorflow`](https://www.coursera.org/learn/intro-tensorflow/home/welcome)<br>

## 目次
- [Week 3:Keras Sequential API](#jump1)
  - [Load CSV](#jump2)
  - [Data Preprocessing](#jump3)
<br><br><br>

## <a name="jump1">Week 1</a>
　**How to make/train/test a model with Keras**<br>
　We can create Tensorflow models layer-by-layer with Keras Sequential API, but not models having
- shared layers
- re-use layers
- multi inputs or outputs
<br><br><br>


## <a name="jump2">Load CSV</a>
　**Read data from CSV file**<br>
　We need to 
- name all columns
- set a label column
- set default values for columns we want to use

```python
ALL_COL = ['a', 'b', 'c', 'lab', 'useless']
LABEL_COL = 'lab'
DEFAULTS = [[0.0], [0.0], [0.0], [0.0], ['na']]

## tf.data.experimental.make_csv_dataset API
## if mode == 'train', dataset is requested to be shuffled
## 
def create_dataset(pattern, batch_size=1, mode='eval'):
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)

## features_and_labels returns features and label lists
    dataset = dataset.map(features_and_labels)

    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    return dataset

```

<br><br><br>

## <a name="jump3">Load CSV</a>
　**Read data from CSV file**<br>
　We need to 
- name all columns
- set a label column
- set default values for columns we want to use

```python
```
> We can get dataset from CSV files by using get_dataset() function. Each dataset is
- having 5 data (batch_size)
- using LABEL_COL as the target (label_name)
- recognizing '?' as NA data
- [`and so on`](https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)

This function will use all available columns and the names from CSV file, but if we need to use specific columns then can use 'column_names' and 'select_columns' option.

```python
SELECTED_COLUMNS = ['target','age','sex','class']
temp_dataset = get_dataset(train_file_path, 
  selected_columns = SELECTED_COLUMNS)

## Take n numbers of dataset
n = 5
temp_dataset.take(n)
```

<br><br><br>

### <a name="jump4">Data Preprocessing</a>
　A CSV file can contain variable types of data, so we need to convert from those mixed types to a fiexed length vectors before feeding the data into our model.

**Continuous Data**
Need to 
- set Default values
- pack them into a single column
- normalize

```python
## Select countinous data columns
SELECTED_COLUMNS = ['target','age','n_siblings_spouses','fare']

## set DEFAULTS values and make dataset function
DEFAULTS = [0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path, 
  selected_columns = SELECTED_COLUMNS, 
  column_defaults = DEFUALTS)

## pack function will pack all the continuous columns
## tf.stack
def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

## We can get n numbers dataset continuous data are packed
packed_dataset = temp_dataset.map(pack)
n = 1
packed_dataset.take(n)
```

Here a PackNumericFeatures class for handling numeric columns
```python
class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

## Select numeric feature columns
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

## Pack all of numeric feature and save in 'numeric' column
packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))
```

And continous data should be normalized
```python
import pandas as pd
## We could get information of dataset description from pd.describe()
## from whole of data
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

## Make normalize function
def normalize_numeric_data(data, mean, std):
  return (data-mean)/std

##partial : when use a function with specific inputs
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

## Make 'numeric' feature column 
## tf.feature_columns.numeric_column API will be run on each batch with normalizer_fn option
## it has information about MEAN and STD and normalize data of each batch
numeric_column = tf.feature_column.numeric_column('numeric', 
  normalizer_fn=normalizer, 
  shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

## tf.keras.layers.DenseFeatures produces a dense tensor following given columns
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()
```

<br><br>

**Categorical Data**
Categorical data columns have contents which should be a one of set of options. 

```python
## set categorical data columns
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third']
}

## create categorical_columns that will be feature columns of tensorflow model
categorical_columns = []

## make lists having categorical data information with tf.feature_column 
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

## tf.keras.layers.DenseFeatures creates a dense tensor based on given columns
categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
categorical_layer(example_batch)
```

<br><br>

**Combined preprocessing layer**
We created two feature column collections above and can make input layer that will extract and preprocess both input types

```python
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
preprocessing_layer(example_batch).numpy()[1]

## preprocessed dataset using 2nd batch data, such as
## ['sex','class','age','n_siblings_spouses','parch', 'fare']
## [1,0,22,1,34,43]
```