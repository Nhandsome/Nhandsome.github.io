---
layout: post
title:  "[Paper] Density Matching for Bilingual Word Embedding"
subtitle:   "Subtitle - Density Matching for BWE"
categories: dev
tags: nlp
comments: true
---
## Describe
> [`Density Matching for Bilingual Word Embedding`](https://arxiv.org/abs/1904.02343)을 읽고 정리<br>
한국어 韓国語

## 目次
- [어떤 내용인가?](#jump1)
- [전체적인 개념잡기](#jump2)
- [구체적인 모델](#jump3)
  - [기본이 되는 개념](#jump4)
  - [Multilingual Adversarial Training(MAT)](#jump5)
  - [Multilingual Pseudo-Supervised Refinement(MPSR)](#jump6)
- [정리](#jump7)

<br><br><br>

## <a name="jump1">어떤 내용인가?</a>
　서로 다른 Word Embedding 공간을 하나의 공간에 매칭시키는 Cross Word Embedding은, 주로 개별적으로 학습된 Word Embedding 공간을 
- 수십~수천개의 Dictionary 단어쌍을 기준으로 매칭시키거나(Supervised)
- 서로 다른 언어라도 자주 사용되는 단어들은 비슷한 분포를 가진다는 성질을 이용,
- Dictionary 단어쌍을 임의로 추측하여 매칭을 진행(Unsupervised) 방식을 취한다.<br>

　하지만 모든 언어에 대하여 학습모델이 잘 작동하는 것은 아니다.(Unsupervised 기준), 이는 Embeeding 공간을 단어들의 모임(Set of discrete points)으로 여기기 때문이며, `Embedding 공간에 내제된 불확실성을 고려하지 못한다.`고 지적하고 있다. <br>

> 논문에서는, **Gausian Mixture Model**을 이용하여 `Embedding 공간을 확률밀도로 변환`,  `Normalizing flow`를 사용하여 두 확률밀도를 매칭시킴으로써, 이 문제를 해결하려고 한다.

<br><br><br>


## <a name="jump2">전체적인 개념잡기</a>
　이 논문에서 제안하는 Method는 `Density Maching for Bilingual Word Embedding(DeMa-BWE)`을 구현하기 위해서 먼저, 학습된 Word Embedding 공간을 확률밀도로 변환하는 과정이 필요하다. Word Embedding 공간에 등장하는 단어 하나하나를 **가우시안 분포의 정점으로 변환하고, 그 정점의 높이를 단어의 출현빈도에 따라 가중치**를 주는 것으로, Gaussian Mixture Model로 확률밀도 분포를 생성한다.<br>

　이렇게 생성된 2개의 분포를 매칭시키는데, 이를 위해서 
- **분포의 변수변환을 이용한 Loss**
- **Back-translation Loss**
- **Weak Supervised Pair Loss**<br>
을 이용한다. (뒤에서 Weak Supervised Pair Loss이 성능에 어떤 영향을 미치는지 확인해볼 필요가 있을 것 같다.) <br>

　하나하나의 단어가 아니라 확률밀도로 Word Embedding 공간을 표현함으로써, **'비슷한 개념의 단어는 존재하지만 정확한 매칭이 안되는 단어'**들의 관계를 해소하고 있는 것이 아닌가...생각이 든다.

<br><br><br>

## <a name="jump3">구체적인 모델</a>

### <a name="jump4">**Notation과 기초가 되는 수식**</a>
  - $$X, Y \in \Bbb{R}^d$$ : **Source and Target Word Embedding 공간의 벡터집합**
  - $$x_i, y_i$$ : **X, Y에서의 실제 단어**
  - $$f_{xy}(\cdot)=W_{xy}, f_{yx}(\cdot)=W_{yx}$$ : **Mapping function과 Matrix**

Word Embedding 공간의 확률밀도에 관하여,
  - $$z, u$$ : **Source and Target 공간의 임의의 벡터**
  - $$p_\theta(z) = N(x;~0,I)$$ : **z에 대한 Prior Distribution**, 가우시안 분포

을 나타내고 있으며, 확률밀도의 변수변환에 의해서<br>
$$z \thicksim p_\theta(z), ~u = f_{xy}(z)$$<br>
$$p_\theta(u) = p_\theta(z)|det(J(f^{-1}(u)))|$$

  > $$T$$(Target Space)를 통해서 원하는 언어$$S_i$$로의 Mapping이 가능하며, `$$M_i$$는 $$T$$로의 Encoding, $$M_i^T$$는 $$T$$로부터의 Decoding 연산`으로 생각할 수 있다. 

   <br><br>

Pipeline

There are some layers in Tensorflow and each of them have specific funtions

Tensor : N-dimensional array of data
- tf.constant produces constant tensors and tf.Variable tensors can be modified.
- stock, slice, reshape : change tensor dimensions
- tf.Variable : The values can be changed by tf.Variable.assign
  - w = tf.Variable (modified during training)
  - x = tf.constant
  - tf.matmul(w,x)

#link to github