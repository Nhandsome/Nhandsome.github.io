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
  - $$f_{xy}(\cdot)=W_{xy},~f_{yx}(\cdot)=W_{yx}$$ : **Mapping function과 Matrix**
<br>
Word Embedding 공간의 확률밀도에 관하여,
<br>
  - $$z, u$$ : **Source and Target 공간의 임의의 벡터**
  - $$p_\theta(z) = N(x;~0,I)$$ : **z에 대한 Prior Distribution**, 가우시안 분포
<br>
을 나타내고 있으며, 확률밀도의 변수변환에 의해서
<br>
$$z \thicksim p_\theta(z), ~u = f_{xy}(z)$$<br>
$$p_\theta(u) = p_\theta(z)|det(J(f^{-1}(u)))|$$

  > 확률변수 X, Y 의 관계를 알고 있을 때, `X의 확률밀도함수를 Y에 관한 확률밀도함수로 변환`할 수 있다. 
  이 개념을 고차원 벡터를 변수로 하는 공간에 적용한 것으로 이해하면 되겠다. 함수 f는 역함수가 존재한다고 가정하였다.

   <br><br>

### <a name="jump5">**Multilingual Adversarial Training(MAT)**</a>
  　Adversarial에서 느껴지는 Generator와 Discriminator의 냄새.. 어떤 언어에서 출발했느냐와는 상관없이, $$T$$(Target Space)를 통해서 들어온 단어인지, 원래 존재하던 단어인지만 판단하면 되기 때문에`(Pivot Space를 사용했기 때문에), 언어 수 만큼을 Discriminator만 필요로 한다.`　Discriminator의 목적함수를 살펴보면,<br><br>
  ![式](/assets/img/dev/umwe_1.jpg)
  - $$J\scriptsize D\tiny i$$ : **목적함수**, 이 식 전체를 최적화(최소/최대)하는 D를 학습하고 싶다.
  - $$D_j$$ : **언어$$_j$$의 Discriminator**, Binary Classfier이다. 즉, 진짜 언어$$_j$$의 단어라면 1, 뭔가 만들어진 수상한 단어라면 0을 출력하는 함수인데, 이 판단의 성능을 학습해간다.
  - $$L_d$$ : **손실함수(Loss Function)**, [Cross Entropy Loss](https://deeplearningdemystified.com/article/fdl-3)를 사용하며, $$D_j$$가 예측을 잘하면 0에 가까운 값을 출력한다.
  - $$L_d(1,D_j(x_j))$$ : 실제 언어$$_j$$의 단어($$x_j$$)를, $$D_j$$가 '진짜(1)'라고 잘 판단하는가? 
  - $$L_d(0,D_j(M_j^T  M_i x_i))$$ : 언어$$_i$$가 j에 Mapping된 단어($$M_j^T  M_i x_i$$)를, $$D_j$$가 '가짜(0)'라고 잘 판단하는가?

  > N개의 언어는 각자 Discriminator를 하나씩 가지고 있고, `'외부에서 들어오는 단어'와 '내부의 단어'를 구분하는 능력`을 학습한다. 한 번의 Iteration으로 모든 Discriminator를 한 번씩은 학습(for j in N)하며, i는 랜덤으로 선택된다.
  
  <br>

  　Discriminator가 학습되어지고 있으니, Generator($$M_i$$)들도 경쟁하기위해 학습을 진행해야한다.<br><br>
  ![式](/assets/img/dev/umwe_2.jpg)
  - $$J\scriptsize M\tiny i$$ : **목적함수**, 이 식 전체를 최적화(최소/최대)하는 M을 학습하고 싶다.
  - $$M_j^T  M_i x_i$$ : **2개의 Generator를 사용한 가짜 언어$$_j$$**, 모든 언어가 공유하는 Target Space를 거쳐 언어$$_j$$를 만들어본다.
  - $$L_d(1,D_j(M_j^T  M_i x_i))$$ : 위에서 학습된 $$D_j$$가, $$M_j^T  M_i x_i$$를 '진짜(1)'로 판단하도록 잘 속이고 있는가?

  > `Discriminator를 속이기 위한 2개의 Generator를 학습`힌다. 한 번의 Iternation에 적어도 한 번씩은 학습이 진행되며(for i in N 그리고 random(j)), i = j가 되는 경우도 존재해, 이 경우는 [Cycle GAN](https://junyanz.github.io/CycleGAN/)의 모양새가 된다.

  <br><br>

### <a name="jump6">**Multilingual Pseudo-Supervised Refinement(MPSR)**</a>
  　`CSLS와의 관계 설명이 틀렸을 수도 있습니다. 논문만으로는 조금 애매한 관계로, 코드를 보고 틀린 부분이 있다면 다시 수정하겠습니다.`<br>
  　Unsupervised Cross-lingual Word Embedding과정에서의 Refinement는, `학습의 중간점검/튜닝`과정인 듯 하다. Unsupervised이기 때문에, 확신이 없는(뜻이 전혀 다른)단어들도 '마치 자기 pair인 마냥 학습되어져 있는 경우'가 존재하며, 이를 해소하기 위한 과정이다. <br>
  　이 논문에서도 그렇고 다른 논문에서도 자주 쓰이는 방법인데, **사용빈도가 높은 단어들 위주로 Cross-Lingual Similarity Scaling(CSLS)을 실시,** 나름 자신이 있는 Pair를 위주로 매 학습시 Mapping을 조금씩 조절한다. 먼저 CSLS를 살펴보면,<br><br>
  ![式](/assets/img/dev/umwe_3.jpg)
  - $$N_Y(x)$$ : 단어x가 $$S_Y$$에 Mapping 되었을 때, 그 **주변에 있는 10개의 단어 집합** (10개는 논문에서의 Setting)
  - $$cos(x,y)$$ : **Cosine 유사도**, 두 벡터가 유사한 값을 가지면 1에 가까운 값, 다른 값을 가지면 0에 가까운 값을 출력

  > **직관적으로 식을 이해해보자.**
  - x와 y는 MAT모델이 pair라고 학습한 단어 묶음이 될 것이다. 
  - x와y가 유사하다면 $$cos(x,y)$$값이 커지며, CSLS값도 증가할 것이다.
  - $$N_Y(x)$$는 y이외에 9개의 단어벡터(y')들이 있을 것이며, 그 벡터들도 x와 유사해 $$cos(x,y')$$값들이 커지면, CSLS값은 작아진다.($$N_X(y)$$의 경우도 마찬가지)
  - 즉, `'x와 y의 Mapping관계가 애매하지 않고, 되도록이면 다른 단어들과 확실히 구분되었으면 좋겠다.'`라는 바람이 들어있는 식이다.
  
  <br>

  　위의 과정을 통해서 **사용빈도도 높고 다른 단어들에 비해 확실히 구분되는(**[**hubness**](https://en.wiktionary.org/wiki/hubness)**문제를 해결한) 단어쌍**을 얻을 수 있다. 이를 이용하여 학습의 Refinement를 진행하는데,<br><br>
  ![式](/assets/img/dev/umwe_4.jpg)
  - $$Lex(i,j)$$ : **Pseudo-Supervised Lexicon**, CSLS를 통해 만들어진 언어$$_i$$와 언어$$_j$$간의 사전(Dictionary)
  - $$M_i x_i$$ : **$$T$$로의 Mapping**, 언어$$_i$$에서 Target Space로의 Mapping
  - $$L_r(M_i x_i, M_j x_j)$$ : **손실함수(Loss Function)**, Mean Square Loss 사용

  > $$Lex(i,j)$$를 대상으로 하는 $$(M_i x_i, M_j x_j)$$는, `유사한 뜻이라고 생각되는 단어들의 Target Space로의 Mapping`을 뜻한다. 이 Loss를 최소화 함으로써, 신뢰도 높은 단어들을 위주로 한 번 더 $$M_i$$를 조절해보는 것이다.

  <br>

  　이외에도, **Orthogonalization과 Mean_CSLS를 사용한 평가**에 관한 내용도 짧게 소개되어 있다.

  <br><br>

## <a name="jump7">정리</a>
　내가 만약 이 논문을 아무런 생각없이 읽었다면, **'아 그렇구나...'**하며 당연하게 읽었을지도 모를 것 같다. 하지만, `Multilingual Adversarial Training에서 GAN이 기하급수적으로 늘어나는 문제`, `Generator가 겹치면서 일부 Generator만 학습되고 일부는 전혀 학습이 진행되지 않는 문제`를 어떻게 해결할 수 있을지 찾고 있던 와중에 발견한 이 논문은 상당히 의미있는 내용이었다.

- **언어별로 Discriminator를 하나씩** 두고 학습을 진행하는 것,
- **랜덤하게 선택된 언어의 쌍**으로 Generator를 학습하는 것<br>  이 두가지를 챙겨가자.
