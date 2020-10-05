---
layout: post
title:  "[Paper] UNSUPERVISED MULTILINGUAL WORD EMBEDDINGS"
subtitle:   "Subtitle - UNSUPERVISED MULTILINGUAL WORD EMBEDDINGS"
categories: dev
tags: nlp
comments: true
---
## Describe
> [`UNSUPERVISED MULTILINGUAL WORD EMBEDDINGS`](https://www.aclweb.org/anthology/D18-1024.pdf)를 읽고 정리?한 글.<br>
한국어 韓国語

## 目次
- [어떤 내용인가?](#jump1)
- [전체적인 개념잡기](#jump2)
- [구체적인 모델](#jump3)
  - [기본이 되는 개념](#jump4)
  - [Multilingual Adversarial Training(MAT)](#jump5)
  - [Multilingual Pseudo-Supervised Refinement(MPSR)](#jump6)
- [정리](#jump7)


**　작성 중인 논문에 도움이 되는 부분이 있어 참고한 논문입니다.<br> 다른 분들의 '논문해석/분석'과 비교하면 수준낮은 글이 될 것 같습니다. 하지만, 영어논문을 읽기 전에 `대충 어떤 내용의 논문인가..`라는 것을 파악하고 싶은 분들에게는 도움이 되지 않을까 싶습니다.**

<br><br><br>

## <a name="jump1">어떤 내용인가?</a>
　개인적으로는... 사실 여태 생각하고 있던 논문의 아이디어를 조금 바꿔야할지도 모르는...`조금 더 빨리 발견했어야하는 논문`이었다.
많은 개념이 비슷한 것은 물론, 대충 생각만해봐도 이 논문에서 소개한 모델의 성능이 더 좋을 것 같은..<br>
　하지만, 논문을 정리하면서 새로운 생각이 떠오를 것을 기대하며, 간단히 소개하고자 한다.
- Unsupervised 학습으로 언어 간의 Word Embedding을 학습시키는 것이 목표로써,
- 매 학습시 `랜덤하게 2개의 언어를 선택`, 여려개의 Discriminator/Generator 를 학습시키는 Multilingual Adversarial Training(MAT),
- 위의 학습결과를 `Target Space에서 튜닝`하는 Multilingual Pseudo-Supervised Refinement(MPSR)<br>
이 두가지를 주로 다루고 있다.

> Word Embedding은 학습방법(Pivot의 사용/학습의 방향)에 따라 성능이 달라지기 마련이다. 이 연구에서는 `학습할 언어를 랜덤으로 선택`하는 방법으로 `간단하지만 다양한 학습방법을 고려한 모델`을 제안한다.

<br><br><br>


## <a name="jump2">전체적인 개념잡기</a>
　기존의 Bilingual Word Embedding(BWE)모델, 특히 일부 Unsupervised 모델[WORD TRANSLATION WITHOUT PARALLEL DATA](https://arxiv.org/pdf/1710.04087.pdf)은 특정 언어간의 Embedding 학습에서 Supervised 모델보다 뛰어난 성능을 보이고 있다.<br>　하지만, `Bilingual이라는 단어에서 알 수 있듯, 2개의 언어 간의 Embedding학습만이 가능`하며, 비슷한 방식으로 언어의 수만 늘려나가면, N(N-1) 혹은 N(N-1)/2 번의 학습이 필요하게 되고, 그 효율성은 크게 떨어지게 될 것이다. 그리고, 자칫하면 개별적인 학습을 여러번 진행하는 것일 뿐, 의미없는(학습 간 상호작용이 전혀 없는) 반복학습이 되어버릴지도 모른다. `즉, Multiliingual로의 적용이 쉽지만은 않다`라는 것을 알 수 있다.<br><br>

　학습의 효율성을 개선할 수 있는 방법이 Pivot(공통된/Target) Space를 이용하는 방법이다. 모든 언어가 공유할 수 있는 대표언어 혹은 대표공간을 만드는 개념인데, `일단 프랑크푸르트공항으로 가면 대부분의 유럽국가로 갈 수 있다.`라는 (내 마음속의) 사실과 비슷하다. 모든 언어 간, Embedding 학습을 진행하는 것이 아니라, 모든 언어를 특정 공간(혹은 특정 언어)에 Embedding시키는 것이다. '한국어-일본어'의 변환은 학습하지 못했지만, '한국어-영어-일본어'의 과정으로 이를 대체할 수 있으며, N번의 학습으로 모든 언어간의 변환을 커퍼할 수 있는 것이다.<br>
　하지만, 'Pivot Space와 1개 언어'의 학습을 개별적으로 강화하는 과정으로, `학습속도의 효율성을 보장하는 것으로 성능의 효율성을 보장하지는 않는다.` 쉽게 이야기하면,
'인천-프랑크푸르트-프라하'의 최저금액을 검색하는 것이 아니라, '인천-프랑크푸르트'와 '프랑크푸르트-프라하'의 최저가를 검색하는 느낌이려나..   <br>

> 구체적/수학적인 이야기를 제하고 `개념적으로 이해한 UNSUPERVISED MULTILINGUAL WORD EMBEDDINGS의 장점`은

- 기본적인 구조로써 **Pivot Space를 이용한 학습**으로, `학습시간의 효율성을 보장한다.`라는 것,
- 학습 과정마다 Pivot Space에 모여드는 **각 Embedding의 관계들을 조율**하면서, `학습이 따로따로 노는 것을 방지`한다는 것이다.
- 「①복잡한 노선 만들지 말고 프랑크푸르트로 다 모이세요.」
- 「②프랑크푸르트 경유 기준으로 모든 노선을 합리적으로 만들어볼게요.」

<br><br><br>

## <a name="jump3">구체적인 모델</a>
　[논문](https://www.aclweb.org/anthology/D18-1024.pdf)에는 그림과 알고리즘도 잘 정리되어 있으니, 필요하다면 논문을 살펴보자.

### <a name="jump4">**기본이 되는 개념**</a>
  - $$T$$ : **Target Space**, 모든 언어의 Embedding Space는 T를 통해 다른언어로 Mapping된다.
  - $$S_i$$: **Embedding Space**, 언어$$_i$$의 Embedding Space
  - $$E_i$$ : **Set of Embedding**, $$S_i$$에서 선택된 단어의 집합
  - $$M_i$$ : **Mapping to T**, 언어$$_i$$를 $$T$$(Target Space)에 Mapping 시키는 행렬
        - $$M_t' = M_t^T  M_t = I$$,
        - $$(M_j^T  M_i)E_i \in S_j$$,

  > $$T$$(Target Space)를 통해서 원하는 언어$$S_i$$로의 Mapping이 가능하며, `$$M_i$$는 $$T$$로의 Encoding, $$M_i^T$$는 $$T$$로부터의 Decoding 연산`으로 생각할 수 있다. 

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
