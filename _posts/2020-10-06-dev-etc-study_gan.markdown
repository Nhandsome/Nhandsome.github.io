---
layout: post
title:  "[Study] Build Basic Generative Adversarial Networks(GANs)①"
subtitle:   "Subtitle - GAN Coursera"
categories: dev
tags: etc
comments: true
---
## Describe
> [`Build Basic Generative Adversarial Networks (GANs)`](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/home/welcome)를 청강, 개인적인 정리①<br>
한국어 韓国語

## 目次
- [Week 1: Intro to GANs ①](#jump1)
  - [Binary Cross Entropy(BCE) Loss](#jump2)

**　Coursera에 GAN 강좌가 올라와서 가볍게 청강 중입니다. 기본적인 신경망학습, Pytorch 지식이 있는 사람들 대상으로 하고 있는 것 같지만, 초반부에 조금씩 설명해주는 듯 하네요. 딱히 GAN에 대한 설명을 하기 위한 글은 아니고, 개인적으로 정리하는 수준의 글입니다.**

<br><br><br>

## <a name="jump1">Week 1: Intro to GANs ①</a>
　**GAN의 목표**는, Generator와 Discriminator의 경쟁학습을 통해, `원하는 이미지를 생성할 수 있는 Generator를 학습`시키는 것이다. 좋은 Discriminator가 있어서 좋은 Generator가 학습되었다면, 이후에 Discriminator는 없어도 된다.<br><br>
　**GAN의 활용**
- 그림을 대충 그려놓으면 그럴듯한 작품으로 바꿔주는 기능
- 언어를 자동으로 생성하는 기능
- 부족한 데이터베이스에 그럴듯한 데이터를 생성하는 것...

<br>　**Pre-trained GAN 체험**
- [StyleGAN:존재하지 않는 얼굴 생성](https://github.com/NVlabs/stylegan)
- [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)

<br>　**Discriminator는 일종의 Classifier로써**, $$P\left(Y_c | X_f\right)$$ 
- X라는 feature 정보가 주어졌을 떄, Y라는 class가 어떤게 가장 그럴듯한지 학습한다.
- Class는 고양이/강아지/새/식물.. 이 될 수도 있고, Fake/Real이 될 수도 있고...
- 실제 Y와 판단한 $$Y_c$$의 값을 Cost로 하여, 파라미터 $$\theta$$를 학습하는데,
- 이 Discriminator가 잘 학습될수록, Generator의 좋은 스승이 된다.

<br>　**Generator는 노이즈($$\varepsilon$$)를 Input으로 필요로하며,**, $$P\left(X_f | Y_c\right)$$ 
- Y라는 class를 목표로, X라는 feature를 그럴듯하게 만들어낸다.
- 만들어진 Fake feature $$X'_f$$를 Discriminator에게 보내, 평가를 받는다.
- Discriminator의 판단 $$P(Y_c \vert X'_f)$$ 을 Cost로 하여, $$\theta$$를 학습,
- 좋은 Discriminator로 충분히 학습이 끝나면, 노이즈 $$\varepsilon$$와 $$P(X_f)$$로 언제든지 Fake자료를 만들 수 있다.

### <a name="jump2">Binary Cross Entropy(BCE) Loss</a>
　**Binary Cross Entropy Loss**는 자주 등장하는 개념이며, 이해하기 쉽게 설명이 되어 있어서 정리해보려고 한다. 엔트로피와 같은 개념은 다른 자료를 찾아보면 구체적인 이야기가 있을 것이니, GAN관점에서 `Binary문제(Fake or Real)을 판단할 때, Discriminator는 어떤 생각을 하고 있을까?`를 생각해보자.<br>
　먼저, 2가지 판단을 하지않으면 안된다. [정밀도와 재현율](https://ko.wikipedia.org/wiki/%EC%A0%95%EB%B0%80%EB%8F%84%EC%99%80_%EC%9E%AC%ED%98%84%EC%9C%A8)은 어디선가 들어보았고, 항상 헷갈리고... 즉, Fake를 Fake라고 판단할 수 있는 능력과, True를 True라고 판단할 수 있는 능력이 필요하다. '뭐 똑같은 소리아니야?'라고 생각할 수 있지만, **눈에 보이는 모든 사람을 죄인이라고 판단하는 경찰**을 상상해보면, 이 경찰은 **'모든 죄인을 잡는 좋은 경찰일까, 무고한 시민을 항상 조사하는 귀찮은 경찰일까?'** 애매하지 않은가...
　**이 느낌을 갖고 아래의 식을 이해해보자.**

$$-\frac 1 m \displaystyle\sum_{i=1}^m \Big[ y^i logD(x^i,\theta) + (1-y^i)log(1-D(x^i,\theta)) \Big]$$

- $$-\frac 1 m \displaystyle\sum_{i=1}^m$$ : **평균**, m번의 계산을 하고 그 평균을 구할 것이다.
- $$y^i$$ : **정답($$Y_c$$)**, Fake라면 0, True라면 1이다.
- $$D(x^i,\theta)$$ : **$$X_f$$에 대한 Discriminator의 평가**, Fake라고 생각하면 0에 가깝게, True라고 생각하면 1에 가깝게 출력.
- $$y^i logD(x^i,\theta)$$ : **그 둘의 Cross Entropy**인데, 어떤 개념인지 보자면

|**$$y^i$$**|**$$D(x^i,\theta)$$**|**$$y^i logD(x^i,\theta)$$ = -Loss**|
|:---:|:---:|:---:|
|0|0~1|0|
|1|0.99|0.00...01|
|1|0.01|-Infinity|

　\$$y^i = 1\$$(True)일 때, \$\$D(x^i,\theta)\$\$가 좋은 평가를 하면 Loss를 0에 가깝게 하고, 나쁜 평가를 하면 Loss를 무한대에 가깝게 한다.(식의 앞에 -가 있으니까.) **그리고 $$(1-y^i)log(1-D(x^i,\theta))$$는 위의 표와 반대로, $$y^i = 0$$(False)일 때 Loss를 발생시킨다.**

> 따라서 두개의 Cross Entropy Loss를 더한 후 그 값을 최소화시키는 $$\theta$$를 찾는 것이, Discriminator의 과제인 것이다. **(확률적으로)악질 범죄자는 잘 잡으면서, 무고한 시민은 체포하지 않는 경찰**이 탄생.

<br><br>

**[`Build Basic Generative Adversarial Networks (GANs)`](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/home/welcome)강의의 1주차 과정①　정리 끝. 1주차 내용의 50%정도 인 것 같습니다. 4주차 강의까지 있고, 이후에 Advanced강의 Apply강의까지 있는 것 같은데, 시간이 날 때마다 정리하겠습니다.**


