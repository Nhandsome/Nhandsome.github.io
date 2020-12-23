---
layout: post
title:  "[Study] Build Basic Generative Adversarial Networks(GANs)④"
subtitle:   "Subtitle - GAN Coursera"
categories: dev
tags: etc
comments: true
---
## Describe
> [`Build Basic Generative Adversarial Networks (GANs)`](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/home/welcome)를 청강, 개인적인 정리<br>
Week 4: Conditional GAN & Controllable Generation<br>
한국어 韓国語

## 目次
- [Week 4: Conditional GAN & Controllable Generation](#jump1)

**　수식이 잘 보이지 않을 떄에는 페이지 새로고침(F５)을 한 번 해보세요. 뭐가 문제인지ㅠ..**<br>
　조금 더 똑똑한 GAN을 만들기 위해서, 새로운 Class정보를 추가시킨다. 단순히 '강아지 사진을 만들어 줘'라는 것을 넘어, '진돗개/불독/시바 사진을 만들어 줘'라는 학습으로 발전하는 것이다. 사실 별내용이 없는 파트라 대충 개념만 짚고 넘어가겠다.

<br><br><br>

## <a name="jump1">Week 4: Conditional GAN & Controllable Generation</a>
　**기존 GAN이 가지고 있는 불안정성?**을 해결하기 위한 GAN의 진화. Discriminator와 Generator의 경쟁 간에 나타날 수 있는 문제점들을 소개하고, 그를 해결하기 위한 새로운 Loss Function을 제안한다..<br><br>

||**Unconditional GAN**|**Conditional GAN**|**Controllable GAN**|
|:---:|:---:|:---:|:---:|
|Input|무작위로 추출|원하는 Class에서 추출|원하는 Feature를 가진 데이터에서 추출|
|Class|X|O|X|
|특징|뭐라도 만들어본다.|Class에 맞춰서 만든다.|특징을 조절하며 만든다.|

<br>　**Conditional GAN**
- Unconditional GAN이 임의의 벡터정보를 기반으로 '강아지 사진을 생성'하는 Generator를 학습했다면,
- Conditional GAN은 '강아지의 종을 선택'할 수 있는 Generator를 학습한다.
- Discriminator가 단순히 강아지같은 사진을 판별하는 것이아니라,
- Class(견종)정보와 매칭되는 강아지 사진을 판별하고 있기 때문이다. 

<br>　**Controllable GAN**
- 강아지 사진을 생성하는 Generator를 학습시켜
- 몇가지 Input을 넣어 결과물을 살펴봤더니,
- 털의 길이가 긴 요크셔테리어, 짧은 요크셔테리어 사진이 나왔다고 하자. 
- 우리는 두 사진의 Input의 평균으로 중긴 길이의 털을 가진 요크셔테리어를 생성할 수 있을지도 모른다.
- 하지만 Input 정보에 비해, Output 정보가 많은 경우가 대부분이므로, 상관관계가 생길 수 밖에 없다.
- 예를 들어, 털의 길이를 조절했더니 꼬리/주둥이/귀의 길이도 전체적으로 길어질 수 있다.(Entanglement)

<br>　**Classifier Gradients**
- 하나의 학습방법이자, 응용기술이 될 것 같다.
- 준비물 : 사람얼굴 사진을 만드는 Generator, 사진에 안경이 있는지 없는지 판단하는 Classifier
- Noise Vector를 Input으로 사람얼굴을 계속 생성하는데,
- 결과물에 안경이 있는지 없는지 Classifier가 판단한다.
- 그 피드백을 바탕으로 Noise Vector가 업데이트되며,
- 안경 쓴 사람얼굴을 생성하는 (Noise Vector, Generator)쌍을 구할 수 있다.
- 확실한 성능을 가진 Classfier를 통해 Noise Vector 최적화를 위한 Gradients 계산문제로 생각할 수 있다.

