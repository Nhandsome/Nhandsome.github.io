---
layout: post
title:  "[Study] Build Basic Generative Adversarial Networks(GANs)②"
subtitle:   "Subtitle - GAN Coursera"
categories: dev
tags: etc
comments: true
---
## Describe
> [`Build Basic Generative Adversarial Networks (GANs)`](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/home/welcome)를 청강, 개인적인 정리<br>
Week 2: Deep Convolutional GAN<br>
한국어 韓国語

## 目次
- [Week 2: Deep Convolutional GAN](#jump1)

**　강의를 들으며 개인적으로 생각나는 부분을 적고 있습니다. 자세한 내용에 대해서는 따로 링크를 달아두도록 하겠습니다. 실제로 참고를 하고 있는 링크글들이기 떄문에 도움이 되실거라 생각합니다.**<br>
**　수식이 잘 보이지 않을 떄에는 페이지 새로고침(F５)을 한 번 해보세요. 뭐가 문제인지ㅠ..**

<br><br><br>

## <a name="jump1">Week 2: Deep Convolutional GAN</a>
　**Deep Convolutional GAN**을 만들기 위한 기초 개념들을 설명하고 있다. [Convolution Neural Network](https://umbum.dev/223?category=751025)는 이미지의 학습 같이 복잡한 처리를 위해 사용되는 하나의 학습방식이다.<br><br>
　**Activation Function**
- Gradient를 정보로 사용해야하기 때문에 미분가능하며,
- Deep learning을 위해 비선형인 함수여야한다.(원점을 지나는 수십개의 서로 다른 직선함수를 중첩하더라도, 결국엔 원점을 지나는 직선함수(선형))
- Sigmoid : O/X의 판단에서는 좋은 선택
- ReLU : 일반적인 선택
- `Activation Function에 관한 내용`은 [이곳](https://ganghee-lee.tistory.com/32)을 참고, `Gradiant가 왜 필요한가에 관한 내용`은 [이곳](https://www.youtube.com/watch?v=GmtqOlPYB84&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc&index=21)의 딥러닝에 관한 영상을 참고하기.

<br>　**Batch Normalization**
- Batch를 생성할 떄 마다, 평균/분산을 정규화하여 학습을 진행한다.
- 모든 Batch의 학습이 종료되면 평균/분산을 합치고,
- 통합된 평균/분산을 기준으로, 테스트 데이터를 정규화/테스트한다. 
- **전체 데이터를 정규화하는 것이 아니기 때문에 효율이 좋다.**

<br>　**Convolution이란?**
![式](/assets/img/dev/gan_1.jpg)
- [Coursera강의](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/home/welcome)의 이미지
- Convolution 연산을 통해 `이미지의 특징(Filter)를 학습할 수 있다.`(Stride=1)
- 위의 예제는 **세로 경계**라는 특징을 가진 Filter를 나타내고 있다.
- 나아가 '눈동자/코/귀'를 인식하는 Filter를 학습할 수 있고, 이미자가 고양이인지 개인지 판단할 수 있는 Feature로 사용한다.

<br>　**Padding**
- 이미지의 가장자리를 만드는 것(0으로 채우는 경우가 많다.)
- Filter는 이미지를 한번씩 훑으며 지나가는데, 가장자리의 경우 다른 부분보다 스캔횟수가 적어진다.
- 가장자리의 정보량이 줄어드는 것을 방지하기위해, 0으로 된 액자로 가장자리를 채워준다.

<br>　**Pooling/Upsampling**
- Window에서 가장 큰 값을 취하거나, 평균을 취하거나..
- 이미지의 크기를 줄이는 것이 Pooling이고,
- 이미지 크기를 키우기 위해, 픽셀을 복제하거나 하는 것이 Upsampling이다.
- 둘 다 학습할 필요가 없는 단순한 작업이다.

<br>　**Transposed Convolution**
- 이미지를 Upsampling하기 위해 Filter를 학습하는 과정으로,
- 이미지의 각 부분을 필터를 통해 확대시킨다.
- 격자무늬(패턴)이 생기는 문제점이 있다.
- 둘 다 학습할 필요가 없는 단순한 작업이다.

<br><br>　

**이후, 논문소개([UNSUPERVISED REPRESENTATION LEARNING
WITH DEEP CONVOLUTIONAL
GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf))로 마무리.**


