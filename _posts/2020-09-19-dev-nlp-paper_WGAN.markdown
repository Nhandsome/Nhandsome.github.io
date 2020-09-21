---
layout: post
title:  "[Paper] ABWGAN"
subtitle:   "Subtitle - Bicycle Travel in Europe"
categories: dev
tags: nlp
comments: true
---
## Describe
> [`Wasserstein GAN based on Autoencoder with back-translation for cross-lingual embedding mappings`](https://www.sciencedirect.com/science/article/pii/S0167865519303538)을 읽고 정리?한 글.<br>
한국어 韓国語

<!-- ## 目次
- [ヨーロッパ自転車旅行ってなに？](#jump1)
- [どこに行ったの?](#jump2)
  - [チェコ](#jump3)
  - [ドイツ](#jump4)
  - [フランス](#jump5)
  - [スイス](#jump6)
  - [イタリア](#jump7)
  - [スロベニア](#jump8)
  - [クロアチア](#jump9)
  - [ハンガリー](#jump10)
  - [スロバキア](#jump11)
  - [オーストリア](#jump12) -->


　작성 중인 논문에 도움이 되는 부분이 있어 참고한 논문입니다.<br> 다른 분들의 '논문해석/분석'과 비교하면 수준낮은 글이 될 것 같습니다. 하지만, 영어논문을 읽기 전에 `대충 어떤 내용의 논문인가..`라는 것을 파악하고 싶은 분들에게는 도움이 되지 않을까 싶습니다.


## 어떤 내용인가?
　Generative Adversarial Network(GAN)을 활용한 Cross-lingual Word Mapping(CWM)가 주목을 받고 있다.
하지만 일반적인 GAN base CWM는 JS Distance를 사용, 이는 GAN학습에 악영향을 미치기도한다.<br>
　이를 해결하기 위해, Wasserstein GAN based on autoencoder with back-translation(ABWGAN)를 제안, 
- Wasserstein Distance를 이용한 WGAN 사전학습
- Back-translation with Target-side(BT-TS)를 이용한 튜닝<br>
에 대한 연구 논문이다.

> `Crose-lingual Word Mapping(Embbeding) using GAN 이라는 주제에 관심이 있으신 분`들이 읽을만한 논문인 것 같다. 기존의 GAN based CWM을 개선하는 연구이지, Wasserstein Distance나 GAN, CWM에 대한 개념을 설명하고 있지는 않다.

## 인트로
　CWM은 서로 다른 언어의 단어를, 일부 정보만을 사용하여, 비슷한 뜻을 가진 단어와 맵핑시키는 과정이다. 대부분의 기존 CWM은 Parallel Resource(아마 똑같은 뜻을 가졌을 것이라고 생각되는 단어의 쌍)의 규모가 확보되어야 그 성능이 보장된다. <br>　하지만 `Parallel Resource 정보는 이미 잘 알고 있는 유사한 언어(영어, 라틴계열언어)에서 잘 정의되지만, 정보가 적은 언어에 대해서는 규모가 작아지고 그 결과 CWM는 나쁜 성능`을 보인다.

　여기서 조금 변형된 것이 `GAN을 이용한 CWM`으로, Parallel Resource 정보가 없는 상황에서 단어들의 쌍을 찾아나간다.(Unsupervised Cross-lingual Word Embbeding) GAN의 개념은 'Generator(속이는 자)'와 'Discriminator(판단하는 자)'의 경쟁학습으로 생각할 수 있으며,<br>**CWM를 예로 들자면, Apple에 대하여**
- `속이는 자`는 '「동물」,「식물」,「과일」,「귤」,「사과」'처럼 `Apple과 상응하는 '어떤 한국어'를 찾기 위해 '아무 한국어'를 반복적으로 제안`한다. 
- 반대로 `판단하는 자`는 「동물」,「식물」,「과일」,...이라는 단어가 `Apple에 상응하는 '진짜 한국어'인지 판단`한다. 
- 판단결과를 피드백으로 `'속이는 자'는 '진짜 한국어'에 가까운 단어들을 학습하고, '판단하는 자'는 '정밀한 판단'을 학습`한다.

**(참고)Parallel Resource가 없는데 어떻게 학습하지?**
- 트럼프(Trumph), 서울(Seoul), 코로나(CORONA)와 같이 사전(Dictionary)정보가 필요없는 고유명사의 쌍
- 라틴계열의 경우, 똑같은 알파벳을 사용하는 단어의 쌍
<br>등을 'Parallel Resource'후보로 사용하여, 주변의 단어들을 쌍을 그럴듯하게 학습해나간다.

　`**이 논문에서는 GAN을 이용한 CWM을 조금 더 개선하고자 한다.**`


## 참고 : Wasserstein GAN
　GAN에서 '학습을 한다'라고 하는 것은, `'정답이라고 생각하는 것 - 예측한 것'의 거리(Loss)을 피드백으로 똑똑해져가는 과정｀이다. Wasserstein GAN은 그 거리를 측정하는 기준으로 'Wasserstein Distance'를 사용하는데,<br>
**Wasserstein Distance를 사용하는 것이 왜 득이 되는지를 간단히 보자면**
- 학습을 위해 필요한 `'거리'라는 개념은 두 분포간의 차이를 설명하는 수치`로, 그 종류가 다양하다.
- TV / KL / JS / EM(Wasserstein) 4가지의 거리측정법이 대표적인 예인데,
- 일반적인 GAN에서 사용되는 거리측정법은 JS로, 'TV→KL→JS'진화과정?을 거친 거리측정법이다.
- 이 종족?의 거리측정법은 `'O,X'방식으로 답을 말해주기 때문에(분포가 같냐? 다르냐?), 학습이 곤란한 경우`가 있다.
- 하지만 `EM 거리측정을 사용하면, '지금 거리 차이가 어느정도인지' 수치로 답을 알려준다.`
- 학습하는 입장에서, 두 분포가 똑같아지도록 파라미터를 조정하는 작업이 편해진다. 

　위의 해석은 그저 개념을 이해하기 위한 엉터리 설명에 가깝습니다. Wasserstein Distance의 자세한 설명은 [이곳](https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i)을 참고하시기 바랍니다.

## 방법론
**이 논문에서의 3스텝**
1. 전처리 
  - 각 언어 Mapping Length를 Normalizing한다. (영어는 정보가 많으니 많은 단어가 있을 것이고, 아프리카어는 정보가 별로 없으니 단어가 적을지도 모른다.)
  - 너무 사용빈도가 적은 단어들은 제외한다. 
  - 이런 과정들은, 효과가 증명된 선행 연구들 중 필요한 부분을 따라?하는 것
1. 사전 Mapping 학습
  - Wasserstein GAN을 학습하는 과정
  - Generator1(속이는 자)는 '사과'라는 한국어단어의 공간좌표를 받으면, `그럴듯한 영어단어의 공간좌표를 출력`한다.
  - Discriminator(판단하는 자)는 어떤 영어단어의 공간좌표를 받으면, 그게 `실제 존재하는 영어단어인지 아닌지 판단`한다.
  - Generator2(속이는 자 친구)는 `Discriminator의 판단결과`와 Generator1가 만든 `영어공간좌표를 다시 한국어공간좌표로 되돌리는 과정`을 분석/학습한다.
1. BT-TS
  - Wasserstein GAN을 조정하는 과정
  - 사전 Mapping 학습을 통해 '한→영'/'영→한' 공간좌표를 변환하는 `Generator 2개를 얻었다.`
  - 사전 학습에서는 '한국어단어를 넣어 Generator를 학습'했는데,
  - 영어단어를 '영→한', '한→영'변환을 시행하고, `영어단어가 제대로 복구되었는지 판단/학습`한다.

## 정리
　이외의 디테일은, Orthogonality를 위한 파라미터들의 조정과정, BT-TS의 당위성, Loss Function의 내용과 같은 것이 실려있다.
Embbeding하는 언어에 따라 파라미터의 설정도 달라져야 한다는 것을 알 수 있으며, 파라미터의 설정을 자동화하는 것을 다음 연구의 목표로하고 있다고 한다.<br>
　개인적으론 [MUSE](https://github.com/facebookresearch/MUSE)를 사용한 Cross-lingual Word Embbeding 과정에서 Parallel Resource가 전혀 생성되지 않아(일부 언어에 대한 성능이 나빠서) 곤란했었는데, ABWGAN을 참고해보려고 한다.



