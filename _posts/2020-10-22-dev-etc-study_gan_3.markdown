---
layout: post
title:  "[Study] Build Basic Generative Adversarial Networks(GANs)③"
subtitle:   "Subtitle - GAN Coursera"
categories: dev
tags: etc
comments: true
---
## Describe
> [`Build Basic Generative Adversarial Networks (GANs)`](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/home/welcome)를 청강, 개인적인 정리<br>
Week 3: Wasserstein GANs with Gradient Penalty<br>
한국어 韓国語

## 目次
- [Week 3: Wasserstein GANs with Gradient Penalty](#jump1)
  - [GAN의 문제점](#jump2)
  - [문제점의 해결방안](#jump3)

**　강의를 들으며 개인적으로 생각나는 부분을 적고 있습니다. 자세한 내용에 대해서는 따로 링크를 달아두도록 하겠습니다. 실제로 참고를 하고 있는 링크글들이기 떄문에 도움이 되실거라 생각합니다.**<br>
**　수식이 잘 보이지 않을 떄에는 페이지 새로고침(F５)을 한 번 해보세요. 뭐가 문제인지ㅠ..**

<br><br><br>

## <a name="jump1">Week 3: Wasserstein GANs with Gradient Penalty</a>
　**기존 GAN이 가지고 있는 불안정성?**을 해결하기 위한 GAN의 진화. Discriminator와 Generator의 경쟁 간에 나타날 수 있는 문제점들을 소개하고, 그를 해결하기 위한 새로운 Loss Function을 제안한다..<br><br>

### <a name="jump2">GAN의 문제점</a>
　**Mode Collapse**
- **Mode : 확률분포에서 피크를 나타내는(우리가 학습해야하는) 부분**인데,
- 일반적인 경우에 이 Mode가 하나일리가 없다. 
- 예)'5'이 숫자는 몇인가요? {0,1,2,...,9},10개의 Mode가 존재
- 여기서 나타나는 Generator의 꼼수
- **숫자 1과 7**은 사람이 봐도 헷갈리는 숫자인데,
- `Discriminator가 속을 수 밖에 없는 Mode만 공략`하는 경향이 발생,
- Generator는 1과 7이라는 숫자만 생성하고, Discriminator를 잘 속이고 있다고 생각한다.
- 이 경우, Generator의 성장은 어느순간 멈춰버린다.

<br>　**Problem with BCE Loss**
- 참고 : [BCE Loss](https://nhandsome.github.io/dev/2020/10/06/dev-etc-study_gan/#jump2)
- Discriminator와 Generator의 불균형
- BCE에서 **Discriminator는 Real/Fake**의 판단을 하지만, **Generator는 복잡한 수식**을 학습해야한다.
- 학습초기에는 서로가 멍청?해서 쿵짝이 잘맞고 재미있는 학습이 되지만,
- Discriminator의 성능이 월등히 좋아지면, Generator는 학습의 의지를 잃어버리게 된다.
- 즉, 손실함수의 Gradient(Discriminator의 피드백)를 이용하여 최적화를 해야하는데, Gradient값이 극단적이라(의미있는 정보가 없어) 더 이상 학습이 불가능하게 된다. `Vanishing Gradients`

<br>

### <a name="jump3">문제점의 해결방안</a>
<br>　**Wasserstein Loss**
- Wasserstein Loss에 관해서는 따로 한 번 공부를 해야할 것 같다.
- 단순히 Loss를 구한다고 하면, 정답의 분포/예측의 분포에서 하나씩의 샘플을 찾고
- 2개의 차이를 계산하는 방식이다.
- `Earth Mover's Distance`는 두개의 분포를 두개의 덩어리로 보고 전체적인 모양을 비교하는 느낌으로
- 이 방법을 이용한 것이 Wasserstein Loss이다.

|**BCE Loss**|**W-Loss**|
|:---:|:---:|
|Discriminator가 0~1의 값을 출력|Critic이 모든 실수를 출력|
|$$-\Bbb{E}(log(d(x)))-\Bbb{E}(1-log(d(g(z))))$$|$$\Bbb{E}(c(x))-\Bbb{E}(c(g(z)))$$|

- `BCE Loss`에서 함수(d)는 0~1의 값만 출력하고, log가 추가되면 **-inf~0**이 되면서(기울기도 0) 
- Discriminator(d)의 성능이 좋을 경우, Loss가 극단적으로 변한다(Vanishing Gradients)
- `W-Loss`에서 함수(c)는 실수의 값을 출력하며, log도 없어 극단적인 판단(기울기가 0)을 하지 않고
- Generator가 학습을 할 수 있는 피드백을 언제든지 전달할 수 있다.

<br>　**Condition on W-Loss**
- 그렇다고 임의의 함수(c)를 사용해도 되는 것은 아니다.
- 함수(c)는 `1-Lipchitz Continuous`조건을 만족하지 않으면 안되는데,
- 이는 함수(c)의 미분함수의 [L2 Norm](http://taewan.kim/post/norm/)이 -1~1 범위에 존재한다는 의미이다.
- 이 조건을 만족함으로써, **연속/미분가능**은 물론, 학습과정 중의 **안정된 피드백**을 보장한다.

<br>　**1-Lipchitz Continuous Enforcement**
- **Weight Clipping** : Gradient descent(기울기의 피드백)으로 큰 값이 들어오더라도, 함수(c)에 반영하는 최대/최소값은 고정시킨다.
- W-Loss의 성능을 제한시켜버린다는 단점이 있다.
- `Gradient Penalty` : 함수(c)에 Regularization term을 추가하여 값을 제한하는 방법

$$\Bbb{E}(c(x))-\Bbb{E}(c(g(z))) + \lambda reg$$

$$reg = \lVert \triangledown c(r) \rVert_2 - 1$$

$$r = \epsilon x + (1-\epsilon)g(z)$$

- 아래의 식부터 살펴보면, $$\epsilon$$은 랜덤의 수로
- Real값과 Fake값을 임의의 비율로 합성하여 새로운 값(r)을 생성한다.
- reg는 그 새로운 값(r)이 1에 가까워지도록 노력하며,
- $$\lambda$$ 파라미터 만큼의 영향력을 가져, 원래의 W-Loss에서 함수(c)가 1-Lipchitz Continuous를 어느정도 따르도록 강제한다.

> Wasserstein GAN을 사용하면, 기존의 GAN이 가지고 있던 문제점
- Mode Collapse : 다양한 선택지 중에 Generator가 유리한 값만 생성하는 것
- Vanishing Gradients : Generator가 학습의지를 잃어버리는 피드백<br>
을 해소할 수 있다.

> Wasserstein GAN은
- Real분포와 Fake분포 전체를 비교하는 문제가 되며,
- Discriminator는 두 분포에서 나온 값들은 잘 구분할 수 있는 함수(Critic)를 학습,
- Generator는 함수(Critic)의 학습을 방해한다.

<br><br>　

**[From GAN to WGAN 한국어 번역자료](https://github.com/yjucho1/articles/blob/master/fromGANtoWGAN/readme.md) 참고로 마무리**


