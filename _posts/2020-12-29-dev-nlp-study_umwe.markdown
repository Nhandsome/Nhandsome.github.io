---
layout: post
title:  "[NLP] Unsupervised Multilingual Word Embedding Model"
subtitle:   "Subtitle - Playing with unsupervised multilingual word embedding model"
categories: dev
tags: nlp
comments: true
##published: false
---
## Describe
> Do various experiments with [`UMWE model`](https://github.com/ccsasuke/umwe), play with it.<br>

## 目次
- [UMWE란?](#jump1)
- [Supervised(identical char)](#jump2)
  - [de, it - EN](#jump3)
  - [ja - EN](#jump4)
  - [ja - KO](#jump5)
- [Unsupervised](#jump6)
  - [Unsupervised : de - ??](#jump7)
  - [Unsupervised : ja - ??](#jump7)
<br><br><br>

## <a name="jump1">UMWE란?</a>
　[UMWE 논문을 정리한 글](https://nhandsome.github.io/dev/2020/10/02/dev-nlp-paper_UMWE/)을 참고.<br>
(조금 더 설명을 적어볼까?)

<br><br><br>


## <a name="jump2">Supervised(identical char)</a>
　**각 언어의 Embedding 공간에서 동일한 단어를 사용하는 쌍을 사전으로 사용**<br>
  Supervised Word Embedding Model은 `어떤 사전지식`을 필요로 한다. 잘 정리된 Dictionary 정보를 이용하는 것이 일반적이지만, identical char기능을 사용하면, `완전히 똑같은 단어 쌍들을 Dictionary로 사용`한다. 예를 들어 **숫자(1, 2, ..)나 기호(!, ?, ..), 영어단어(data, NY, iphone, ..)**와 같은 단어들은 어떤 언어의 문서에서나 공통적으로 사용될 가능성이 있다.

```python
def load_identical_char_dico(word2id1, word2id2):
  pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
  ...
  ...
  pairs = sorted(pairs, key=lambda x : word2id1[x[0]])
  dico = torch.LongTensor(len(pairs), 2)
  for i, (word1, word2) in enumerate(pairs):
    dico[i, 0] = word2id1[word1]
    dico[i, 1] = word2id2[word2]
```

> 
  - 중복되는 단어가 존재하는 경우, (w1, w1)의 단어쌍으로 Dictionary를 만든다.
  - word embedding info를 사용하여 사용빈도가 높은 순으로 정렬한다.

  - 한국어 - 영어의 경우,
  - 각각 20만개의 단어 중, 15518개의 단어가 중복
  - 이 15518개의 단어쌍을 기준으로 다른 단어쌍을 학습
<br><br>


**Optimize MPSR : 사전지식을 이용하여 학습을 진행**<br>
  지금 확보하고 있는 `어떤 사전지식`을 이용해서 **매핑을 조정**한다. 
  사전 지식이 있기 때문에, encoding decoding 하는 것으로 간단하게 학습을 진행 할 수 있다.

```python
def mpsr_step(self, stats):
        # loss
        loss = 0
        for lang1 in self.params.all_langs:
            lang2 = random.choice(self.params.all_langs)

            x, y = self.get_mpsr_xy(lang1, lang2, volatile=False)
            loss += F.mse_loss(x, y)
```
>
  - 랜덤으로 언어를 선택하고 mpsr을 계산
  - [mpsr](https://nhandsome.github.io/dev/2020/10/02/dev-nlp-paper_UMWE/#jump6)를 참고
  - mapping(x) = y' 와 y 의 Loss를 계산, mapping을 최적화


<br><br>


  다시 처음으로 돌아가서 >> 이번에는 indentical pair를 이용하지 않고, CSLS를 이용해서 사전지식을 뽑아온다. 
  CSLS는 가장 자신있는 단어 페어를 선택하는 것으로 생각하면 된다. 자세한 내용은 참고 ~~

  계속해서 이과정을 반복한다. 

  사용했던 슬라이드 사진 추가
  실험 결과들
  왜 일본어가 잘 되지 않았는지
  맵핑 결과를 보자



여기서부터 다시 쓰기ㅣㅣㅣㅣㅣㅣㅣㅣ
