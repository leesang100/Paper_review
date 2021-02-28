# Feature Projection for Improved Text Classification
## Abstract
 - 감정 분류에서 긍정적인 감정을 나타내는 단어(good,nice)와 부정적인 감정을 나타내는 단어(bad,terrible)가 존재
 - 특정 클래스를 나타내지 않는 공통 features가 존재(voice,screen)
 - 딥러닝은 강력한 표현 학습을 통해 차별적 feature을 생성하는데 상당한 발전이 있었지만 본 논문에서는 개선할 필요가 있다고 생각
 - 본 논문은 기존 feature를 공통 feature의 직교 공간에 projection
 - 결과 projection은 공통 feature에 수직적이며 분류에 더 차별적
 - CNN,RNN,Transformer 및 bert에 적용

## Introduction
 - 텍스트 분류는 감정 분류,질문 분류,속임수 탐지와 같은 매우 광범위한 응용 분야를 가짐
 - 표현 학습은 딥러닝의 핵심 강점 중 하나
 - 표현 학습을 더욱 개선하여 표현을 분류에 대해 더 차별적으로 만들 것을 제안
 - 본 논문에서는 이전 연구에서 시도한 적 없는 새로운 방법으로 feature production을 탐구
 - 감정 분류에서 '가격','배터리'와 같은 단어는 어떤 감정도 나타내지 않음,즉 차별적이지 않음
 - 차별적이지 않은 단어들은 최종 분류에 대한 feature representation을 생성하기 위해 표현학습을 방해
 - feature 표현 학습을 개선 하여 분류에 대한 차별성을 높이기 위한 새로운 feature projection방법을 제안
 - Feature Purification Network(FP-NET)정의

![image](https://user-images.githubusercontent.com/70500214/109423951-f0a9c000-7a24-11eb-8bd1-e4d4d56c8786.png)

 - C-net은 GRL(Gradient Reverse Layer)를 사용하여 여러 클래스에서 공유되는 분류에 대한 차벌적인 힘이 거의 없는 공통 feature b를 추출 
 - 동시에 P-net은 전통적인 feature 추출기를 사용하여 입력 문장 또는 문서에 대한 특징 벡터 a를 학습
 - 그런 다음 벡터 a가 공통 feature b의 벡터 위에 projection되어 입력 문장의 고유한 공통 feature를 나타내는 projection 벡터 c를 얻음
 - 벡터 a를 공통 feature c의 직교 방향에 projection하여 분류를 위한 최종 순수 faeture를 생성 
 - 이 직교 프로젝트는 공통적인 feature를 없애고 시스템이 그러한 차별적 feature에만 초점을 맞추도록 하는 것이라는 것은 매우 분명하고 직관적

## Feature Purification Network
 - P-net은 입력 계층, Feature Extractor Fp,OPL 및 최종 분류 계층 Cp으로 4부분으로 구성
 - C-net 또한 입력 계층,Feature Extractor Fc,GRL 및 분류 계층 Cc으로 구성
 - 핵심 아이디어는 Fp에 의해 계산된 faeture 벡터 fp는 C-net의 Fc에 의해 추출된 feature 벡터 fc의 직교 방향에 투영
 - 즉, fp는 최종 분류를 위해 정제될 차별적 의미 공간에 투영

    ## C-net Module
     - C-net의 목표는 분류 작업에 대해 차별적이지 않은 입력 예제의 의미 정보인 공통 기능을 추출하는 것
     - 공통적인 feature은 문제의 모든 클래스의 공유하는 것
     - 분류자 Cc는 다른 클래스를 구분하는 데 이 분류기를 사용 하면 안됨
     - 공통 feature를 얻기 위해 Fc 뒤에 GRL를 추가하여 gradient 방향을 반전시킴
     - 이를 통해 클래스 간에 공유되는 공통 feature를 얻을 수 있음
 
    ## P-net Module
     - P-net의 목표는 먼저 입력 예제에서 전체 의미 정보를 추출한 다음 분류를 위해 정제된 의미 공간에 투영하는 것
     - 이를 위해 Fp에 의해 추출된 feature fp를 Fc에 의해 추출된 공통 feature fc의 직교 방향으로 투영을 수행
     - 공통 feature 벡터에 직교하는 feature 공간은 분류에 매우 효과적이고 순수해야 함
     - 전통적인 feature 벡터 fp를 이 직교 feature 공간에 투영하면 차별적 정보를 보존하고 분류 작업에 도움이 되지 않으며 심지어 혼란스러운 클래스의 공통 feature을 제거
     - OPL은 이 목표를 달성 하는 데 도움
     ![image](https://user-images.githubusercontent.com/70500214/109425257-c6f39780-7a2a-11eb-92be-11c09103a487.png)
     - 위 그림은 2차원 공간 예제를 사용한 OPL의 아이디어를 보여줌
     - fp는 전통적인 feature 벡터를 나타냄,fc는 공통 feature 벡터를 나타내고 fp*는 projection feature 벡터이며,fp~는 최종 직교 projection feature 벡터


## Experiments
 - Algorithm
 
 ![image](https://user-images.githubusercontent.com/70500214/109425449-9d873b80-7a2b-11eb-8a9f-a0169e9155e5.png)

## Datasets
 - MR : 영화 리뷰 데이터(positive,negative)
 - SST2 : Sentiment Treebank(negative,positive)
 - TREC : 질문 분류 데이터
 - SNLI : 널리 사용되는 텍스트 기반 데이터
 
 ![image](https://user-images.githubusercontent.com/70500214/109425555-1edece00-7a2c-11eb-8d54-cf400b782a14.png)

## Baselines
 - LSTM
 - CNN
 - Transformer
 - Bert

![image](https://user-images.githubusercontent.com/70500214/109425580-4170e700-7a2c-11eb-9bce-898f8e383221.png)

 - 기본 베이스 모델과 FP-net을 추가한 모델의 정확도를 각각 비교
 - 다른 모델과 비교하여 BERT 모델의 성능이 높으며 FP-net을 추가한 BERT모델의 정확도가 가장 높게 나옴

## Ablation Experiments and Analysis
![image](https://user-images.githubusercontent.com/70500214/109426455-1fc62e80-7a31-11eb-8f09-56d4835ef717.png)

 - FP-Net의 각 구성 요소의 효과를 분석하기 위해 두가지 절제 실험 수행
 - FP+CNN-G(O,G-O)는 CNN을 기능 추출기로 사용하는 동안 제거된 GRL(OPL,GRL과 OPL 모두)을 사용하여 FP-Net을 나타냄
 - 실험의 파라밑는 정확히 같음
 - GRL 또는 OPL을 제거하든 동시에 제거하든 정확도는 완전한 FP-Net에 비해 크게 떨어짐
 - FP+CNN-O 실험에서는 OPL을 제거하고 GRL을 유지하는데, 이는 직교 투영 대신 fp-fc를 사용한다는 것을 의미, 결과는 정확도가 감소하여 투영 작업이 필요하다는 것을 의미

![image](https://user-images.githubusercontent.com/70500214/109426721-6a947600-7a32-11eb-8ea9-f0714fd9e6ce.png)

 - FP-Net에 의한 정확도 향상은 매개 변수 수의 증가 때문이 아님을 보여줌
 - 기존 CNN과 transformer의 매개 변수를 두배로 늘리고 FP+CNN,FP+Trans와 비교
 - Dp는 이중 파라미터 크기를 의미, 예를 들어 TransDp는 기준선의 transformer 블록 수를 3개에서 6개로 늘림
 - 실험 결과 모델의 매개 변수 수를 늘리게 되면 정확도는 약간 항샹되지만 제안된 모델과 여전히 큰 차이를 보임

## Conclusion
 - 텍스트 분류에 대한 표현을 개선하기 위해 FP-NET을 제안
 - feature projection을 기반으로 함
 - 분류에 대해 차별적이지 않은 공통 feature를 식별하기 위해,그리고 공통 feature의 직교 방향에 전통적인 feature을 투영하는 feature 투영을 위해 두개의 하위 네트워크 사용
