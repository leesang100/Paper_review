# Predicting the Type and Target of Offensive Posts in Social Media

## Abstract
- 소셜 미디어에 모욕적인 콘텐츠가 증가함에 따라 모욕적인 메시지를 식별하는 연구들이 존재해옴
- 이전 연구에서는 문제를 전체적으로 고려하지 않고 혐오발언, 사이버 폭력 또는 사이버 공격과 같은 매우 구체적인 유형의 공격적 콘텐츠를 탐지하는데 초점을 맞춤
- 본 연구에서는 몇 가지 다른 종류의 공격 콘텐츠를 목표
- 특히 소셜 미디어에서 불쾌한 메시지의 유형과 대상을 식별하여 작업을 계층적으로 모델링
- 세밀한 3계층 주석 체계를 사용하여 불쾌한 내용에 대해 주석이 달린 트윗이 있는 새로운 데이터 세트인 공격 언어 식별 데이터 세트(OLID)를 소개

## Introduction
- 불쾌한 콘텐츠를 식별하는 문제를 해결하기 위한 가장 일반적인 전략 중 하나는 공격적 콘텐츠를 인식할 수 있는 시스템을 훈련시키는 것
- 이전 연구에서 제안된 서로 다른 접근 방식 간의 유사성을 분석하고 (욕설) 언어가 특정 개인이나 실체를 지향하는지, 또는 일반화된 그룹을 지향하는지, 그리고 남용되는 내용이 명시되어 있는지를 구별하는 유형이 필요하다고 주장
- 일반적으로 본 논문에서는 다음 세 가지 일반 범주를 포함하는 새로운 3단계 계층 주석 스키마를 제안함으로써 위의 아이디어를 확장

  - A: 공격 언어 탐지
  - B: 공격 언어 분류 
  - C: 공격 언어 대상 식별

- 개인을 대상으로 한 모욕이 일반적으로 사이버 괴롭힘으로 알려져 있고 그룹을 대상으로 한 모욕이 혐오 발언으로 알려져 있다는 점을 고려하면, OLID의 계층적 주석 스키마의 사용이 다양한 공격적 언어 식별 및 특성화 작업에 유용한 자원이 된다고 생각함

![image](https://user-images.githubusercontent.com/70500214/117045964-15832200-ad4b-11eb-8840-9e56195ca107.png)


## Hierarchically Modelling Offensive Content
- OLID 데이터 세트에서는 언어가 공격적인지 아닌지를(A), 유형(B) 및 대상(C)을 구분하기 위해 계층적 주석 스키마를 3단계로 분할

 - Level A: Offensive language Detection
   - Not Offensive(NOT): 공격 또는 욕설을 포함하지 않은 게시물
   - Offensive: 어떤 형태의 허용되지 않은 언어(욕설) 또는 직접적 간접적 공격적 언어를 포함하는 게시물, 모욕,협박,욕설 등
 
 - Level B: Categorization of Offensive Language
   - Targeted Insult(TIN): 개인,집단 또는 다른 사람에 대한 모욕/협박
   - Untargeted(UNT): 대상 없는 간접적 욕설과 욕을 포함하는 게시물

 - Level C: Offensive Language Target Identification
   - Individual(IND): 개인을 겨냥한 게시물
   - Group(GRP): 민족, 성별, 성적 지향성, 정치적, 종교적 신념 기타 공통 특성으로 인해 하나의 집단으로 간주되는 사람들을 대상으로하는 게시물
   - Other(OTH): 이전 두 범주에 속하지 않는 게시물(조직,상황,사건,이슈) 

## Data Collection
- 트위터 API를 사용하여 트위터에서 키워드 검색
- Table2 9개의 키워드를 사용하여 주석을 작성

![image](https://user-images.githubusercontent.com/70500214/117057700-bcba8600-ad58-11eb-8665-6eedddf763f7.png)

- 각 수준의 라벨에 대한 훈련 및 시험으로 데이터를 분류하는 방법은 표 3에 나와 있음
- OLID를 수집할 때 관찰한 주요 과제 중 하나는 각 클래스에 대해 충분한 수의 인스턴스를 포함하는 데이터 세트를 생성하는 것
- 다른 연구들에서와 유사하게 본 데이터에서 하위 작업 B 및 C의 크기에서 클래스 불균형이 일어남 

![image](https://user-images.githubusercontent.com/70500214/117058332-73b70180-ad59-11eb-9454-e3f5bfd1505c.png)

## Experiments and Evaluation
- 다양한 모델을 실험
- SVM,BiLSTM,CNN 모델을 이용

![image](https://user-images.githubusercontent.com/70500214/117058448-96491a80-ad59-11eb-938b-7ecb181b17a8.png)

## Conclusion and Future Work
- 공격 언어의 유형과 표적에 대한 주석을 가진 새로운 데이터 세트인 OLID를 제시
- 이 데이터는 소셜 미디어에서 위반 유형과 대상의 주석을 포함하는 최초의 데이터 세트이며, 흥미로운 연구 방향을 열어줌
