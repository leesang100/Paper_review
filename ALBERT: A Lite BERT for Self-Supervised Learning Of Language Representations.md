# ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS

# Abstract
 - 자연어 처리 작업에서 사전 교육을 할 때 모델의 크기를 늘리면 다운스트림 작업에서 성능이 향상되는 경우가 많음
 - 하지만 GPU/TPU  메모리 제한과 긴 학습 시간 때문에 어느 시점부터는 모델의 크기를 증가 시키는 것의 어려움이 있음
 - 이런 문제를 해결하기 위해 메모리 소비를 낮추고 BERT의 훈련 속도를 높이기 위한 두가지 파라미터 감소 기법을 제시
 - 제안된 방법이 BERT에 비해 훨씬 더 잘 확장되는 모델로 이어진다는 것을 보여줌
 - 문장 간의 일관성을 모델링 하는데 초점을 맞추고 다중 문장 입력으로 다운스트림 작업에 일관되게 도움이 되는 SELF-SUPERVISED LOSS 사용
 - BERT-LARGE에 비해 파라미터가 적으면서 GLUE,RACE,SQuAD 벤치마크에서 SOTA를 달성

# Introduction
 - 사전 학습은 언어 표현 학습에서 좋은 성능을 내고 있음
 - RACE테스트에서 제시한 44.1%보다 45.3%가 높은 고성능 사전 학습된 언어 표현을 모델의 향상률을 보여줌 
 - 이런 증거는 대규모 네트워크가 최첨단 성능을 달성하는데 매우 중요하다는 것을 보여줌
 - 대형 모델로 사전 교육하고 작은 모델로 증류하는것이 일반적 관행
 - 더 나은 NLP모델을 갖는 것이 더 큰 모델을 갖는 것 만큼 쉬운 일인가의 답변은 사용가능한 하드웨어의 메모리 제한을 드러냄
 - 최신 모델의 파라미터가 수억개 또는 수십억개인 경우가 많기 때문에 모델을 확장할때에 이런 한계에 부딪히기 쉬움
 - communication overhead(생산적인 작업을 수행하는 대신 팀과 커뮤니케이션하는 데 소비하는 시간의 비율)는 모델의 파라미터 수에 정비례하기 때문에 분산 학습에서도 학습속도에 크게 방해될 수 있음
 - 앞에서 언급한 문제에 대한 솔루션으로는 모델의 병렬화랑 메모리 관리가 있음
 - 이런 솔루션은 메모리 제한 문제를 해결하지만 communication overhead는 해결 못함
 - 논문에서는 기존의 BERT 아키텍처보다 파라미터가 훨씬 적은 A Lite BERT(ALBERT)아키텍처를 설계하여 앞에서 언급한 모든 문제 해결
 - ALBERT는 모델의 크기에 주요 장애물을 줄이는 두가지 파라미터 변수 감소 기술을 통합
    1. 분리된 임베딩 매개변수화
       - 큰 어휘 임베딩 행렬을 두개의 작은 행렬로 분해하여 hidden layer의 크기와 어휘 임베딩 크기를 분리
       - 분리를 통해 어휘 임베딩의 파라미처 크기를 크게 늘리지 않고도 hidden size를 쉽게 확장 가능
    2. 교차 계층 파라미터 공유
       - 네트워크 깊이와 함께 파라미터가 커지는 것을 방지
 
 - 두 기법 모두 성능을 크게 손상시키지 않고 BERT에 대한 파라미터 수를 크게 줄여 파라미터 효율성을 향상
 - BERT-large와 유사한 ALBERT 구성은 파라미터 변수가 18배 더 적으며 약 1.7배 더 빠르게 훈련 가능
 - 파라미터 감소 기술은 훈련을 안정시키고 일반화에 도움을 주는 정규화의 한 형태로도 작용
 - ALBERT의 성능을 더욱 향상시키기 위해 sentence-order prediction(SOP)에 대한 self-supervised도 도입
 - SOP는 문장 간 일관성에 중심을 두고 원래 BERT에서 제안된 NSP 손실의 비효율성을 해결하도록 설계
 - 설계의 결과로 BERT-Large보다 파라미터가 적지만 훨씬 더 나은 성능을 달성하는 더 큰 ALBERT로 확장 할 수 있음
 - GLUE,SQuAD,RACE 벤치마크에 대한 SOTA결과 수립


# RELATED WORK

 <h3>Scaling Up Representation Learning For Natural Language</h3>
 
 - 자연어의 학습 표현은 NLP 작업에 유용한 것으로 나타나며 널리 채택됨
 - 지난 2년 동안 가장 중요한 변화 중 하나는 표준 또는 상황에 맞는 사전 학습에서 전체 네트워크 사전 학습 다음에 작업별 미세 조정으로 전환하는 것
 - 이런 작업에서 모델 크기가 클수록 성능이 향상됨이 나타남
 - 자연어 이해 작업에서 hidden size,hidden layer,attention head가 더 많은 것이 더 좋은 성능을 나타냄
 - 하지만 모델 크기와 계산 비용 문제에서 hidden size를 1024에서 멈춤
 - 특히 GPU/TPU 메모리 제한 측면에서 계산 제약으로 인해 대형 모델로 실험하기 어려움
 - 최신 모델들이 수 억단위에 파라미터를 가진 것을 고려하면 쉽게 메모리 제한에 도달할 수 있음
 - 이 문제를 해결하기 위해 과거에는 추가 순방향 패스의 비용으로 하위 선형 메모리 요구 사항을 줄이기 위해 그래디언트 체크포인트라고 하는 방법을 제안,중간 활성화를 저장할 필요 없도록 다음 계층에서 각 계층의 활성화를 재구성하는 방법 제안,모델 병렬화를 사용할 것을 제안
 - 대조적으로 본 논문의 파라미터 감소 기술은 메모리 소비를 줄이고 훈련 속도를 증가 시킴

 <h3>Cross-Layer Parameter Sharing</h3>
 
 - Layer간 파라미터를 공유하는 아이디어는 transformer 아키텍처를 통해 탐구 되었지만 이 연구는 사전 학습/미세조정보다는 표준 인코더-디코더 작업에 대한 훈련에 초점을 둠
 - cross-layer parameter를 가진 네트워크가 표준 transformer보다 언어 모델링 및 주제-동사 수일치에서 보다 더 나은 성능을 보임
 - 최근 transformer 네트워크를 위한 심층 평형 모델(DQE)을 제안하고 DQE가 특정 layer의 입력 임베딩과 출력 임베딩이 동일한 평형점에 도달할 수 있음을 보여줌
 - 임베딩이 수렴하기보다는 진동하다는 것을 보여줌
 - 파라미터 공유 transformer와 표준 transformer를 결합하여 표준 transformer의 파라미터 수를 더욱 증가 시킴

 <h3>Sentence Ordering Objectives</h3>
 
 - Albert는 텍스트의 연속된 두 세그먼트의 순서를 예측하는 것에 기초한 사전 학습 손실을 사용
 - Skip-thought와 FastSent 문장 임베딩은 문장의 인코딩을 사용하여 인접 문장의 단어를 예측함으로써 학습됨
 - 문장 임베딩 학습의 다른 목표로는 근처 이웃들만이 아닌 미래 문장 예측과 명시적 담화 표지어 예측이 있음
 - 논문의 loss는 두개의 연속된 문장의 순서를 결정하기 위해 문장 임베딩을 학습하는 문서 순서 목표와 유사
 - 대부분의 작업과 달리 논문의 loss는 문장보다 텍스트 세그먼트에 정의됨
 - BERT는 쌍의 두번째 세그먼트가 다른 문서의 세그먼트와 스왑 되었는지 여부를 예측하여 loss를 사용함
 - 실험에서 이러한 loss을 비교하고 sentence ordering이 더 어려운 사전 학습 작업이며 특정 다운스트림 작업에 더 유용하다는 것을 발견
 
# The Elements of Albert

 - Albert에 대한 설계 제시 원본 BERT 아키텍처의 해당 구성에 대한 비교를 제공

 <h3>Model Architecture Choices</h3>
  
  - Albert 아키텍처의 뼈대는 GELU function을 가진 Transformer의 인코더를 사용(BERT와 유사)
  - BERT의 표기 규칙을 따르며, 임베딩의 크기를 E, 인코더 layer의 수를 L, 숨겨진 크기(hidden size)를 H로 표시
  - 이어서 feed-forward/filter 크기를 4H로, attention head의 수를 H/64로 설정
  - BERT의 설계 선택에 대해 Albert는 세 가지 주요 기여 존재

    <h4>Factorized embedding parameterization</h4>
    
    - 모델링 관점에서 WordPiece 임베딩은 context-independent 표현을 학습하는 반면, hidden layer 임베딩은 context-dependent 표현을 학습하는 것
    - 자연어 처리는 vocabulary 크기 V가 커야 성능이 잘 나옴
    - E=H일 경우, H를 늘리면 크기가 VxE인 임베딩 행렬의 크기가 증가
    - 이로 인해 수십억개의 파라미터가 있는 모델이 쉽게 생성될 수 있으며, 대부분은 학습에서 드물게 업데이트 
    - Albert는 임베딩 파라미터를 factorizatoin하여 두개의 작은 행렬로 분해 
    - one-hot 벡터를 크기 H의 hidden space의 직접 투영하는 대신 먼저 크기 E의 낮은 차원 임베딩 공간에 투영한 다음 hidden space에 투영
    - 이런 분해를 사용하여, 임베팅 파라미터를 O(VxH)에서 O(VxE+ExH)로 줄임
    - 파라미터 감소는 H>>E일때 유의함
    - 모든 Wordpiece에 동일한 E를 사용하는 것을 선택
    - 왜냐하면, 다른 단어의 임베딩 크기를 갖는 것이 중요한 전체 워드 임베딩에 비해 문서에 훨씬 고르게 분포하기 때문

    <h4>Cross-layer parameter sharing</h4>
    
    - Albert의 경우 파라미터 효율성을 향상시키는 또 다른 방법으로 cross-layer 파라미터 공유를 제안
    - Albert는 계층 간에 모든 파라미터를 공유하는 것
    - design decision을 Section 4.5에서 실험
    - 그림1는 BERT-large 및 Albert-large를 사용하여 각 layer에 대한 입출력 임베딩의 L2distance및 cosine similarity를 보여줌
    - 위 결과를 통해 가중치 공유가 네트워크 파라미터 안정화에 영향을 미친다는 것을 보여줌
    - BERT의 비해서 두 값은 모두 감소하지만 0으로 수렴되지 않음
    
    ![image](https://user-images.githubusercontent.com/70500214/111804150-26024900-8913-11eb-82af-8b98e8b7a312.png)
    
    <h4>Inter-sentence coherence loss</h4>
    
    - BERT는 MLM loss와 NSP loss를 사용
    - 그러나 후속 연구에서 NSP의 영향을 신뢰할 수 없으며 이를 제거
    - NSP의 비요율성의 주된 이유는 MLM과 비교할때 쉬운 task이기 때문이라고 추측
    - 본 연구는 문장 간 모델링이 언어 이해의 중요한 측면이라고 주장하지만, 일관성에 기반한 loss를 제안
    - Albert의 경우 문장 순서 예측(SOP)loss를 사용하여 주제 예측을 피하고 대신 문장 간 일관성을 모델링하는 데 초점을 둠
    - SOP loss는 BERT(동일한 문서의 두 연속 세그먼트)와 동일한 기술을 positive의 예로서 사용하고,negative의 예로서 동일한 두 연속 세그먼트를 사용하지만 순서가 바뀐채로 사용
    - 이것은 모델이 담화 수준의 일관성 속성에 대한 세밀한 차이를 학습하도록 강요
    - NSP는 SOP 작업을 전혀 해결 할  수 없는 것으로 밝혀짐
    - SOP는 NSP 작업을 합리적으로 해결할 수 있으며,이는 잘 못 정렬된 일관성 단서를 분석하기 때문
    - 결과적으로 Albert 모델은 다중 문장 인코딩 작업에 대한 다운스트림 작업 성능을 지속적으로 향상 시킴
   
     ![image](https://user-images.githubusercontent.com/70500214/111806536-6ebb0180-8915-11eb-9afe-930225d3970d.png)
     
     <h4>Model Setup</h4>
    
    - Table 1에서 비교할 수 있는 파라미터 설정을 사용하여 BERT모델과 Albert모델 간의 차이를 제시
    - 위에서 설명한 설계 때문에,Albert 모델은 해당 BERT 모델에 비해 파라미터 수가 훨씬 적음
    - Albert-large는 bert-large에 비해 약 18개 적은 파라미터를 가지고 있음
    - H=2048을 사용하는 Albert-xlarge 구성은 60M 파라미터를 가지며,H=4096을 사용하는 Albert-xxlarge 구성은 235M 파라미터,즉 BERT-large 파라미터의 약 70%를 가짐
    - Albert-xxlarge의 경우 계산적 비용을 생각하여 24 layer대신 12 layer 사용
    - 파라미터의 효율성의 개선은 Albert의 설계 선택의 가장 중요한 장점
    - 장점을 나타내기 전에 실험 설정을 더 자세히 소개


# Experimental Results

   <h4>Experimanetal Setup</h4>
    
   - BookCorpus와 영어 위키 백과를 사용하여 BERT 설정에 따름
   - 입력을 [CLS] x1 [SEP] x2 [SEP]로 포맷
   - x1,x2는 두개의 세그먼트
   - 최대 입력 길이를 512로 제한하고 10%확률로 512보다 짧은 입력 시퀀스를 무작위로 생성
   - BERT와 마찬가지로 XLNet에서와 같이 SentencePiece를 사용하여 토큰화된 30000개의 vocabulary 사용
   - n-gram 마스킹을 사용하여 MLM 대상에 대한 마스크된 입력을 생성, 각 n-그램 마스크의 길이는 무작위로 선택
   - 논문에서는 n-그램의 최대길이를 3으로 설정(n=3)
   - 모든 모델 업데이트는 4096의 배치 크기와 learning rate=0.00176, Lamb optimier를 사용
   - 125000 스텝에 대해 모든 모델 훈련
   - Cloud TPU V3에서 교육 수행했으며 TPU의 수는 모델 크기에 따라 64-512개 까지 다양

   <h4>Evaluation Benchmarks</h4>
    
   <h5>Intrinsic Evaluation(고유성 평가)</h5>
    
   - SQuAD 및 RACE의 개발 세트를 기반으로 개발 세트 만듬
   - MLM 및 문장 분류 작업에 대한 정확도를 보고
   - 모델이 수렴되는 방식을 확인 하는데만 사용
 
   <h5>Downstream Evaluation</h5>
   
   - 세가지 인기 벤치마크에서 모델 평가
   - 일반 언어 이해 평가(GLUE)벤치마크, 스탠포드 질문 답변 데이터 세트(SQuAD)의 두가지 버전 및 검사로부터 읽기 이해(RACE) 데이터 세트
   - early stopping을 수행하며 리더보드에 최종 비교를 제외한 모든 비교를 보고,테스트 세트 결과도 보고함
   - GLUE 데이터 세트의 경우 5개의 실행에 대한 중위수 보고

   <h5>Overall Comparison Between BERT and Albert</h5>
   
   - 파라미터 효율의 향상은 Table2에서와 같이 Albert-large의 설계 선택의 가장 중요한 이점을 보여줌
   - BERT-large의 파라미터의 약 70%만 사용하여 다음과 같은 몇가지 대표적인 다운스트림 작업에 측정할때 BERT-large보다 상당한 개선을 달성
   - 동일한 학습 환경에서 BERT 모델에 비해 더 높은 데이터 처리량을 가짐
   - BERT-lare를 기준으로 삼는다면, Albert-large가 데이터를 통해 반복하는 속도가 약 1.7배 더 빠르지만, Albert-xxlarge는 더 큰 구조 댸문에 약 3배 느리다는 것을 관찰
   - 각 설계 선택의 개별 기여도를 정량화 하는 절제 실험 수행


   ![image](https://user-images.githubusercontent.com/70500214/111812188-36b6bd00-891b-11eb-9836-e8e074d8c0d6.png)

  <h5>Factorized Embedding Parameterization</h5>
   
   - Table3은 Albert 기반 구성 설정을 사용하여 동일한 대표적인 다운 스트림 작업을 사용하여 vocabulary 임베딩 크기 E의 변경효과를 보여줌
   - BERT와 같은 not-shared에서는 임베딩 크기가 클수록 성능이 향상되지만 크게 향상되지는 않음
   - Albert의 All-shared에서는 128크기의 임베딩이 가장 좋은 것으로 보임
   - 이 결과는 향후 모든 설정에서 임베딩 크기 E=128로 사용,추가 확장 필요하면 확장
   - 
   ![image](https://user-images.githubusercontent.com/70500214/111813028-0a4f7080-891c-11eb-9a61-efc55cbcecde.png)

  <h5>Cross-Layer Parameter Sharing</h5>
   
   - Table4는 두가지 임베딩 크기(E=768,E=128)를 가진 Albert를 사용하여 다양한 layer 간 파라미터 공유 전략을 위한 실험 제시
   - 모든 파라미터 공유(albert), 파라미터 공유하지않음(bert) 및 attention 파라미터만 공유되거나 FFN 파라미터의 공유를 비교
   - All-Shared 전략은 두 조건 모두에게 성능을 저하시키지만 E=128의 경우 E=768에 비해 덜 심각
   - 대부분의 성능 저하가 FFN layer 파라미터를 공유하는 것에서 오는것 처럼 봉지만 attention 파라미터를 공유하면 E=128(+0.1),E=768(-0.7)로 감소
   - 파라미터를 여러 layer 간에 공유하는 다른 전략도 존재
   - 예를 들어, L layer를 크기 M의 N 그룹으로 나누고 각 크기 M 그룹은 파라미터를 공유 할 수 있음
   - 전반적으로 그룹 크기 M이 작을수록 성능이 더 좋다는것을 보여줌 그러나 크기 M을 줄이면 전체 파라미터가 크게 증가
   - 모든 공유 전략을 기본 옵션으로 선택
   - 
  ![image](https://user-images.githubusercontent.com/70500214/111814391-c0678a00-891d-11eb-8a8b-cd5b83378adb.png)
  
  <h5>Sentence order prediction(SOP)</h5>
  
   - Albert를 사용하여 none,NSP,SOP라는 추가적인 문장 간 손실에 대한 세가지 실험 조건을 비교
   - 전체 과제와 다운 스트림 과제 모두 해당
   - NSP loss가 SOP 작업에 긍정적인 힘을 가져오지 않는 다는 것을 보여줌
   - 따라서 NSP는 결국 주제 이동만 모델링하게 된다고 결론을 내릴 수 있음
   - SOP loss는 NSP 작업에 비교적 잘 해결,SOP 작업은 훨 씬 더 좋은 성과를 냄
   - 더욱이 SOP loss는 다중 문장 인코딩 작업의 다운 스트림 작업 성능을 약 1%향상 하는것으로 나타남
   - 
  ![image](https://user-images.githubusercontent.com/70500214/111815079-98c4f180-891e-11eb-93cd-44fdecdf44cc.png)
  
  <h5>What If We Train For The Same Amount Of Time?</h5>
   
   -  Table2의 속도 향상 결과는 BERT-large의 데이터 처리량이 Albert-xxlarge에 비해 약 3.17배 높다는 것을 나타냄
   -  긴 훈련은 대개 더 나은 성능을 이끌기  때문에, 데이터 처리량을 제어하기 보단 실제 훈련 시간을 제어하는 비교를 수행
   -  400k 훈련 단계 이후의 BERT-large 모델의 성능을 비교 이는 125k 훈련 단계를 가진 Albert-xxlarge 모델을 훈련하는데 필요한 시간과 대략 같음
   -  거의 동일한 시간동안 훈련한 후 ,Albert-xxlarge는 bert-large보다 훨씬 좋은 결과를 나타냄
 
  ![image](https://user-images.githubusercontent.com/70500214/111815636-420be780-891f-11eb-88dc-a3014c085b36.png)

  <h5>Additional Training Data And Dropout Effect</h5>
   
   - 지금까지는 위키백과,BookCorpus 데이터 세트만 사용
   - XLNET과 Roberta에서 모두 사용하는 추가 데이터의 영향에 대한 측정을 보고
   -  fig2에서는 추가 데이터가 없는 상태와 있는 상태에서 편차 MLM 정확도를 표시,추가 데이터가 있는 성능이 더 증가함
   -  SQuAD 벤치마크를 제외하고 다운스트림 작업에서 성능 향상을 관찰
   -  1M step을 학습한 후에도 가장 큰 모델은 여전히 학습 데이터에 맞지 않는다는 점에 유의
   -  모델 용량을 더욱 늘리기 위해 dropout을 제거
   -  fig2에서 dropout을 제거하면 MLM 정확도가 크게 향상된다는 것을 보여줌
   -  경험적 및 이론적 증거가 있으며,이는 컨볼류션 신경망의 배치 정규화와 중퇴 조합이 해로운 결과를 가질 수 있음을 보여줌
   -  dropout이 대형 transformer 기반 모델에서 성능을 해칠 수 있다는 것을 처음으로 보여줌
   -  그러나 Albert의 기본 네트워크 구조는 transformer의 특별한 경우이며, 이 현상이 다른 transformer 기반 아키텍처와 함께 나타나는지 여부는 확인 불가능
   
  ![image](https://user-images.githubusercontent.com/70500214/111817258-26094580-8921-11eb-9a08-427bc16c80b4.png)
  
  ![image](https://user-images.githubusercontent.com/70500214/111817279-2dc8ea00-8921-11eb-9b2b-6ba9e8274f6e.png)
  
  <h5>Current State-of-the-art on NLU Tasks</h5>
  
   - BERT에서 사용된 training데이터와roberta에서 사용된 추가 데이터를 활용
   - 미세 조정을 위한 두가지 설명, 단일 모델과 앙상블에서 최첨단 결과를 보고
   - 두가지 모두에서 single-task 미세조정을 수행
   - 5회 실행시 중앙값 결과를 보고
   - 단일 모델 Albert은 최고의 성능 설정을 통합, Albert-xxlarge 구성은 MLM과 SOP loss를 결합간 것이며 드랍아웃은 없음
   - GLUE 및 RACE 벤치마크의 경우 12 layer 및 24 layer 아키텍처를 사용하여 서로 다른 학습 단계에서 미세 조정되는 앙상블 모델에 대한 예측을 평균
   - SQuAD의 경우, 다중 확률을 갖는 범위에 대한 예측 점수를 평균
   - 단일 모델과 앙상블 결과는 모두 Albert가 세가지 벤치마크에서 SOTA를 달성
  ![image](https://user-images.githubusercontent.com/70500214/111819257-7bdeed00-8923-11eb-8cd7-463d6dcebda3.png)
  
  ![image](https://user-images.githubusercontent.com/70500214/111819313-86998200-8923-11eb-8fba-cd6806647e0d.png)
  
# Discussion
 - Albert-xxlarge는 BERT-large보다 파라미터가 적고 결과가 상당히 우수하지만,구조가 더 크기 떄문에 계산 비용이 많이듬
 - 다음 단계로는 Albert의 추론 속도를 높이는 것
 - SOP가 더 나은 언어표현으로 이어지는 설득력 있는 증거를 가지고 있지만, self-supervised 학습에서 결과적인 표현에 대한 추가적인 표현을 생성할 수있는 더 많은 차원이 있을 수 있다고 가정


# 느낀점
 - albert의 base 모델은 파라미터 수도 적고 속도도 bert-base보다 빠르지만 성능에서 조금 아쉬운점이 있다.
 - dropout이 있는거보다 없는게 더 좋은 성능을 낼수 있다는게 신기했다 어떤 상황에서 dropout이 안좋은 성능을 내는지 자세히 

 
