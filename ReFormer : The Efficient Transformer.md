# ReFormer : The Efficient Transformer
 - 구글 리서치 팀,2020

# Abstract

 - transformer의 효율성 향상을 위한 두가지 기술 소개
 - dot-product attention을 locality-sensitive hashing으로 대체 사용
 - standard residuals에서 reversible residual로 대체 사용
 - 결과 모델을 Reformer라고 부르며 동등한 성능에서 메모리 효율이 훨씬 높고 긴 시퀀스 데이터에서 훨씬 빠름

# Introduction

 - transformer는 좋은 성능을 나타내고 널리 사용되지만 모델을 사용하는데 막대한 자원이 필요하며 메모리 비용이 많이 듬
 - Transformer의 문제점
   - N 레이어가 있는 모델의 메모리는 역 전파를 위해 activations가 저장되어야 한다는 사실 때문에 단일 레이어 모델보다 N배 큼
   - 중간 피드-포워드 레이어의 깊이 d ff가 attention activations의 깊이 d model보다 훨씬 크므로 메모리 사용의 많은 부분을 차지
   - 메모리 복잡도가 높음
 - Transformer의 해결방안
   - Reversible layer를 사용하여 activation의 단일 복사본만 저장, N 요인이 사라짐
   - 피드-포워드 layer 내부의 activation을 분할하고 chunk 처리하여 dff 요소를 제거하고 layer 내부의 메모리를 절약
   - locality-sensitive hashing에 기초한 attention 계산은 O(LlogL)로 대체

# Locality-Sensitive Hashing Attention

   <h3>Memory-efficient attention</h3>
    - 어텐션 계산시 Q, K, V 모두 [batch_size, length, dmodel] 의 크기를 가진다고 가정
    - length가 매우 길 경우 문제는 QK^T
    - 이것은 긴 시퀀스에서의 성능을 방해시키는 요인이 됨
    - 모든 QK^T 행렬 값을 모두 사용하는 것은 중요 요소가 되지않음
    - 실제로 pi하나에 대해서 각각 계산해 두었다가,백 워드 패스 때 그래디언트 계산시 사용하면 더 효율적일 수 있음


   <h3>Hashing attention</h3>
    - LSH attention을 위해 2개의 텐서를 가지고 시작(Q=K,V)
    - attention 계산 방식은 유지
    - softmax(QK^T)에 관심을 두고 연구
    - softmax는 가장 큰 값 들에 의해 결정되어 지므로, 각 qi별로 가장 가까운 key들에만 포커스 해도 됨
    - key중에서 가장 가까운 이웃을 찾는 방법이 무엇이 있는지 고민
    
   <h3>Locality sensitive hashing</h3>
   
   ![image](https://user-images.githubusercontent.com/70500214/112457224-e4542100-8d9e-11eb-8fcf-566aef6a6c35.png)

   - 고차원 공간에서 가장 가까운 이웃을 빠르게 찾는 문제는 LSH으로 해결
   - 버켓 사이즈의 1/2개 만큼의 랜덤 벡터(dk 사이즈)를 만들어 두고 그걸 곱하면 나오는 값이 미리 정해둔 버켓들중 같은 버켓에 계속 같이 들어가면 비슷한 것이라는 가정
    
   <h3>LSH attention</h3>
   
   ![image](https://user-images.githubusercontent.com/70500214/112457480-2a10e980-8d9f-11eb-89ba-9b663c879610.png)
   
   -  single query position i at a time
   -  Pi = i 번째 Query가 attention할 수 있는 key set
   -  z= partition function
   -  각 query는 자신의 위치보다 작은 key에만 attention을 할 수 있다는 의미
    
# Reversible Transformer

 - RevNet의 아이디어를 transformer에 적용
 - revnet block 안의 attention과 feed-forward layer를 combine
 - F가 attention layer, G가 feed-forward layer가 되는것 
 ![image](https://user-images.githubusercontent.com/70500214/112458272-f84c5280-8d9f-11eb-8a5c-39698040eaee.png)
 
 ![image](https://user-images.githubusercontent.com/70500214/112458289-fbdfd980-8d9f-11eb-8486-60c94fd40b34.png)
 
 - Reversible Transformer는 각 layer 안에 activation을 저장할 필요가 없음

# Chunking
 
 - feed-forward layer 내 계산은 시퀀스 내 위치 간 완전한 독립적이기 대문에 여러 개의 chunk로 split할 수 있음
 - 모든 위치에 대한 작업을 병렬로 처리하고 한번에 하나의 chunk로 작동하여 메모리를 줄일 수 있음
 

# Experiments
 - 기존 transformer와 위 아이디어를 비교한 메모리와 시간 복잡도 

![image](https://user-images.githubusercontent.com/70500214/112458947-a9eb8380-8da0-11eb-9e67-3efb58d0f506.png)

 - enwik8과,imagenet 64를 가지고 실험
 - bpd는 bit per dim, 하나의 차원을 저장하는데 얼마의 비트가 필요한지

 ![image](https://user-images.githubusercontent.com/70500214/112460043-d6ec6600-8da1-11eb-8ffc-fee55ea881f8.png)
 
 ![image](https://user-images.githubusercontent.com/70500214/112459468-36964180-8da1-11eb-801a-9d042a993e66.png)

 - key-query 공유한 모델에 성능차이가 거의 없음
 - Reversible layer의 영향은 크게 미치지 않음

 ![image](https://user-images.githubusercontent.com/70500214/112460537-5bd77f80-8da2-11eb-8746-1830fc7ccfe8.png)
 
 - 병렬 Hashing을 많이 할 수록 Transformer와 성능이 비슷함


 ![image](https://user-images.githubusercontent.com/70500214/112460744-92ad9580-8da2-11eb-8e45-8dad43796c47.png)

 - Reformer Layer를 증가시키면 성능이 향상

 ![image](https://user-images.githubusercontent.com/70500214/112460829-aa851980-8da2-11eb-97e7-a259c758f08b.png)

 - Transformer는 문장 길이가 길수록 계산향이 증가
 - Reformer는 문장의 길이릐 변화와 속도에 큰 연관성이 없음

# Conclusion
 - Reformer는 긴 시퀀스에서 효율적으로 실행할 수 있는 아키텍처와 transformer의 모렐링 용량을 결합
 - Reformer의 Transformer는 매우 긴 시퀀스 외에도 시계열 예측,음악,이미지, 비디오 생성과 같은 영역으로 가져올 가능성을 가짐

# 느낀점
 - Reformer는 긴 시퀀스에서 좋은 성능을 보여주었지만 짧은 시퀀스에서는 어떤 결과를 가져오는지 얘기가 없었다. 짧은 시퀀스에서도 좋은 성능을 낼 수 있는 방법은 무엇인지 고민해 보아야겠다.
 - Reformer를 활용한 BERT가 만들어 질 수 있을까 생각을 해보았다. 
 - 기존 BERT의 성능도 우수하지만 긴 시퀀스에 대해서 더 잘 적용되면 좋지 않을까 생각해보았다.
 - 성능 측정을 텍스트 말고 이미지 데이터에서도 측정 하였는데 나쁘지않는 성능을 나타내서 다른 분야에서도 쓸수 있을까 생각이 든다.
 - 하지만 논문에서는 성능 측정을 여러 많은 데이터로 하지않아서 다른 데이터에서는 어떤 성능을 나타내는지 의문점이 든다.
