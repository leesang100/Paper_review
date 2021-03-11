# ATTENTION IS ALL YOU NEED


# ABSTRACT  
 - sqeunce transduction models는 인코더와 디코더를 포함하는 복잡한 recurrent와 convolutional 신경망에 기초
 - 성능이 가장 좋은 모델은 attention 메커니즘을 통해 인코더와 디코더를 연결
 - recurrent와 convolutional 신경망을 제거한 attention 메커니즘만을 기반으로한 새로운 네트워크 아키텍처인 트랜스포머를 제안
 - 두 가지 기계 번역에 대한 실험에서 모델은 품질이 우수하면서도 더 병렬화되고 훈련 시간이 훨씬 적게 소요된다는 것을 보임
 - 2014년 WMT English to German 번역대회에서 28.4 BLEU를 달성하면서 기존 최고 모델보다 2BLEU이상 향상하며 state of the art을 달성
 - Transformer가 크고 제한된 훈련 자료로 영어 구문 분석에 성공적으로 적용함으로써 다른 작업에 잘 일반화 되었음을 보임


# Introduction
 - 언어 모델링 및 기계 번역과 같은 시퀀스 모델링 및 변환 문제에서 최신 접근 방식으로서 RNN,LSTM,GRU은 확고히 확립됨
 - 그 후 많은 노력이 recurrent 언어모델과 인코더-디코더 아키텍처의 경계를 계속적으로 밀어 붙임
 - recurrent모델은 일반적으로 입력 및 출력 시퀀스의 기호 위치를 따라 계산
 - 계산 시간의 단계에 위치를 정렬하면 이전 은닉 상태 ht-1의 함수로서 은닉 상태 ht의 시퀀스를 생성하고 위치 t에 대한 입력을 생성
 - 이런 sequential 특징은 병렬화가 불가능,시퀀스 길이가 중요
 - 최근 연구는 factorization tricks와 confitional computation을 통해 계산 효율의 상당한 향상을 달성
 - 그러나 sequential 계산의 근본적인 제약은 남아 있음
 - attention 메커니즘은 다양한 작업에서 설득력 있는 시퀀스 모델링 및 전이 모델의 필수적인 부분이 되었으며,입력 또는 출력 시퀀스의 거리에 관계 없이 종속성을 모델링
 - 본 연구에서 recurrent을 사용하지않는 대신 입력과 출력 사이의 global dependencies을 끌어내기 위해 전적으로 attention 메카니즘에 의존하는 모델 아키텍쳐인 transformer 제안
 - 트랜스포머는 훨씬 더 많은 병렬화를 가능하게 하며, 8개의 P100 GPU에서 12시간만 훈련을 받은  후 번역 품질에서 sota를 달성

# Background
 - sequential 계산을 줄이는 목표는 또한 확장 GPU,ByteNet,ConvS2S의 기초를 형성하는데,이 모든 것은 모든 입력 및 출력 위치에 대해 병렬로 숨겨진 표현을 계산하여 convolution신경망을 기본 구성 블록으로 사용
 - 이러한 모델에서 두개의 임의의 입력 또는 출력 위치로부터의 신호를 관련시키는 데 필요한 작업수는 ConvS2S의 경우 선형이고 ByteNet의 경우 대수적으로 위치 사이의 거리에서 증가
 - 이것은 먼 위치 사이의 의존성을 배우는 것을 더 어렵게 만듬
 - transformer에서 이것은 비록 평균 attention 가중치 위치로 인해 효과적인 해결책의 비용이지만, 다중 헤드 어텐션으로 대응되는 효과로 인해 일정한 작업 수로 감소
 - self-attention는 시퀀스의 표현을 계산하기 위해 단일 시퀀스의 다른 위치와 관련된 attention 메카니즘
 - self-attention는 읽기 이해,추상적 요약,텍스트 수반 및 학습 과제 독립적 문장 표현을 포함한 다양한 작업에서 성공적으로 사용
 - end-to-end 메모리 네트워크는 recurrent attention mechanism을 기반으로 하며 간단한 언어 질문 응답 및 언어 모델링 작업에서 우수한 성능을 보임
 - Transformer는 시퀀스 recurrent RNN 또는 convolution 없이 입출력 표현을 계산하기 위해 전적으로 self-attention에 의존하는 첫번째 transduction model

# Model Architecture
 - 대부분 competitive neural sequence transduction models은 인코더-디코더 구조를 가지고 있음
 - 인코더는 기호 표현의 입력 시퀀스(x1,,,xn)를 연속 표현 순서(z1,,,zn)에 매핑
 - z가 주어지면 디코더는 한 번에 하나의 요소의 출력 시퀀스(y1,,yn)를 생성
 - 각 단계에서 모델은 auto-regressive되며,다음 단계를 생성할 때 이전에 생성된 기호를 추가 입력으로 사용
 - transformer는 인코더와 디코러 모두에 대해 쌓은 self-attention과 point-wise,fully connected layers를 사용하여 이 전체 아키텍처를 따름
![image](https://user-images.githubusercontent.com/70500214/110662167-98cf3e00-8208-11eb-96c5-4eabd9c396d5.png)

    # Encoder and Decoder Stacks
     - 인코더
        - 인코더는 N=6개의 동일한 레이어로 구성된 스택으로 구성, 각 계층에는 두개의 sub-layer가 존재
        - 첫번째는 multi-head self-attention mechaism이고,두번째는 단순한 position-wise fully connected feed-forward network
        - 두개의 sub-layer 주위에 residual connection을 사용하고 layer normalization를 사용
        - 즉, sub-layer의 출력은 LayerNorm(X+Sublayer(x))이며, 여기서 Sublayer(x)는 sub-layer 자체에 의해 구현되는 함수
        - 이런 residual connection을 용이하게 하기 위해 모델의 모든 sub-layer와 임베딩 layer는 model d=512차원의 출력값을 가짐
        
     - 디코더
        - 디코더 또한 N=6개의 동일한 레이어로 구성됨
        - 각 인코더 계층의 두 sub-layer 외에도 디코더는 multi-head attention을 수행할 sub-layer를 추가
        - 마찬가지로 sub-layer에 residual connection을 사용한 뒤, layer normalization을 수행
        - 디코더에서는 인코더와 달리 순차적으로 결과를 만들어내야 하기 때문에,self- attention을 변형(masking)
        - masking을 통해, position i보다 이후에 있는 position에 attention을 주지 못하게 함
        - 즉,position i에 대한 예측은 미리 알고 있는 output들에만 의존을 하는 것
        - 
       ![image](https://user-images.githubusercontent.com/70500214/110664780-0e3c0e00-820b-11eb-81a3-2c4ad1cbf7ec.png)
       
        - a를 예측할 때는 a 이후에 있는 b,c에는 attention이 주어지지않음
        
    # Attenion
     - attention은 단어의 의미처럼 특정 정보에 좀 더 주의를 기울이는 것
     - attention함수는 query+key-value->output으로의 변환을 수행
     - query,key,value,output은 모두 벡터
     - output은 value들의 가중합으로 계산되며, 그 가중치는 query와 연관된 key의 호환성 함수(compatibility funcition)에 의해 계산
    
    # Scaled Dot-Product Attention
    ![image](https://user-images.githubusercontent.com/70500214/110668969-388fca80-820f-11eb-94fe-d2424cf1b870.png)

    ![image](https://user-images.githubusercontent.com/70500214/110667700-de423a00-820d-11eb-9006-1be57f47d4ae.png)
     - input은 dk dimension의 query와 key들 dv dimension의 value들로 이루어짐
     - 이때 모든 query와 key에 대한 dot-product를 계산하고 각각을 ![image](https://user-images.githubusercontent.com/70500214/110667942-1f3a4e80-820e-11eb-9b13-e272743fbd94.png)로 나누어줌
     - dot-product를 하고 ![image](https://user-images.githubusercontent.com/70500214/110668035-35480f00-820e-11eb-9490-ba8da8b83e37.png)로 scaling을 해주기 때문에 Scaled Dot-Product Attention이라함
     - 그리고 여기에 softmax를 적용해 value들에 대한 weights를 얻음
     - key와 value는 attention이 이루어지는 위치에 상관없이 같은 값을 갖게 됨, 이대 query와 key에 대한 dot-product를 계산하면 각각의 query와 key 사이의 유사도를 구할 수 있음
     - ![image](https://user-images.githubusercontent.com/70500214/110668494-b7383800-820e-11eb-9450-837b3941bbce.png)로 scaling을 해주는 이유는 dot-products의 값이 커질수록 softmax 함수에서 기울기의 변화가 거의없는 부분으로 가기 때문 
     - softmax를 거친 값을 value에 곱해준다면,query와 유사한 value일수록,즉 중요한 value일수록 더 높은 값을가지게 됨
     - 중요한 정보에 관심을 둔다는 attention의 원리에 알맞은 것

    # Multi-Head Attention
    ![image](https://user-images.githubusercontent.com/70500214/110669004-404f6f00-820f-11eb-9e69-964d9e5603ff.png)
    
    ![image](https://user-images.githubusercontent.com/70500214/110669325-90c6cc80-820f-11eb-8fd1-5b1b2eff68af.png)
     - d<sub>model</sub> dimension의 key,value,query들로 하나의 attention을 수행하는 대신 key,value,query들에 각각 다른 학습된 linear projection을 h번 수행하는게 더 좋다고 생각
     - 즉 동일한 Q,K,V에 각각 다른 weight matrix W를 곱해주는 것, 이때 parameter matrix는 ![image](https://user-images.githubusercontent.com/70500214/110669732-f74bea80-820f-11eb-8cae-6e13ca513f11.png)
     - projection이라고 하는 이유는 각각의 값들이 parameter matrix와 곱해졌을때 d<sub>k</sub>,d<sub>v</sub>,d<sub>model</sub>차원으로 project되기 때문
     - 이렇게 project된 key, value, query들은 병렬적으로 attention function을 거쳐 d<sub>v</sub> dimension output 값으로 나오게 됨
     - 그 다음 여러개의 head를 concatenate하고 다시 projection을 수행,그래서 최종적인 d<sub>model</sub> dimension output값이 나오게 됨
    
    # Applications of Attention in out Model
     - transformer는 세가지 다른 방식으로 다중 헤드 어텐션을 사용
      - 인코더-디코더 attention layer에서 query는 이전 디코더 계층에서 나오고 key와 value값은 인코더의 출력에서 나옴, 이렇게 하면 디코더의 모든 위치가 입력 시퀀스의 모든 위치에 참석 가능,시퀀스 간 모델에서 일반적인 인코더-디코더 attention 메커니즘을 모방

      - 인코더에는 self-attention layer가 포함 됨,self-attention layer에서 모든 key,value,query는 동일한 위치에서 옴, 따라서 인코더의 각 위치는 이전 layer의 모든 위치를 고려할 수 있음, 만약 첫번째 layer라면 positional encoding이 더해진 input embedding이 됨

      - 디코더도 비슷, 그러나 auto-regressive 속성을 보존하기 위해 디코더는 출력을 생성할 시 다음 출력을 고려해서는 안됨, 즉 이전에 설명한 masking을 통해 이전 위치는 참조 불가능,이 masking은 dot-product를 수행할 때 -무한대로 설정함으로써 masking out시킴, 이렇게 설정되면 softmax를 통과할때 0이되므로 position에 attention을 주는 경우가 없어짐

    # Position-wise Feed-Forward Networks
     - attention sub-layer 외에도, 인코더와 디코더의 각 layer은 완전히 연결된 Feed-Forward 네트워크를 포함, 이 네트워크는 각 위치에 개별적으로 그리고 동일하게 적용(position-wise)
     - 네트워크는 두 번의 linear transformation과 activation function ReLU로 이루어짐
     
     ![image](https://user-images.githubusercontent.com/70500214/110675554-5c0a4380-8216-11eb-95df-7e0a1a418e6c.png)
     
     - 서로 다른 위치에서 linear transformation은 동일하지만 layer마다 매개변수는 다르게 사용
     - kernel size가 1이고 channel이 layer인 convolution을 두번 수행한 것으로도 위 과정을 이해 가능
     - 입출력 치수는  d<sup>model</sup>=512 그리고 내부 layer의 치수는 d<sub>ff</sub>=2048
     
    # Embedding and Softmax
     - 다른 시퀀스 전이 모델과 유사, 학습된 임베딩을 사용하여 입출력 토큰을 d<sub>model</sub>차원 모델의 벡터로 변환
     - 일반적인 학습된 linear transformation과 softmax 함수를 사용하여 디코더 출력을 예측 한 다음 토큰 확률로 변환
     - 이 모델에서는 2개의 임베딩 layer와 pre-softmax linear transformation 사이에 같은 weight 행렬을 사용

    # Positional Encoding
     - transformer는 recurrence와 convolution이 없기 때문에 모델이 시퀀스의 순서를 이용하기 위해서는 시퀀스에서 토큰의 상대적 또른 절대적 위치에 대한 정보를 주입해야함
     - 이를 위해, 인코더와 디코더 스택의 하단에 있는 입력 임베딩에 'positional encoding'을 추가
     - positional encoding은 임베딩과 동일한 치수 모델을 가지므로 두개를 합칠 수 있음
     - 이 작업에서 주파수가 다른 사인,코사인 함수 사용
     
     ![image](https://user-images.githubusercontent.com/70500214/110675566-5f053400-8216-11eb-9381-4e0a2aba70cc.png)
     
     - pos는 position,i는 dimension이고 주기가 10000<sup>2i/d<sub>model</sub></sup>*2π인 삼각함수
     - 즉,pos는 sequence에서 단어의 위치이고 해당 단어는 i에0부터 ![image](https://user-images.githubusercontent.com/70500214/110676249-1a2dcd00-8217-11eb-95d1-c566030070f5.png)까지를 대입해 d<sup>model</sup>차원의 positional encoding vector를 얻게 됨
     - k=2i+1일 때는 cosine 함수를,k=2i일 때는 sine함수를 이용
     - 이런 방식으로 positional encoding vencor를 pos마다 구한다면 비록 같은 column이라고 할지라도 pos가 다르다면 다른 값을 가지게 됨
     - 즉,pos마다 다른 pos와 구분되는 positional encoding값을 얻게 된다는 것
     - 가능한 여러 함수 중 sinusoidal version을 선택한 이유는 학습 때보다 더 긴 sequence를 만나도 추정이 가능하기 때문

# Why Self-Attention
 - x1,x2...xn를 동일한 길이의 z1,z2,...znn의 다른 시퀀스에 매핑하는데 일반적으로 사용되는  recurrent와 convolution layer와 비교
 - self-attention가 적합한 이유
    - layer 당 전체 계산량이 적음
    - 계산이 병렬화될 수 있음, 병렬적으로 한번에 많은 계산을 할 수 있는데,recurrence의 경우 순차적으로 계산해야하기 때문에 계산 병렬화 불가능
    - 장거리 학습의 가능 여부,attention을 통해 모든 부분을 확인하니 rnn에 보다 훨씬 먼거리에 있는 시퀀스를 잘학습할 수 있음

   ![image](https://user-images.githubusercontent.com/70500214/110726867-8a5f4180-825d-11eb-8cc9-8eaf07f1c51f.png)
   
  
# Training
 - 모델에 대한 교육 시스템 설명


   # Training Data and Batching
    - 약 450만 개의 문장 쌍으로 구성된 표준 WMT 2014 English-german 데이터 세트에 대해 교육
    - 문장은 약 37000개의 토큰의 공유 소스 대상 어휘를 가진 바이트 쌍 인코딩을 사용하여 인코딩
    - English-French의 경우 36M 문장으로 구성된 훨씬 큰 WMT 2014 영-프랑스 데이터 세트를 사용, 토큰을 32000개의 워드피스 어휘로 분할
    - 문장 쌍은 대략적인 시퀀스 길이만큼 함께 배치
    - 각 교육 배치에는 약 25000개의 소스 토큰과 25000개의 대상 토큰을 포함하는 문장 쌍 집합이 포함

   # Hardware and Schedule
    - 8개의 NVIDIA P100 GPU로 하나의 머신에서 교육
    - 기본 transformer모델은 12시간,big transformer는 3.5일

   # Optimizer
    - Adam β1 = 0.9, β2 = 0.98 and e = 10<sub>−9</sub>와 함께 사용
    - 아래의 공식을 따라 학습 속도를 변화
    ![image](https://user-images.githubusercontent.com/70500214/110743695-cef9d580-827b-11eb-9818-ed424bff33aa.png)
    - 이는 첫번째 warmup_step 훈련 단계에 대해 학습속도를 선형적으로 증가시키고 이후 단계 수의 역 제곱근에 비례하여 감소시키는 것과 일치
    - warmup_step=4000

   # Regularization
    - Residual Dropout
      - 드랍아웃을 sub-layer 입력에 추가하고 정규화하기 전에 각 sub-layer의 출력에 적용
      - 인코더와 디코더 스택 모두 임베딩과 positional encoding의 합계에 드랍아웃을 적용
      - P<sup>drop</sup>=0.1

    - Label Smoothing
      - training동안 e<sup>ls</sup>=0.1값의 label smoothing사용
      - 모델이 더 확실치 않다는 것을 학습하지만 정확도와 BLEU 점수가 향상되기 때문

![image](https://user-images.githubusercontent.com/70500214/110745737-103fb480-827f-11eb-8b5f-972b6d3fd59d.png)

# Results
   # Machine Translation
    - WMT 2014 English-German translation작업에서 big-Transformer모델은 이전 보고된 모델들보다 더 좋은 성능을 나타냄
    - 기본 transformer도 경쟁 모델의 교육 비용의 일부만으로 이전에 발표된 모든 모델과 앙상블을 능가
    - WMT 2014 English-French translation작업에서 교육 비용의 1/4 미만으로 이전에 발표된 모든 단일 모델을 능하가는 점수를 달성

![image](https://user-images.githubusercontent.com/70500214/110747007-0a4ad300-8281-11eb-8880-a742b69d3682.png)

![image](https://user-images.githubusercontent.com/70500214/110747033-120a7780-8281-11eb-89c5-4ad09aaf97f6.png)

   # Model Variations
    - Transformer의 다양한 구성 요소의 중요성을 평가하기 위해 English-to-German 번역에 대한 성능 변화를 측정하면서 다양한 방식으로 기본 모델을 변화시킴
    - Table3의 A에서 attention Head 수와  attetion key,attention value 값을 변화시킴
    - single-head attention의 BLEU값이 최고 설정치보다 0.9 나쁜 반면,너무 많은 헤드와 함께 품질도 떨어짐
    - B에서 attention key크기 d<sup>k</sup>를 줄이면 모델 품질이 저하되는것을 관찰
    - 호환성 판정이 쉽지않고 dot product보다 더 젏교한 호환성 기능이 유리함을 시사
    - C와D에서 예상대로 더 큰 모델이 더 좋고 dropout이 과적합을 피하는데 매우 도움된다고 관찰
    - E행에서 sinusoids대신 positional embedding으로 교체하고 기본 모델과 거의 동일한 결과를 관찰

   # English Constituency Parsing
    - Transformer가 다른 작업으로 
   
   
    





    



         
