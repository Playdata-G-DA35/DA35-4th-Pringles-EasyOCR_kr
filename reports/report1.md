# 모델링 기법 선정

## 1. **프로젝트 주제**
    - 주제: 손글씨 인식 모델 성능 개선 - EasyOCR Fine-tuning
    - 목적: EasyOCR fine-tuning을 통해 손글씨 인식 성능을 개선
    
## 2. **데이터 설명**
### 1-1. 한글 학습 데이터 생성
| No | Arguments | Description | Custom |
| --- | --- | --- | --- |
| 1 | -i, --input_file | 기본으로 제공되는 학습 단어 모음(dictionaries)이 아닌, 직접 구축한 학습 단어 모음을 사용하고 싶을 때 사용 | 기본 한글 dic 수정 → 숫자+영문+한국어 5,888 기초 낱말(국립국어연구원 발표)+외래어 사전 조합 8,347개 |
| 2 | --output_dir | 생성데이터를 저장하는 위치를 지정할 수 있으며, 기본값은 '/out' 디렉토리 |  |
| 3 | -c, --count | 생성할 학습 데이터의 개수로, 기본값은 1,000 | train 100000 / validation 1000 |
| 4 | -l, --language | 학습데이터의 언어를 변경하고자 할 때 사용 | -l ko |
| 5 | -t, --thread_count | 학습데이터 생성 시 사용할 CPU 코어의 개수 | None |
| 6 | -f, --format | 생성되는 이미지 사이즈로, 기본값은 32 pixel | -f 80 -or 0 Height pixel을 80으로 변경 |
| 7 | -ft, --font | 생성할 학습데이터에 사용할 특정 폰트파일 지정 시 사용 | default 글씨체를 삭제하고 [네이버 나눔 손글씨 글꼴](https://clova.ai/handwriting/list.html) 50개 추가 |
| 8 | -b,  --background | 사용할 배경의 종류를 정의 0→ Gaussian Noise; 1→ Plain white; 2→ Quasicrystal; 3→ Pictures | -b 1 Plain White로 설정 |
| 9 | -m, --margins | 렌더링할 때 텍스트 주위의 여백을 정의 | -m 0,0,0,0  여백을 없애서 붙어있는 단어의 경우 더 잘 인식될 수 있도록 설정 |
| 10 | -k, --skew_angle | 생성된 텍스트의 기울기 각도를 정의 | -k 15 -rk  -15에서 15도 사이에서 삐뚤어지게 생성하여 손글씨체 형태 반영  |

![image](image/data.png)

### 1-2. 학습 데이터 변환
이미지 파일 목록과 각 이미지 파일의 label이 저장된 gt.txt 파일로 변환

### 1-3. 학습데이터를 lmdb 포맷으로 변환
실제 학습에서 사용할 lmdb 포맷으로 학습데이터를 변환

## 3. **모델링 기법**
   - **학습데이터셋**: Train100,000 / Validation 1,000
      - 이유? - 학습의 목표를 validation 데이터에 대한 정확도를 올리는게 아니라 test 데이터를 얼마나 잘 맞추느냐에 포인트를 두었음. 따라서 training 데이터를 최대한으로 잡아 학습할 양을 늘리는데 초점을 둠.
- 미세조정(Fine-tune) 학습을 위한 **Pre-trained 모델**: [EasyOCR](https://github.com/JaidedAI/EasyOCR) 프로젝트의 '**korean_g2.pth**'
- **Batch size**: 192
- **lr**: 1
- **최적화 함수 (Optimizer)**
    - default: Adadelta
    - **custom: Adam**
- **Loss 함수**
    - Prediction
        - CTC : **CTCLoss**
        - Attn : **CrossEntropyLoss**
- **Data processing**
    
    
    | No | Arguments | Default | Custom |
    | --- | --- | --- | --- |
    | 1 | --select_data | MJ-ST | / |
    | 2 | --batch_ratio | 0.5-0.5 | 1 |
    | 3 | --imgH | 32 | 80 |
    | 4 | --imgW | 100 | 100 |
    | 5 | --character | 0123456789abcdefghijklmnopqrstuvwxyz | 'korean_g2' > 'characters' 사용 |
    
        
- **Model Architecture**
    - 학습모델의 모듈 조합: **'None-VGG-BiLSTM-CTC'**
    
    | No | Arguments | Default | Custom |
    | --- | --- | --- | --- |
    | 1 | --Transformation | None|TPS | None |
    | 2 | --FeatureExtraction | VGG|RCNN|ResNet | VGG |
    | 3 | --SequenceModeling | None|BiLSTM | BiLSTM |
    | 4 | --Prediction | CTC|Attn | CTC |
    | 5 | --input_channel | 1 | 1 |
    | 6 | --output_channel | 512 | 256 |
    | 7 | --hidden_size | 256 | 256 |

### **3-3. 모델 학습**

[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) 프로젝트 root에서 학습

- ***code***
    
    ```python
    !python3 train.py \
    --data_filtering_off \
    --train_data "/content/drive/MyDrive/easy_ocr_kr_NVBC/workspace/data_lmdb/train" \
    --valid_data "/content/drive/MyDrive/easy_ocr_kr_NVBC/workspace/data_lmdb/validation" \
    --select_data / \
    --batch_ratio 1 \
    --Transformation "None" \
    --FeatureExtraction "VGG" \
    --SequenceModeling "BiLSTM" \
    --Prediction "CTC" \
    --saved_model "/content/drive/MyDrive/easy_ocr_kr_NVBC/workspace/pre_trained_model/korean_g2.pth" \
    --input_channel 1 \
    --output_channel 256 \
    --hidden_size 256 \
    --imgH 80 \
    --imgW 100 \
    --FT 
    ```
    
- ***option***
    
    ```
    ------------ Options -------------
    exp_name: None-VGG-BiLSTM-CTC-Seed1111
    train_data: /content/drive/MyDrive/easy_ocr_kr/workspace/data_lmdb/train
    valid_data: /content/drive/MyDrive/easy_ocr_kr/workspace/data_lmdb/validation
    manualSeed: 1111
    workers: 4
    batch_size: 192
    num_iter: 300000
    valInterval: 2000
    saved_model: /content/drive/MyDrive/easy_ocr_kr/workspace/pre_trained_model/korean_g2.pth
    FT: True
    adam: False
    lr: 1
    beta1: 0.9
    rho: 0.95
    eps: 1e-08
    grad_clip: 5
    baiduCTC: False
    select_data: ['/']
    batch_ratio: ['1']
    total_data_usage_ratio: 1.0
    batch_max_length: 25
    imgH: 80
    imgW: 100
    rgb: False
    character:  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘
    sensitive: False
    PAD: False
    data_filtering_off: True
    Transformation: None
    FeatureExtraction: VGG
    SequenceModeling: BiLSTM
    Prediction: CTC
    num_fiducial: 20
    input_channel: 1
    output_channel: 256
    hidden_size: 256
    num_gpu: 1
    num_class: 1009
    ---------------------------------------
    ```
    
- ***dataset log***
    
    ```
    --------------------------------------------------------------------------------
    dataset_root: /content/drive/MyDrive/easy_ocr_kr/workspace/data_lmdb/train
    opt.select_data: ['/']
    opt.batch_ratio: ['1']
    --------------------------------------------------------------------------------
    dataset_root:    /content/drive/MyDrive/easy_ocr_kr/workspace/data_lmdb/train	 dataset: /
    sub-directory:	/.	 num samples: 100000
    num total samples of /: 100000 x 1.0 (total_data_usage_ratio) = 100000
    num samples of / per batch: 192 x 1.0 (batch_ratio) = 192
    --------------------------------------------------------------------------------
    Total_batch_size: 192 = 192
    --------------------------------------------------------------------------------
    dataset_root:    /content/drive/MyDrive/easy_ocr_kr/workspace/data_lmdb/validation	 dataset: /
    sub-directory:	/.	 num samples: 1000
    --------------------------------------------------------------------------------
    ```
    
- ***train log***
    
    ```
    [1/300000] Train loss: 2.50540, Valid loss: 1.29134, Elapsed_time: 68.71375
    Current_accuracy : 68.800, Current_norm_ED  : 0.84
    Best_accuracy    : 68.800, Best_norm_ED     : 0.84
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    한번                        | 한번                        | 0.9966	True
    재산                        | 재산                        | 0.9996	True
    불러일으키다                    | 불임으다                      | 0.3046	False
    논하다                       | 논하다                       | 0.9999	True
    키                         | 구                         | 0.8717	False
    --------------------------------------------------------------------------------
    [2000/300000] Train loss: 0.06096, Valid loss: 0.02102, Elapsed_time: 404.18034
    Current_accuracy : 98.200, Current_norm_ED  : 0.99
    Best_accuracy    : 98.200, Best_norm_ED     : 0.99
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    뛰어나오다                     | 뛰어나오다                     | 0.8866	True
    큰딸                        | 큰딸                        | 0.9513	True
    재산                        | 재산                        | 0.9996	True
    리트머스                      | 리트머스                      | 0.7872	True
    신디이트                      | 신디이트                      | 0.9853	True
    --------------------------------------------------------------------------------
    [4000/300000] Train loss: 0.00165, Valid loss: 0.01216, Elapsed_time: 708.81233
    Current_accuracy : 99.100, Current_norm_ED  : 0.99
    Best_accuracy    : 99.100, Best_norm_ED     : 0.99
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    등장하다                      | 등장하다                      | 0.6699	True
    3                         | 3                         | 0.9937	True
    단어                        | 단어                        | 0.9970	True
    나빠지다                      | 나빠지다                      | 0.9986	True
    팩시밀리                      | 팩시밀리                      | 0.9970	True
    --------------------------------------------------------------------------------
    [6000/300000] Train loss: 0.00042, Valid loss: 0.01090, Elapsed_time: 1013.74369
    Current_accuracy : 98.900, Current_norm_ED  : 0.99
    Best_accuracy    : 99.100, Best_norm_ED     : 0.99
    --------------------------------------------------------------------------------
    Ground Truth              | Prediction                | Confidence Score & T/F
    --------------------------------------------------------------------------------
    그리로                       | 그리로                       | 0.8917	True
    예비                        | 예비                        | 1.0000	True
    리보솜                       | 리보솜                       | 0.9967	True
    불안하다                      | 불안하다                      | 0.9940	True
    말하다                       | 말하다                       | 0.9957	True
    --------------------------------------------------------------------------------
    ```