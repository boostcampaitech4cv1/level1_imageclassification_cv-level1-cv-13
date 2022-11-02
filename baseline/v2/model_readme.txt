
model91

results

Epoch[72/150](460/472) || training loss 0.08565 || training accuracy 96.95% || lr 0.000125
Calculating validation results...
[Val] acc : 76.53%, loss: 0.11 || best acc : 76.67%, best loss:  0.1

--------epoch 72 early stopping--------
--------epoch 72 early stopping--------

얼리스탑 10으로 설정됐을 때 72에폭에 서 끝났다


변경할만한 조건

Adam으로 변경하고 lr 0.0001로 줄이고 배치 16으로 내리고
모델 efficientNet_v2_m, resize 380으로 늘려서 학습

s 로 바꾸고 사이즈를 키웠는데도 안됐던 것은 SGD는 배치가 커야 학습이 잘되고
Adam은 배치가 작아야 학습이 잘되는데 SGD로 설정하고 배치를 줄여서 그런 것이 아닐까
그리고 Adam으로 했다면 lr이 작아야 잘 되는데 0.001로 lr이 커서 그랬던 것 아닐까 생각된다



effi_v2_s

Epoch[1/200](460/472) || training loss 0.03264 || training accuracy 98.83% || lr 0.0001
Calculating validation results...
New best model for val accuracy : 97.96%! saving the best model..
[Val] acc : 97.96%, loss: 0.037 || best acc : 97.96%, best loss: 0.037

2 epoch
결과 어린 남자를 자주 어린 여자라고 인식한다


model104 다같이 돌리기로 한 모델
effi_v2_s 0.0001, 32

Epoch[14/200](940/945) || training loss 0.02254 || training accuracy 99.22% || lr 0.0001
Calculating validation results...
[Val] acc : 98.33%, loss: 0.027 || best acc : 98.76%, best loss: 0.018

--------epoch 14 early stopping--------
--------epoch 14 early stopping--------


빨간 옷 어린 남자를 여자라고 인식함


model106 : model104에서 togray, val batch 500으로 증가

Epoch[14/200](940/945) || training loss 0.03613 || training accuracy 99.06% || lr 0.0001
Calculating validation results...
[Val] acc : 91.43%, loss: 0.035 || best acc : 92.09%, best loss: 0.021

--------epoch 14 early stopping--------
--------epoch 14 early stopping--------

4epoch에서 최저 로스를 찍고 그 후에 멈춤


model108 : model106에서 batch16으로 바꿈

Epoch[54/200](1880/1890) || training loss 0.0112 || training accuracy 100.00% || lr 2.5e-05
Calculating validation results...
[Val] acc : 92.22%, loss: 0.017 || best acc : 92.35%, best loss: 0.012

Finished f1 0.6860 acc 76.5238

다른 얘기
--------
val batch 250
Epoch[0/200](20/472) || training loss 1.846 || training accuracy 48.28% || lr 0.0001
Epoch[0/200](40/472) || training loss 0.9552 || training accuracy 73.52% || lr 0.0001
Epoch[0/200](60/472) || training loss 0.5682 || training accuracy 84.69% || lr 0.0001
Epoch[0/200](80/472) || training loss 0.4998 || training accuracy 84.53% || lr 0.0001
Epoch[0/200](100/472) || training loss 0.3809 || training accuracy 88.12% || lr 0.0001
Epoch[0/200](120/472) || training loss 0.3719 || training accuracy 88.52% || lr 0.0001
Epoch[0/200](140/472) || training loss 0.3845 || training accuracy 87.73% || lr 0.0001
Epoch[0/200](160/472) || training loss 0.3414 || training accuracy 88.12% || lr 0.0001
Epoch[0/200](180/472) || training loss 0.3187 || training accuracy 90.55% || lr 0.0001
Epoch[0/200](200/472) || training loss 0.2706 || training accuracy 90.70% || lr 0.0001
Epoch[0/200](220/472) || training loss 0.2613 || training accuracy 91.80% || lr 0.0001
Epoch[0/200](240/472) || training loss 0.2491 || training accuracy 91.64% || lr 0.0001
Epoch[0/200](260/472) || training loss 0.2438 || training accuracy 91.64% || lr 0.0001
Epoch[0/200](280/472) || training loss 0.2015 || training accuracy 94.06% || lr 0.0001
Epoch[0/200](300/472) || training loss 0.2027 || training accuracy 92.97% || lr 0.0001
Epoch[0/200](320/472) || training loss 0.2427 || training accuracy 92.66% || lr 0.0001
Epoch[0/200](340/472) || training loss 0.2223 || training accuracy 92.50% || lr 0.0001
Epoch[0/200](360/472) || training loss 0.2044 || training accuracy 93.59% || lr 0.0001
Epoch[0/200](380/472) || training loss 0.1975 || training accuracy 93.98% || lr 0.0001
Epoch[0/200](400/472) || training loss 0.165 || training accuracy 95.55% || lr 0.0001
Epoch[0/200](420/472) || training loss 0.1414 || training accuracy 95.62% || lr 0.0001
Epoch[0/200](440/472) || training loss 0.1859 || training accuracy 93.83% || lr 0.0001
Epoch[0/200](460/472) || training loss 0.1559 || training accuracy 95.23% || lr 0.0001
Calculating validation results...
New best model for val accuracy : 89.50%! saving the best model..
[Val] acc : 89.50%, loss: 0.11 || best acc : 89.50%, best loss: 0.11
10 Epoch left until early stopping..


val batch 1000
Epoch[0/200](20/472) || training loss 1.846 || training accuracy 48.28% || lr 0.0001
Epoch[0/200](40/472) || training loss 0.9552 || training accuracy 73.52% || lr 0.0001
Epoch[0/200](60/472) || training loss 0.5682 || training accuracy 84.69% || lr 0.0001
Epoch[0/200](80/472) || training loss 0.4998 || training accuracy 84.53% || lr 0.0001
Epoch[0/200](100/472) || training loss 0.3809 || training accuracy 88.12% || lr 0.0001
Epoch[0/200](120/472) || training loss 0.3719 || training accuracy 88.52% || lr 0.0001
Epoch[0/200](140/472) || training loss 0.3845 || training accuracy 87.73% || lr 0.0001
Epoch[0/200](160/472) || training loss 0.3414 || training accuracy 88.12% || lr 0.0001
Epoch[0/200](180/472) || training loss 0.3187 || training accuracy 90.55% || lr 0.0001
Epoch[0/200](200/472) || training loss 0.2706 || training accuracy 90.70% || lr 0.0001
Epoch[0/200](220/472) || training loss 0.2613 || training accuracy 91.80% || lr 0.0001
Epoch[0/200](240/472) || training loss 0.2491 || training accuracy 91.64% || lr 0.0001
Epoch[0/200](260/472) || training loss 0.2438 || training accuracy 91.64% || lr 0.0001
Epoch[0/200](280/472) || training loss 0.2015 || training accuracy 94.06% || lr 0.0001
Epoch[0/200](300/472) || training loss 0.2027 || training accuracy 92.97% || lr 0.0001
Epoch[0/200](320/472) || training loss 0.2427 || training accuracy 92.66% || lr 0.0001
Epoch[0/200](340/472) || training loss 0.2223 || training accuracy 92.50% || lr 0.0001
Epoch[0/200](360/472) || training loss 0.2044 || training accuracy 93.59% || lr 0.0001
Epoch[0/200](380/472) || training loss 0.1975 || training accuracy 93.98% || lr 0.0001
Epoch[0/200](400/472) || training loss 0.165 || training accuracy 95.55% || lr 0.0001
Epoch[0/200](420/472) || training loss 0.1414 || training accuracy 95.62% || lr 0.0001
Epoch[0/200](440/472) || training loss 0.1859 || training accuracy 93.83% || lr 0.0001
Epoch[0/200](460/472) || training loss 0.1559 || training accuracy 95.23% || lr 0.0001
Calculating validation results...
New best model for val accuracy : 76.77%! saving the best model..
[Val] acc : 76.77%, loss: 0.11 || best acc : 76.77%, best loss: 0.11
10 Epoch left until early stopping..

val batch size에 따라 val acc는 달라지지만 val loss는 같다
그렇다면 model을 평가할 때 val acc보다 val loss를 기준으로 평가하는 것이 더 합리적인 것 아닌가란 생각

model111 v2_l 모델에 0.0001, 64 적용

Epoch[15/200](460/472) || training loss 0.03211 || training accuracy 99.30% || lr 0.0001
Calculating validation results...
[Val] acc : 78.68%, loss: 0.032 || best acc : 78.89%, best loss: 0.026

Finished	f1 : 0.7095	acc : 78.0476


model112 model111에서 dataset을 profiledataset으로 변경

Epoch[13/200](480/483) || training loss 0.01562 || training accuracy 99.61% || lr 0.0001
Calculating validation results...
[Val] acc : 78.02%, loss: 0.77 || best acc : 78.66%, best loss:  0.5

3 epoch에서 최저로스는 찍고 끝났다 뭔가 로컬 미니멈에 빠진 것 같다 lr를 키워서 다시 해봐야겠다


model113

lr 0.0002 으로 돌리던건데 돌리다가 취소함

Epoch[9/200](480/483) || training loss 0.05485 || training accuracy 98.67% || lr 0.0002
Calculating validation results...
New best model for val accuracy : 79.83%! saving the best model..


model114

Epoch[28/200](460/472) || training loss 0.02624 || training accuracy 99.45% || lr 0.0001
Calculating validation results...
[Val] acc : 98.62%, loss: 0.025 || best acc : 98.89%, best loss: 0.013
0 Epoch left until early stopping..

lr decay를 없앴더니 오히려 더 적은 epoch에서 끝났다
Finished	0.6779	75.1429
약간 더 안좋은 성능을 보이는 것 같다





