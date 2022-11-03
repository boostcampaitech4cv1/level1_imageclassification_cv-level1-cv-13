
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

결과 어린 남자를 자주 어린 여자라고 인식한다

