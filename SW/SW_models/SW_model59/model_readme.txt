mother model : model58

modi :
centercrop(320, 400)으로 변경

이유 :
사진 좌우 공간도 의미가 없어서 데이터들을 보고 안짤리게 근접해서 삭제 대신 상하는 좀더 늘렸음

result:
Epoch[49/50](220/236) || training loss 0.1604 || training accuracy 95.23% || lr 0.00025
Calculating validation results...
New best model for val accuracy : 74.29%! saving the best model..
[Val] acc : 74.29%, loss: 0.18 || best acc : 74.29%, best loss: 0.18

마지막까지 accuracy가 갱신됐다
Epochs를 올리거나 Adam으로 변경해봐야할 것 같다