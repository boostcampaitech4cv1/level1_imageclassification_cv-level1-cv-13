mother model : model59
efficient_v2_l

modi :
epoch 60으로 늘리고
centercrop도 (440, 320)으로 변경


이유 :
model59에서 마지막까지 리더보드 갱신해서 60까지 늘려봤다
그리고 centercrop은 순서를 헷갈림
가로, 세로가 아니고 h,w 여서 반대로 생각하고 있었다...

result:
Epoch[59/60](220/236) || training loss 0.1567 || training accuracy 94.84% || lr 0.00025
Calculating validation results...
[Val] acc : 74.74%, loss: 0.16 || best acc : 74.74%, best loss: 0.16

0.6852	75.8095 에서
0.6854	76.1270 리더보드 갱신

거의 차이는 없긴 한데 이게 좀더 사람이 이해할 수 있는 결과인 것 같다
이전에는 옆에 검은 패딩 데이터가 있고 위아래 많이 짤렸다
하지만 현재 이미지는 비율이 안맞는데 그것을 resize할 때 1:1 로 맞추기 때문에 위아래로 짜부되는 느낌이 있긴 하다
