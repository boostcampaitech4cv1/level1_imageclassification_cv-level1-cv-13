mother model : 
efficient_b3

modi :
epoch 60으로 늘리고 resize 299, val batch 500으로 줄임

이유 :
resize 크기를 늘렸을 때 성능이 좋아져서 resize를 기준으로 늘릴 수 있는 모델을 찾았다
299크기와 b3 모델이 적당해서 선정하였고 val batch는 메모리 부족으로 절반으로 줄였다

result:
Epoch[59/60](220/236) || training loss 0.2732 || training accuracy 91.25% || lr 0.00025
Calculating validation results...
New best model for val accuracy : 84.68%! saving the best model..
[Val] acc : 84.68%, loss: 0.26 || best acc : 84.68%, best loss: 0.26

val acc는 높아지긴 했으나 inference data를 직접 보면 이전 best 모델 보다 더 많이 틀리는 것 같다
마스크 쓴 젊은 남자를 여자로 인식하거나 하는 문제가 있었다
혹시 60까지 올려서 overfitting이 된 건가 하는 느낌이 들어 epoch 50으로 줄이고
centercrop도 더 키워야할 것 같다