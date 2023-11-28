# VAEplusplus-Variational-AutoEncoder-for-Heterogeneous-One-Class-Collaborative-Filtering-torch-

VAE++: Variational AutoEncoder for Heterogeneous One-Class Collaborative Filtering
Wanqi Ma1, 2, 3, Xiancong Chen1, 2, 3, Weike Pan1, 2, 3∗, Zhong Ming1, 2, 3

torch version

test with parameters below:

python train.py --dataset Rec15 --transaction target_train --examination auxiliary --test target_test --user_num 36917 --item_num 9621 --batch_size 500 --hiddenDim 100 --reg_scale 0.0 --optimizer Adam --lr_rate 0.001

result:

NDCG@5 = 0.1850 
Precision@5: 0.0555 
Recall@5: 0.2776 
F1@5: 0.0925 
NDCG@5: 0.1850 
1-call@5: 0.2776 



On Rec15:

CUDA_VISIBLE_DEVICES=2 python train.py --dataset Rec15 --transaction target_train --examination auxiliary --test target_test --user_num 36917 --item_num 9621 --batch_size 500 --hiddenDim 100 --reg_scale 0.0 --optimizer Adam --lr_rate 0.0001


================================================================================================================================================================================================================================================================================================

On ML10M:

CUDA_VISIBLE_DEVICES=0 python train.py --dataset ML10M --transaction ML10M.HOCCF.copy1.target --examination ML10M.HOCCF.copy1.auxiliary --test ML10M.HOCCF.copy1.test --user_num 71567 --item_num 10681 --batch_size 500 --hiddenDim 100 --reg_scale 0.0 --optimizer Adam --lr_rate 0.0001


================================================================================================================================================================================================================================================================================================

On Netflix:

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Netflix --transaction Netflix.HOCCF.copy1.target --examination Netflix.HOCCF.copy1.auxiliary --test Netflix.HOCCF.copy1.test --user_num 480189 --item_num 17770 --batch_size 500 --hiddenDim 100 --reg_scale 0.0 --optimizer Adam --lr_rate 0.0001

