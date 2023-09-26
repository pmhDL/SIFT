""" run SIFT """
import os

def run_exp(shot=1, query=15, num_aug=100):
    way = 5
    gpu = 1
    dataname = 'mini'   # mini tiered cifar_fs cub
    modelname = 'wrn28' # wrn28 res12

    the_command = 'python3 main.py' \
                  + ' --shot=' + str(shot) \
                  + ' --train_query=' + str(query) \
                  + ' --val_query=' + str(query) \
                  + ' --way=' + str(way) \
                  + ' --model_type=' + modelname \
                  + ' --dataset=' + dataname \
                  + ' --dataset_dir=' + '/data/'+dataname+'/'+modelname \
                  + ' --gpu=' + str(gpu) \
                  + ' --gradlr=' + str(0.1) \
                  + ' --num_aug=' + str(num_aug) \
                  + ' --classifiermethod=' + 'nonparam' \
                  + ' --cls=' + 'lr' \
                  + ' --selectm=' + 'randomselect' \
                  + ' --setting=' + 'in'

    os.system(the_command + ' --phase=st_te')

# setting: in tran
# cls: knn, lr, svm

print('inductive')
#print('transductive')
for num_a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]:
    run_exp(shot=1, query=15, num_aug=num_a)
    run_exp(shot=5, query=15, num_aug=num_a)
