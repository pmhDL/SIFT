""" run Distribution Calibration (DC) """
import os

def run_exp(shot=1, query=15, num_aug=1):
    way = 5
    gpu = 0
    dataname = 'mini'   # mini tiered cub cifar_fs
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
                  + ' --num_aug=' + str(num_aug) \
                  + ' --classifiermethod=' + 'nonparam' \
                  + ' --cls=' + 'lr'

    os.system(the_command + ' --phase=dc_te')

for num_a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]:
    run_exp(shot=1, query=15, num_aug=num_a)
    run_exp(shot=5, query=15, num_aug=num_a)
