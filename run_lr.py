import os

def run_exp(shot=1, query=15, lr=0.1):
    way = 5
    gpu = 1
    dataname = 'mini' # mini, tiered, cifar_fs, cub
    modelname = 'wrn28' # wrn28, res12

    the_command = 'python3 main.py' \
                  + ' --shot=' + str(shot) \
                  + ' --train_query=' + str(query) \
                  + ' --val_query=' + str(query) \
                  + ' --way=' + str(way) \
                  + ' --model_type=' + modelname \
                  + ' --dataset=' + dataname \
                  + ' --dataset_dir=' + '/data/'+dataname+'/'+modelname \
                  + ' --gpu=' + str(gpu) \
                  + ' --gradlr=' + str(lr) \
                  + ' --setting=' + 'in' \
                  + ' --classifiermethod=' + 'nonparam' \
                  + ' --cls=' + 'lr'

    os.system(the_command + ' --phase=lr_te')

# setting: in, tran

run_exp(shot=1, query=15, lr=0.2)
run_exp(shot=5, query=15, lr=0.2)