# model = 'robnet_large_v1'
model_param = dict(C=36,
                   num_classes=10,
                   layers=20,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=True,
                   AdPoolSize=1)
dataset = 'cifar10'
dataset_param = dict(data_root='./data/cifar10',
                     batch_size=32,
                     num_workers=2)
report_freq = 10
seed = 10
gpu = 0
save_path = './log'
use_origin_advTrain=False
# resume_path = './checkpoint/RobNet_large_v1_cifar10.pth.tar'

# Attack Params
attack_param = {'attack': True,
                'specific_choice': True,
                'epsilon': 8 / 255.,
                'num_steps': 7,
                'step_size': 0.01,
                'random_start': True}
