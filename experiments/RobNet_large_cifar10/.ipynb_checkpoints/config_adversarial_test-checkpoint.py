model = "proxylessNAS"
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
save_path = './my_experiments'
use_origin_advTrain=False
resume_path = './your_NAS_pth_file'

# Attack Params
attack_param = {'attack': True,
                'specific_choice': False,
                'epsilon': 8 / 255.,
                'num_steps': 7,
                'step_size': 0.01,
                'random_start': True}
