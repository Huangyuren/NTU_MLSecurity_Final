"""
parameter explanation:
    model: It is the log file's title file name.
    save_path: It indicates where the trained model's weight will be.
    resume_path: It indicates where the pretrained weights that we want to load stored.
    gpu: Specify your GPU-id in your current environment.
"""
# model = 'robnet_finetune_7_stage_2_40e+6e'
model = 'robnet_finetune_1_stage_2'
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
save_path = './my_experiments/robnet_finetune_1'
resume_path = './my_experiments/robnet_finetune_1/subnetwork_0_stage_1.pth'
# resume_path = './my_experiments/supernet.pth'

# Attack Params
# Important, if we want to do adversarial training or test pretrained model for white-box attack,
# the "attack" flag should be True, otherwise turn it into False.
attack_param = {'attack': True,
                'specific_choice': True,
                'epsilon': 8 / 255.,
                'num_steps': 7,
                'step_size': 0.01,
                'random_start': True}
