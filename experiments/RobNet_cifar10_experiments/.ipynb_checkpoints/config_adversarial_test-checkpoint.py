"""
parameter explanation:
    model: It is the log file's title file name.
    save_path: It indicates where the trained model's weight will be.
    resume_path: It indicates where the pretrained weights that we want to load stored.
    gpu: Specify your GPU id in your environment
"""
model = "finetune_1_test_benign"
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
resume_path = './my_experiments/robnet_finetune_1/subnetwork_0_stage_2.pth'

# Attack Params
attack_param = {'attack': False,
                'specific_choice': True,
                'epsilon': 8 / 255.,
                'num_steps': 7,
                'step_size': 0.01,
                'random_start': True}
