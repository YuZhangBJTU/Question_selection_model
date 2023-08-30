# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import sys
import torch
from Models.SDNetTrainer import SDNetTrainer
from Utils.Arguments import Arguments

opt = None

parser = argparse.ArgumentParser(description='SDNet')
parser.add_argument('command', help='Command: train', default='train')
parser.add_argument('conf_file', help='Path to conf file.', default='coqa/conf')

cmdline_args = parser.parse_args()
command = cmdline_args.command
conf_file = cmdline_args.conf_file

# command = 'train'
# conf_file = 'coqa/conf'

conf_args = Arguments(conf_file)
opt = conf_args.readArguments()
opt['cuda'] = torch.cuda.is_available()
opt['confFile'] = conf_file
opt['datadir'] = os.path.dirname(conf_file)  # conf_file specifies where the data folder is
opt['RESUME'] = True
opt['MODEL_PATH'] = 'conf~/run_23/best_model.pt'#15, 16, 32

for key,val in cmdline_args.__dict__.items():
    if val is not None and key not in ['command', 'conf_file']:
        opt[key] = val

model = SDNetTrainer(opt)    
    
print('Select command: ' + command)
model.train()

#run: python main.py train coqa/conf