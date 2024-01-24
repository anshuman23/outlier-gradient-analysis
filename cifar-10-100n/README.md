- Helper Code: datainf_influence_functions.py

- CIFAR-10N command: //noise_type can be worst, rand1, aggre
CUDA_VISIBLE_DEVICES=0 python OD-experiment.py --dataset cifar10 --noise_type worst --is_human --seed 0



- CIFAR-100N command:
CUDA_VISIBLE_DEVICES=0 python OD-experiment-CIFAR100N.py --dataset cifar100 --noise_type noisy100 --is_human --seed 0



- ANIMAL-10N command:
python OD-experiment.py --train-dir ../data/Animal10N/train/ --val-dir ../data/Animal10N/test/ --dataset Animal10N --arch vgg19-bn --lr-gamma 0.2 --batchsize 128 --warmUpIter 0 --out-dir ./saved_models/OD0 --gpu 0



- BASELINES label correction - command: //baseline_method can be normalized_margin, self_confidence, confidence_weighted_entropy || noise_type can take worst, rand1, aggre
-> CIFAR10:
CUDA_VISIBLE_DEVICES=0 python baseline-experiment-label.py --dataset cifar10 --noise_type rand1 --is_human --baseline_method confidence_weighted_entropy --seed 0

-> CIFAR100:
CUDA_VISIBLE_DEVICES=0 python baseline-experiment-label.py --dataset cifar100 --noise_type noisy100 --is_human --baseline_method confidence_weighted_entropy --seed 0



- BASELINES influence based - command:  noise_type can take worst, rand1, aggre
-> CIFAR10:
python baseline-experiment-influence.py --dataset cifar10 --noise_type aggre --is_human --seed 0

-> CIFAR100:
python baseline-experiment-influence.py --dataset cifar100 --noise_type noisy100 --is_human --seed 0



- Visualizing OD Images Helper Code:
-> CIFAR10:
python visualize-OD.py --dataset cifar10 --noise_type worst --is_human --seed 0

-> CIFAR100:
python visualize-OD-CIFAR100N.py --dataset cifar100 --noise_type noisy100 --is_human --seed 0
