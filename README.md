# Outlier Gradient Analysis

### Prerequisites and Requirements
The following packages need to be installed (using pip):
- ```argh```
- ```datasets```
- ```evaluate```
- ```numpy```
- ```pandas```
- ```pickle```
- ```matplotlib```
- ```torch```
- ```tqdm```
- ```transformers```
- ```sklearn```
- ```pyod```
- ```cleanlab```
- ```seaborn```
- ```peft```

Also note that for the LLM experiments, you will need to have access to the Llama-2-13B LLM model from Meta and then utilize it using Huggingface. Please see [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) for more details.

___


### Synthetic Data Experiments
First ```cd``` into the ```synthetic``` directory. Run the following code files for each set of experiments as described:
- ```PLOT_TOY_LR.py```: This will plot all the figures for the LR model on the Linear synthetic dataset.
- ```PLOT_TOY_MLP.py```: This will plot all the figures for the MLP model on the Non-Linear Half Moons synthetic dataset.
- ```BASELINES_LABEL_TOY_MLP.py```: This will compare noisy label correction baselines as well as outlier gradient trimming for the MLP setting.
- ```BASELINES_INFLUENCE_TOY_MLP.py```: This will compare influence-based baselines for the MLP setting.

___

### Noisy Vision Data Experiments
First ```cd``` into the ```cifar-10-100n``` directory. Run the following code files for each set of experiments as described:
- For ```CIFAR-10N``` experiments, run the following command (note ```noise_type``` can take the values ```{worst, aggre, rand1}```):
   - ```python OD-experiment.py --dataset cifar10 --noise_type worst --is_human --seed 0```
- For ```CIFAR-100N``` experiments, run the following command:
   - ```python OD-experiment-CIFAR100N.py --dataset cifar100 --noise_type noisy100 --is_human --seed 0```
- For label correction baseline experiments, run the following command (note ```baseline_method``` can take the values ```{normalized_margin, self_confidence, confidence_weighted_entropy}``` and ```noise_type``` can take the values ```{worst, aggre, rand1}```):
   - For ```CIFAR-10N``` (note ```noise_type``` can take the values ```{worst, aggre, rand1}```) :
       - ```python baseline-experiment-label.py --dataset cifar10 --noise_type rand1 --is_human --baseline_method confidence_weighted_entropy --seed 0```
   - For ```CIFAR-100N```:
       - ```python baseline-experiment-label.py --dataset cifar100 --noise_type noisy100 --is_human --baseline_method confidence_weighted_entropy --seed 0```
- For influence-based baseline experiments (all baselines will run):
   - For ```CIFAR-10N``` (note ```noise_type``` can take the values ```{worst, aggre, rand1}```):
       - ```python baseline-experiment-influence.py --dataset cifar10 --noise_type aggre --is_human --seed 0```
   - For ```CIFAR-100N```:
       - ```python baseline-experiment-influence.py --dataset cifar100 --noise_type noisy100 --is_human --seed 0```
- If you would like to visualize samples detected by the outlier gradient analysis method and also compute running times, use:
    - For ```CIFAR-10N``` (note ```noise_type``` can take the values ```{worst, aggre, rand1}```):
       - ```python visualize-OD.py --dataset cifar10 --noise_type worst --is_human --seed 0```
    - For ```CIFAR-100N```:
       - ```visualize-OD-CIFAR100N.py --dataset cifar100 --noise_type noisy100 --is_human --seed 0```

___

### RoBERTa Data Selection Experiments
First ```cd``` into the ```nlp``` directory. Run the following code files in the order described:
  - Run ```python ROBERTA_SELECTION_1_LATEST.py```: This will run the RoBERTa model and generate all results and predictions. Make sure to set the ```task``` variable in the code to the dataset desired (```{mrpc, qnli, qqp, sst2}```)
  - Next, run ```python ROBERTA_SELECTION_2_LATEST.py```: This will make the relevant plots and store them in ```saved_figs_roberta```

___

### LLM Influential Data Identificaiton Experiments
First ```cd``` into the ```nlp``` directory. Run the following code files in the order described:
  - Run ```python LLM_INFL_ID_1.py.py```: Ensure that the ```DATASET``` variable is set to the desired dataset: ```{math_with_reason, math_without_reason, grammars}``` and that the ```base_path``` and ```project_path``` are set as required (this follows as in the DataInf code). Running this file will then load the dataset, finetune LLMs, and also save gradient space for models in ```saved_grads```. It will also compute AUC and Recall for the influence-based baselines.
  - Next, run ```python LLM_INFL_ID_2.py.py```: Ensure that the ```DATASET``` variable is set to the desired dataset. This file will load up the gradient space and modify it to run for outlier gradient trimming. It will generate AUC and ROC for the outlier gradient analysis/trimming method.
  - Next, run ```python LLM_INFL_ID_2.py.py```: This code file will generate any remaining results for plotting and store them in ```saved_figs_llm```.

___
