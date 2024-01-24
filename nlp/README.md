- LLM_INFL_ID_1.py: Run first (make sure DATASET variable is set to relevant dataset), it will load dataset, finetune models, and also save gradient space for models. Plus AUC and Recall for baselines
- LLM_INFL_ID_2.py: Again check the DATASET variable, and this file will load up the gradient space and modify it to run for outlier gradient trimming. Will generate AUC and ROC after.
- LLM_INFL_ID_3.py: For plotting and generating any results

- ROBERTA_SELECTION_1.py: Run Roberta experiment to generate all data and predictions
- ROBERTA_SELECTION_2.py: Plot the results obtained
