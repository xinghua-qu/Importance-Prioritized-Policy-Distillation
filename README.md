# Importance Prioritized Policy Distillation (IPPD)

### Train the teacher policy:
Download the code for RAINBOW from github 
Inside the folder Rainbow-master, run: 

    python main.py —id your_runname —seed 123 —game game_name —T-max 4000000 # game_name example: bank_heist, road_runner, pong, freeway.

The ohyperparameters for training teacher policy can be found in Appendix in the paper.

Then run code to collect the dataset:
	python data_collection.py
### Policy distillation training:
	## IP-KL
	python PD_adaptive_importance_KL.py —game bank_heist

	## IP-CE
	python PD_adaptive_importance_CrossEntropy.py —game bank_heist

	## IP base policy conpression
	python PD_adaptive_importance_compression.py —game bank_heist

	## The detail of parameter setting for policy distillation is provided in Appendix D in our paper

### Policy evaluation:
	python evaluation.py --game bank_heist
