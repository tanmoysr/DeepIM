# DeepIM
Selecting a set of initial users from a social network in order to maximize the envisaged number of influenced users is known as influence maximization (IM). Researchers have achieved significant advancements in the theoretical design and performance gain of several classical approaches, but these advances are almost reaching their pinnacle. Learning-based IM approaches have emerged recently with a higher generalization to unknown graphs than conventional methods. 
The development of learning-based IM methods is still constrained by a number of fundamental hardships, including 1) solving the objective function efficiently, 2) struggling to characterize the diverse underlying diffusion patterns, and 3) adapting the solution to different node-centrality-constrained IM variants.
To address the aforementioned issues, we design a novel framework DeepIM for generatively characterizing the latent representation of seed sets, as well as learning the diversified information diffusion pattern in a data-driven and end-to-end way. Subsequently, we design a novel objective function to infer optimal seed sets under flexible node-centrality-based budget constraints. Extensive analyses are conducted over both synthetic and real-world datasets to demonstrate the overall performance of DeepIM. 

## Instructions:
1. Main Model:
Use [run_model.py](/code/run_model.py) to run the model and [configuration.py](/code/configuration.py) to configure the model.

### Data: 
We compare the performance of DeepIM with other existing approaches using six real-world datasets: Cora-ML, Network Science, Power Grid, Jazz, Digg, and Weibo. Additionally, we employ a synthetic dataset generated using the Erdos-Renyi algorithm with 50,000 nodes.

## Citation
If you use this work, please cite the following article.

Chowdhury, Tanmoy, Chen Ling, Junji Jiang, Junxiang Wang, My T. Thai, and Liang Zhao. "Deep graph representation learning influence maximization with accelerated inference." Neural Networks (2024): 106649.
