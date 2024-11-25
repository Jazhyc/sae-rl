# Interpretable Reinforcement Learning using SAE discovered Feature Steering

Done for the Reprogramming AI hackathon by Apart Research. Currently the code is very messy and I'm not certain when I will clean it up.

There are two main ways to run the training code after getting the requirements:

1. Run the `main.py` file after modifying the if __name__ == "__main__" block to run the desired experiment.

2. Use the `main.ipynb` notebook to run the code in a more interactive way which demonstrates how the code works and various experiment configurations.

Aside from the training code, there is a also a visualize.py script which can be used to visualize the steering of features for a given state. If you clone the repo, you can run the file directly. Otherwise, you will need to download the `features.pkl` file. The visualization UI is quite intuitive to use and you can cycle through labels in a cell by continuously clicking. Not all states will have feature steering vectors, so you can use the next board state button to cycle through states.