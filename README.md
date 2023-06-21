# GeneticAlgoritem Cryptanalysis
This code implements a genetic algorithm for cryptanalysis, specifically for breaking a substitution cipher. The genetic algorithm aims to find the correct permutation of the English alphabet that maps the cipher text to its corresponding plaintext. The algorithm evolves a population of candidate solutions over multiple generations, using selection, crossover, and mutation operators.


## Requirements
Files
- enc.txt: The input cipher text that needs to be decrypted.
- dict.txt: A text file containing a list of English words. This is used to evaluate the fitness of the decrypted text.
- Letter_Freq.txt: A text file containing the frequencies of individual English letters.
- Letter2_Freq.txt: A text file containing the frequencies of letter pairs in English.
THE FILES ARE ALREADY IN THE REPO , if you want to replace then after the clone make sure it will be the same names as above.
## Getting Started- Instructions
Based Windows
write the following in the treminal or powershell and press enter:

git clone https://github.com/DanielleLevy/GeneticAlgoritem.git
### 2 Ways to run:
#### first way:

write the following in the terminal or powershell , after each one press enter:

1. cd GeneticAlgoritem
2. if you want to replace the input files,Place them in the same directory as the genetic_algorithm.py script with the same names and delete the ones that came with the repo.
2. ./main.exe

##### NOTE : if you run from the CMD you should write: 

1. cd GeneticAlgoritem

2. main.exe


#### second way:
Go to the folder where the Repo file is located and double-click on the EXE file.



A GUI window will appear, allowing you to modify the parameters of the genetic algorithm (population size, crossover rate, mutation rate, and max generations). You can also select between Darwinian and Lamarckian modes. If you leave any parameter blank or provide invalid input, default values will be used.
Click the "Run Algorithm" button to start the genetic algorithm.
The algorithm will run for the specified number of generations or until convergence is reached (10 generations with no improvement in the average score).

## Output
1. Graph (plot.png): The code generates a graph that displays the average, best, and worst fitness scores for each generation. This graph provides a visual representation of the fitness scores over the course of the evolutionary process. Additionally, the graph includes the parameters used for the run and the number of function calls. The graph is saved in the code directory upon completion of the execution under the name "plot.png."

2. Plain text file (plain.txt): A plain text file named "plain.txt" is created in the code directory. This file contains the decoded text. It represents the output of the code after it has been processed and transformed into human-readable text.

3. Permutation table file (perm.txt): Another file named "perm.txt" is created in the code directory. This file contains the permutation table, which is a table that indicates the substitutions made during the execution of the code. It provides information about the changes or replacements that occurred during the execution process.

## Background
The Neural Network Pattern Learning project aims to develop a system that can identify patterns in binary strings using neural networks. The goal is to create neural networks capable of predicting the legality of binary strings based on patterns learned from training data.

Neural networks are computational models inspired by the structure and function of the human brain. They are composed of interconnected nodes, called neurons, organized into layers. Each neuron receives input signals, processes them, and produces an output signal that can be passed to other neurons. Neural networks have the ability to learn and generalize from input-output patterns, making them suitable for pattern recognition tasks.

In this project, a genetic algorithm-based approach is employed to build the neural networks. Genetic algorithms are optimization algorithms inspired by the process of natural selection and evolution. They work by iteratively generating a population of candidate solutions, evaluating their fitness, and using genetic operators (such as selection, crossover, and mutation) to evolve better solutions over generations.

## Code Explanation
The code consists of several modules and files that work together to implement the pattern learning system using neural networks. Here's an overview of the key components:

gui.py: This module provides a graphical user interface (GUI) for the user to interact with the system. It allows the user to choose options, input files, and run the building and running processes.

buildnet.py: This module contains the implementation of the genetic algorithm-based network building process. It defines the procedures for initializing the population, evaluating fitness, selecting parents, applying crossover and mutation, and generating the final network structure and weights.

runnet.py: This module implements the functionality for running the trained neural network on new data for classification. It takes the network structure and weights obtained from the building process and applies them to make predictions on test data.

main.py: This module contains the core functions used by buildnet.py and runnet.py. It includes functions for loading the input data, scoring the fitness of candidate networks, implementing the evolution operators, and executing the genetic algorithm.

The code expects two input data files, namely nn0.txt and nn1.txt, which contain binary strings along with their corresponding legality labels (0 or 1). nn0.txt is designed to be easier to identify legality patterns, while nn1.txt is more challenging.

During the building process, the genetic algorithm evolves a population of neural networks by iteratively selecting parents, performing crossover and mutation, and evaluating their fitness based on how well they predict the legality of the binary strings. The process continues until a satisfactory network structure and weights are obtained, which are saved in the wnet.txt file.

The runnet process utilizes the trained network structure and weights from wnet.txt to classify new binary strings and produce an output file with the classification results.

It is important to evaluate the performance of the trained neural networks on both the learning group (training data) and the test group. The accuracy of the networks in predicting the legality of binary strings can be used as a measure of performance.
## Notes
The genetic algorithm uses a fitness function to evaluate the quality of each candidate solution. The fitness function computes the log-probabilities of English letter and letter pair frequencies in the decrypted text, and also adds a bonus for each English word found.
The genetic_algorithm.py script includes the necessary functions to perform selection, crossover, mutation, and fitness scoring. You can modify these functions or experiment with different variations of the genetic algorithm as needed.
