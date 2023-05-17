# GeneticAlgoritem Cryptanalysis
This code implements a genetic algorithm for cryptanalysis, specifically for breaking a substitution cipher. The genetic algorithm aims to find the correct permutation of the English alphabet that maps the cipher text to its corresponding plaintext. The algorithm evolves a population of candidate solutions over multiple generations, using selection, crossover, and mutation operators.


## Requirements
Files
- enc.txt: The input cipher text that needs to be decrypted.
- dict.txt: A text file containing a list of English words. This is used to evaluate the fitness of the decrypted text.
- Letter_Freq.txt: A text file containing the frequencies of individual English letters.
- Letter2_Freq.txt: A text file containing the frequencies of letter pairs in English.

## Getting Started- Instructions
Based Windows
write the following in the treminal or powershell and press enter:

git clone https://github.com/DanielleLevy/GeneticAlgoritem.git
### 2 Ways to run:
#### first way:

write the following in the terminal or powershell , after each one press enter:

1. cd GeneticAlgoritem
2. Place the input files (enc.txt, dict.txt, Letter_Freq.txt, Letter2_Freq.txt) in the same directory as the genetic_algorithm.py script.
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

## Notes
The genetic algorithm uses a fitness function to evaluate the quality of each candidate solution. The fitness function computes the log-probabilities of English letter and letter pair frequencies in the decrypted text, and also adds a bonus for each English word found.
The genetic_algorithm.py script includes the necessary functions to perform selection, crossover, mutation, and fitness scoring. You can modify these functions or experiment with different variations of the genetic algorithm as needed.
