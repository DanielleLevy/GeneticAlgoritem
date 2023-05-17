# GeneticAlgoritem Cryptanalysis
This code implements a genetic algorithm for cryptanalysis, specifically for breaking a substitution cipher. The genetic algorithm aims to find the correct permutation of the English alphabet that maps the cipher text to its corresponding plaintext. The algorithm evolves a population of candidate solutions over multiple generations, using selection, crossover, and mutation operators.


## Requirements
Python 3.x
matplotlib library
numpy library
tkinter library (usually comes pre-installed with Python)
Files
genetic_algorithm.py: The main Python script that contains the implementation of the genetic algorithm.
enc.txt: The input cipher text that needs to be decrypted.
dict.txt: A text file containing a list of English words. This is used to evaluate the fitness of the decrypted text.
Letter_Freq.txt: A text file containing the frequencies of individual English letters.
Letter2_Freq.txt: A text file containing the frequencies of letter pairs in English.

## Usage
Make sure you have Python 3.x and the required libraries installed.
Place the input files (enc.txt, dict.txt, Letter_Freq.txt, Letter2_Freq.txt) in the same directory as the genetic_algorithm.py script.
Run the genetic_algorithm.py script using Python.
A GUI window will appear, allowing you to modify the parameters of the genetic algorithm (population size, crossover rate, mutation rate, and max generations). You can also select between Darwinian and Lamarckian modes. If you leave any parameter blank or provide invalid input, default values will be used.
Click the "Run Algorithm" button to start the genetic algorithm.
The algorithm will run for the specified number of generations or until convergence is reached (10 generations with no improvement in the average score).
The algorithm will output the decrypted plaintext, the best permutation, and generate a plot (plot.png) showing the average, bad, and best scores over generations. The decrypted plaintext will be saved in plain.txt, and the permutation will be saved in perm.txt.
You can check the convergence and the number of fitness function calls in the console output.

## Notes
The genetic algorithm uses a fitness function to evaluate the quality of each candidate solution. The fitness function computes the log-probabilities of English letter and letter pair frequencies in the decrypted text, and also adds a bonus for each English word found.
The genetic_algorithm.py script includes the necessary functions to perform selection, crossover, mutation, and fitness scoring. You can modify these functions or experiment with different variations of the genetic algorithm as needed.
