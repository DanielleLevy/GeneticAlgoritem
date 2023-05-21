import random
import string
import matplotlib.pyplot as plt
import tkinter as tk

import numpy as np

# Parameters
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100
score_calls = 0
N=10


# Load data
with open('enc.txt', 'r') as f:
    CIPHER_TEXT = f.read()

with open('dict.txt', 'r') as f:
    ENGLISH_WORDS = set(word.strip().lower() for word in f)

ENGLISH_LETTER_FREQ = {}
with open('Letter_Freq.txt') as f:
    for line in f:
        frequency, character = line.split()
        ENGLISH_LETTER_FREQ[character] = float(frequency)
ENGLISH_PAIR_FREQ = {}
with open('Letter2_Freq.txt') as f:
    for line in f:
        frequency, characters = line.split()
        ENGLISH_PAIR_FREQ[characters] = float(frequency)


# Define representation
ALPHABET = string.ascii_lowercase
PERMUTATION_SIZE = len(ALPHABET)
INITIAL_PERMUTATION = list(ALPHABET)
random.shuffle(INITIAL_PERMUTATION)

def update_checkboxes():
    """
The update_checkboxes function is called when either the Darwinian or Lamarckian checkbox is selected.
It ensures that only one of the two checkboxes can be selected at a time.

:return: Nothing
"""
    if darwinian_var.get() == 1:
        lamarckian_checkbox.deselect()
    if lamarckian_var.get() == 1:
        darwinian_checkbox.deselect()


# Create a window
window = tk.Tk()
window.title("Genetic Algorithm Parameters")
window.geometry("300x250")

# Create labels and entry widgets for each parameter
pop_size_label = tk.Label(window, text="Population size:")
pop_size_label.pack()
pop_size_entry = tk.Entry(window)
pop_size_entry.insert(0, "100")
pop_size_entry.pack()

crossover_rate_label = tk.Label(window, text="Crossover rate:")
crossover_rate_label.pack()
crossover_rate_entry = tk.Entry(window)
crossover_rate_entry.insert(0, "0.8")
crossover_rate_entry.pack()

mutation_rate_label = tk.Label(window, text="Mutation rate:")
mutation_rate_label.pack()
mutation_rate_entry = tk.Entry(window)
mutation_rate_entry.insert(0, "0.1")
mutation_rate_entry.pack()

max_generations_label = tk.Label(window, text="Max generations:")
max_generations_label.pack()
max_generations_entry = tk.Entry(window)
max_generations_entry.insert(0, "100")
max_generations_entry.pack()

# Create checkboxes for Darwinian and Lamarckian modes
darwinian_var = tk.IntVar()
darwinian_checkbox = tk.Checkbutton(window, text="Darwinian", variable=darwinian_var, command=update_checkboxes)
darwinian_checkbox.pack()

lamarckian_var = tk.IntVar()
lamarckian_checkbox = tk.Checkbutton(window, text="Lamarckian", variable=lamarckian_var, command=update_checkboxes)
lamarckian_checkbox.pack()


def input_check(population_size, crossover_rate, mutation_rate, max_generations):
    """
The input_check function checks the validity of the user input.
    It takes in four parameters: population_size, crossover_rate, mutation_rate and max_generations.
    The function first checks if all inputs are integers or floats by using try-except statement. If any of them is not an integer or float, it returns False immediately. 
    Then it checks if population size is positive and max generations is positive by checking whether they are greater than 0 respectively. 
    Finally it checkes whether crossover rate and mutation rate are between 0 to 1 (inclusive) by checking whether they are less than 1 and greater

:param population_size: Specify the size of the population
:param crossover_rate: Determine the probability of a crossover occurring
:param mutation_rate: Determine the probability of a mutation occuring
:param max_generations: Determine the number of generations that will be run
:return: True if all the inputs are valid, and false otherwise
"""
    try:
        population_size = int(population_size)
        crossover_rate=float(crossover_rate)
        mutation_rate=float(mutation_rate)
        max_generations=int(max_generations)
    except:
        return False

    if population_size <= 0:
        return False

    # check crossover_rate
    if crossover_rate < 0 or crossover_rate > 1:
        return False

    # check mutation_rate
    if mutation_rate < 0 or mutation_rate > 1:
        return False

    # check max_generations
    if max_generations <= 0:
        return False

    return True


def get_params():
    """
The get_params function is used to retrieve the parameters entered by the user in the GUI.
If no values are entered, default values are returned.
The function returns a tuple of 6 elements: pop_size, crossover_rate, mutation_rate and max_generations as integers; darwinian mode and lamarckian mode as booleans.

:return: The parameters entered by the user
"""
    pop_size = pop_size_entry.get()
    crossover_rate = crossover_rate_entry.get()
    mutation_rate = mutation_rate_entry.get()
    max_generations = max_generations_entry.get()
    darwinian_mode = darwinian_var.get()
    lamarckian_mode = lamarckian_var.get()
    if input_check(pop_size,crossover_rate,mutation_rate,max_generations):
        pop_size = int(pop_size)
        crossover_rate = float(crossover_rate)
        mutation_rate = float(mutation_rate)
        max_generations = int(max_generations)
    else:
        pop_size = POPULATION_SIZE
        crossover_rate = CROSSOVER_RATE
        mutation_rate = MUTATION_RATE
        max_generations=MAX_GENERATIONS
    return pop_size,crossover_rate,mutation_rate,max_generations,darwinian_mode,lamarckian_mode


def generate_initial_population(size):
    """
The generate_initial_population function takes a single argument, size, which is the number of permutations to generate.
It returns a list of lists (a 2D array) where each inner list represents one permutation. The length of the outer list
is equal to size and each inner list has length len(INITIAL_PERMUTATION). Each element in an inner list is an integer from 0-9.

:param size: Determine the size of the population
:return: A list of size 'size'
"""
    population = []
    for i in range(size):
        perm = list(INITIAL_PERMUTATION)
        random.shuffle(perm)
        population.append(perm)
    return population

def fitness_score(candidate):
    """
The fitness_score function takes a candidate solution and returns its fitness score.
The fitness score is the sum of the log-probabilities of English letter and letter pair frequencies in decoded text,
plus a bonus for each English word in decoded text. The higher the fitness score, the better.

:param candidate: Pass the candidate solution to the fitness_score function
:return: A score that is the sum of two components:
"""
    global score_calls
    score_calls += 1
    # Convert INITIAL_PERMUTATION to a string before using it in str.maketrans()
    initial_permutation_str = ''.join(INITIAL_PERMUTATION)
    candidate_str = ''.join(candidate)

    # Build substitution table
    table = str.maketrans(initial_permutation_str, candidate_str)

    # Decode text with substitution table
    decoded_text = CIPHER_TEXT.translate(table)

    # Compute letter and letter pair frequencies
    letter_freq = {}
    pair_freq = {}
    for i in range(len(decoded_text) - 1):
        pair = decoded_text[i:i + 2]
        if pair.isalpha():
            pair_freq[pair] = pair_freq.get(pair, 0) + 1
        letter = decoded_text[i]
        if letter.isalpha():
            letter_freq[letter] = letter_freq.get(letter, 0) + 1

    # Compute log-probabilities of English letter and letter pair frequencies in decoded text
    score = 0
    total_letters = sum(letter_freq.values())
    total_pairs = sum(pair_freq.values())

    for letter, freq in letter_freq.items():
        letter_freq[letter]=letter_freq.get(letter)/total_letters

    for pair, freq in pair_freq.items():
        pair_freq[pair]=pair_freq.get(pair)/total_pairs

    for letter, freq in letter_freq.items():
        true_freq = ENGLISH_LETTER_FREQ.get(letter.upper())
        if (freq <= true_freq + 0.01) & (freq >= true_freq - 0.01):
            score += 0.5
    for pair, freq in pair_freq.items():
        true_freq = ENGLISH_PAIR_FREQ.get(pair.upper())
        if (freq <= true_freq + 0.01) & (freq >= true_freq - 0.01):
            score += 0.5

    # Add bonus for each English word in decoded text
    decoded_words = set(decoded_text.lower().split())
    english_words = decoded_words & ENGLISH_WORDS
    score += len(english_words)

    return score

def crossover(parent1, parent2):
    """
The crossover function takes two parents and returns an offspring.
    The crossover function performs a one-point crossover on the parents,
    returning the offspring. If there are duplicate letters in the offspring,
    they are replaced with unused letters from ALPHABET.

:param parent1: Pass in the parent that is being mutated
:param parent2: Create the offspring2
:return: A new individual, which is the result of
"""
    crossover_point = random.randint(1, PERMUTATION_SIZE - 2)

    # Perform one-point crossover
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    # Check for duplicates in offspring1
    unique_letters = set(ALPHABET)
    offspring_letters = set()
    duplicate_letters = []
    for letter in offspring1:
        if letter in offspring_letters:
            duplicate_letters.append(letter)
        else:
            offspring_letters.add(letter)
    duplicate_letters = list(set(duplicate_letters))

    # Replace duplicate letters with unused letters
    for element in duplicate_letters:
        available_elements = list(unique_letters - offspring_letters)
        replacement_element = available_elements.pop()
        for i in range(len(offspring1)):
            if offspring1[i] == element:
                offspring1[i] = replacement_element
                offspring_letters.add(replacement_element)
                break

    return offspring1



def mutate(candidate):
    """
The mutate function takes a candidate solution as input and returns a candidate solution with one of its
elements randomly swapped with another. This function helps maintain genetic diversity in the population by
preventing the algorithm from getting stuck at local optima.

:param candidate: Pass the candidate to be mutated
:return: A new candidate
"""
    mutated_candidate = candidate.copy()  # Create a copy of the candidate
    pos1, pos2 = random.sample(range(len(candidate)), 2)
    mutated_candidate[pos1], mutated_candidate[pos2] = mutated_candidate[pos2], mutated_candidate[pos1]
    return mutated_candidate


def calc_score(population):
    """
The calc_score function takes in a population and returns the fitness scores, sum of fitness scores, average score, worst score and best score.
    Args:
        population (list): A list of individuals.

:param population: Pass in the population of individuals
:return: The fitness scores, sum of the fitness scores, average score and worst score
"""
    fitness_scores = []
    sumFintnesScore = 0
    worst_score = float('inf')
    # Compute fitness scores
    for individual in population:
        score = fitness_score(individual)
        fitness_scores.append(score)
        sumFintnesScore += score
        if score < worst_score:
            worst_score = score
    avg_score = sumFintnesScore / len(population)
    best_score=max(fitness_scores)
    return fitness_scores, sumFintnesScore, avg_score, worst_score,best_score

def evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate,darwinOrLamrk=0):
    """
The evolve_population function takes in a population, fitness scores, crossover rate and mutation rate.
It then computes the relative fitness scores for each individual in the population. It then creates a vector of indices
that is shuffled to randomize it. The function then performs crossover on num_offspring individuals from this vector and 
performs mutation on num_mutants individuals from this vector.

:param population: Pass in the current population of chromosomes
:param fitness_scores: Compute the relative fitness scores
:param sumFintnesScore: Compute the relative fitness scores
:param crossover_rate: Determine the number of offspring that will be created
:param mutation_rate: Determine how many members of the population will be mutated
:param darwinOrLamrk: Determine whether to use the darwinian or lamarckian approach
:return: A new population, but we need to
"""
    relative_fitness_scores = []
    for score in fitness_scores:
        relative_fitness_score = score / sumFintnesScore
        relative_fitness_scores.append(relative_fitness_score)
    vector = []
    for i, score in enumerate(relative_fitness_scores):
        num_times = int(score * sumFintnesScore)
        vector += [i] * num_times
    # Shuffle the vector
    random.shuffle(vector)


    # Perform crossover
    num_offspring = int(len(population) * crossover_rate)
    offspring = []
    for i in range(num_offspring):
        # Choose two random elements from the vector, ensuring they are distinct
        selection = []
        while len(selection) < 2:
            new_element = random.choice(vector)
            if new_element not in selection:
                selection.append(new_element)
        child = crossover(population[selection[0]], population[selection[1]])
        offspring.append(child)

    # Perform mutation
    num_mutants = int((len(population)  * mutation_rate))
    mutants = []
    for i in range(num_mutants):
        parent = random.choice(population)
        mutant = mutate(parent)
        mutants.append(mutant)
    if(darwinOrLamrk==1):
        new_population = offspring + mutants
    else:
        new_population = population + offspring + mutants
    return new_population


def cut_population(population, fitness_scores):
    """
The cut_population function takes in a population and fitness scores, sorts the indices of the
fitness_scores list from highest to lowest, then creates a new population containing only the top 100 individuals.
It returns this new population and its corresponding fitness scores.

:param population: Store the population of individuals
:param fitness_scores: Sort the population by fitness
:return: The top 100 individuals from the population
"""
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

    # Keep only the top 100 indices
    top_indices = sorted_indices[:100]

    # Create a new population containing only the top individuals
    new_population = [population[i] for i in top_indices]
    fitness_scores=[fitness_scores[i] for i in top_indices]

    return new_population,fitness_scores

def chosen_sol(population,fitness_scores):
    """
The chosen_sol function takes in the population and fitness scores of that population.
It then sorts the indices of the fitness scores from highest to lowest, and returns 
the solution with the highest score as well as its corresponding score.

:param population: Pass the population of solutions to the function
:param fitness_scores: Find the best solution from the population
:return: The best solution and its fitness score
"""
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    solution=population[sorted_indices[0]]
    best_f=fitness_scores[sorted_indices[0]]

    return solution,best_f
def encode_txt(solution):
    """
The encode_txt function takes a solution as input and returns the decoded text.
    The function first builds a substitution table using the initial permutation string and the solution.
    Then, it uses this table to decode the cipher text.

:param solution: Encode the text
:return: The decoded text and the permutation dictionary
"""
    initial_permutation_str = ''.join(INITIAL_PERMUTATION)
    # Build substitution table
    table = str.maketrans(initial_permutation_str, ''.join(solution))
    # Decode text with substitution table
    decoded_text = CIPHER_TEXT.translate(table)
    perm_dict = {INITIAL_PERMUTATION[i]: solution[i] for i in range(len(INITIAL_PERMUTATION))}
    sorted_perm_dict = dict(sorted(perm_dict.items()))
    return decoded_text,sorted_perm_dict
def run_genetic_algorithm(population_size, crossover_rate, mutation_rate, max_generations,isfirst=0):
    """
The run_genetic_algorithm function runs the genetic algorithm.
    Args:
        population_size (int): The number of individuals in each generation.
        crossover_rate (float): The probability that two parents will mate and produce an offspring, which receives traits from each parent. Must be between 0 and 1, where 0 represents a 0% chance of crossover taking place and 1 represents a 100% chance of crossover taking place. Note that you can use Python's built-in round function to round this value to the nearest hundredth (e.g., round(0.654, 2)) if necessary for printing or comparison purposes).


:param population_size: Determine the size of the population
:param crossover_rate: Determine the probability of crossover
:param mutation_rate: Determine the probability of a mutation occuring
:param max_generations: Determine how many times the algorithm will run
:param isfirst: Check if this is the first time we run the algorithm
:return: A tuple containing:
"""
    global score_calls
    score_calls=0
    convergence=0
    # Generate initial population
    generations=[]
    avg_scores = []  # replace with actual average scores for each generation
    bad_scores = []  # replace with actual bad scores for each generation
    best_scores=[]
    population = generate_initial_population(population_size)
    # Evolve population
    for generation in range(max_generations):
        fitness_scores, sumFintnesScore, avg_score,worst_score,best_score = calc_score(population)
        generations.append(generation)
        avg_scores.append(avg_score)
        bad_scores.append(worst_score)
        best_scores.append(best_score)
        # Apply selection, crossover, and mutation operators to population
        population = evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate)
        fitness_scores, sumFintnesScore, avg_score,worst_score,best_score = calc_score(population)
        population,fitness_scores=cut_population(population,fitness_scores)
        avg_score=sum(fitness_scores)/len(fitness_scores)
        if abs(avg_score-avg_scores[-1]) <= 1:
            convergence += 1
        else:
            convergence=0
        if convergence == 10:
            sol, fitness = chosen_sol(population, fitness_scores)
            return True, sol, fitness, generations, avg_scores, bad_scores,best_scores,score_calls
    sol, fitness = chosen_sol(population, fitness_scores)
    if(isfirst==1):
        graph_and_txt(sol,generations,avg_scores,bad_scores,best_scores,population_size,crossover_rate,mutation_rate,max_generations,score_calls)
    return False, sol, fitness, generations, avg_scores, bad_scores,best_scores,score_calls

def graph_and_txt(sol, generations, avg_scores, bad_scores, best_scores, pop_size, crossover_rate, mutation_rate, max_generations,score_call):
    """
The graph_and_txt function takes in the solution, generations, average scores, bad scores, best scores and parameters
    for the genetic algorithm. It then creates a plot of the average score distribution over time as well as a line plot
    of the best score over time. The function also encodes and writes out both plaintext and permutation files.

:param sol: Pass the solution to the function
:param generations: Store the generation number for each score
:param avg_scores: Plot the average scores over time
:param bad_scores: Plot the bad scores as a red bar
:param best_scores: Plot the best scores over time
:param pop_size: Set the size of the population
:param crossover_rate: Determine the probability of crossover
:param mutation_rate: Determine the probability of a mutation occuring
:param max_generations: Set the maximum number of generations to run for
:param score_call: Count the number of times the fitness function is called
:return: The plain text and the permutation dictionary
"""
    plain, perm = encode_txt(sol)
    with open("plain.txt", "w") as file:
        file.write(plain)
    with open("perm.txt", "w") as file:
        for char in perm:
            file.write(f"{char}: {perm[char]}\n")

    # Create a figure and axis object
    fig, ax = plt.subplots( figsize=(14, 6))

    # Set the axis labels and title for the score distribution plot
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    ax.set_title("Average, Bad, and Best Scores by Generation")

    # Set the bar width
    bar_width = 0.35

    # Plot the average scores as a blue bar
    ax.bar(generations, avg_scores, color='#5DA5DA', width=bar_width, label='Average Scores')

    # Plot the bad scores as a red bar
    ax.bar([g + bar_width for g in generations], bad_scores, color='#FAA43A', width=bar_width, label='Bad Scores')

    # Add a line plot of the best scores over time
    ax.plot(generations, best_scores, color='#60BD68', linewidth=2, label='Best Scores')

    # Add the parameters as text above the title
    plt.text(0.5, 1.1, f"Population size: {pop_size}   Crossover rate: {crossover_rate}   Mutation rate: {mutation_rate}   Max generations: {max_generations}   Fitness function's calls: {score_calls}", ha='center', va='bottom', transform=ax.transAxes)
    # Add a legend to the score distribution plot
    ax.legend()

    # Save the figure as a PNG file
    plt.savefig("plot.png")

    # Show the figure
    plt.show()


def run_genetic_algorithm_wrapper_to_check_conv(N=0,darwin=0,lamrk=0,checkparam=0,pop_size=0 ,crossover_rate=0, mutation_rate=0, max_generations=0):
    """
The run_genetic_algorithm_wrapper_to_check_conv function is a wrapper function that runs the genetic algorithm with different mutation rates and returns the best solution.
    Args:
        N (int): The number of queens to be placed on an NxN board. Default value is 0, which means it will ask for user input.
        darwin (int): If 1, then run Darwinian Genetic Algorithm; if 0, then run regular GA or Lamarckian GA depending on lamrk parameter value. Default value is 0. 
        lamrk (int): If 1, then run Lamarckian Genetic Algorithm; if 0, then

:param N: Define the number of queens in the nxn board
:param darwin: Determine whether the algorithm is darwinian or lamarkian
:param lamrk: Determine if the algorithm is lamarkian or darwinian
:param checkparam: Check if the function is being called from the wrapper or not
:param pop_size: Set the population size
:param crossover_rate: Determine the probability of crossover
:param mutation_rate: Check the convergence of the algorithm for different mutation rates
:param max_generations: Set the maximum number of generations that the algorithm will run for
:return: The average, worst and best scores for the genetic algorithm
"""
    if(checkparam==0):
        pop_size, crossover_rate, mutation_rate, max_generations,darwin,lamrk=get_params()
        window.destroy()
    if(darwin==1 or lamrk==1):
        flag, sol, fitness, generations, avg_scores, worst_score, best_scores, score_call = darwinian_or_lamark_genetic_algorithm(
            pop_size, crossover_rate, mutation_rate, max_generations, N, darwin, 1)
    else:
        flag, sol, fitness, generations, avg_scores, worst_score, best_scores, score_call = run_genetic_algorithm(
            pop_size, crossover_rate, mutation_rate, max_generations, 1)

    best_fitness_scores=[]
    solutions=[]
    generations_conv=[]
    avg_scores_conv=[]
    bad_scores=[]
    best_scores_conv=[]
    score_calls_conv=[]
    mutation_rates=[]
    if(flag==True):
        for i in range(5):
            best_fitness_scores.append(fitness)
            solutions.append(sol)
            generations_conv.append(generations)
            avg_scores_conv.append(avg_scores)
            bad_scores.append(worst_score)
            best_scores_conv.append(best_scores)
            mutation_rates.append(mutation_rate)
            score_calls_conv.append(score_call)
            mutation_rate=random.uniform(mutation_rate, 1)
            if (darwin == 1 or lamrk == 1):
                flag, sol, fitness, generations, avg_scores, worst_score, best_scores, score_call = darwinian_or_lamark_genetic_algorithm(
                    pop_size, crossover_rate, mutation_rate, max_generations, N, darwin, 0)
            else:
                flag, sol, fitness, generations, avg_scores, worst_score, best_scores, score_call = run_genetic_algorithm(
                    pop_size, crossover_rate, mutation_rate, max_generations, 0)

        max_index = best_fitness_scores.index(max(best_fitness_scores))
        graph_and_txt(solutions[max_index],generations_conv[max_index],avg_scores_conv[max_index],bad_scores[max_index],best_scores_conv[max_index],pop_size,crossover_rate,mutation_rates[max_index],generations_conv[max_index][-1],score_calls_conv[max_index])
        if(checkparam == 1):
            return avg_scores_conv[max_index], bad_scores[max_index], best_scores_conv[max_index],score_calls_conv[max_index]


def local_optimum(candidate,old_fitness,N,darwin):
    """
The local_optimum function takes in a candidate, an old fitness score, the number of iterations to run through
and whether or not we are using darwinian selection. It then runs through N iterations of mutating the candidate and 
checking if it is better than the previous one. If it is better, then that becomes our new candidate and we continue 
to iterate until N has been reached. If darwinian selection is on (darwin=True), then we return the original genom as well as its fitness score.

:param candidate: Pass the current genom to the function
:param old_fitness: Compare the fitness of the new genom with that of the old one
:param N: Determine how many times the local_optimum function will run
:param darwin: Determine if the function should return the original candidate or not
:return: The best candidate and its fitness score
"""
    if(darwin==1):
        original=candidate
    for n in range(N):
        new_genom=mutate(candidate)
        new_fit=fitness_score(new_genom)
        if(new_fit>=old_fitness):
            old_fitness=new_fit
            candidate=new_genom
    if(darwin==1):
        return original,old_fitness
    return candidate,old_fitness

def darwinian_or_lamark_genetic_algorithm(population_size, crossover_rate, mutation_rate, max_generations,N,darwin=0,isfirst=0):
    """
The darwinian_or_lamark_genetic_algorithm function is a genetic algorithm that uses the darwinian or lamark selection method.
    The function receives:
        population_size - number of individuals in each generation (int)
        crossover_rate - probability of crossover operation (float between 0 and 1)
        mutation_rate - probability of mutation operation (float between 0 and 1)
        max_generations - maximum number of generations to evolve for before returning best individual found so far(int). If this value is negative, then the GA will run until convergence.  Note that if you set this parameter to a large value, your GA may take

:param population_size: Determine the size of the population
:param crossover_rate: Determine the probability of a crossover operation being performed on two individuals
:param mutation_rate: Determine the probability of mutation for each gene
:param max_generations: Set the number of generations to run
:param N: Determine the number of queens on the board
:param darwin: Determine whether the algorithm is darwinian or lamark
:param isfirst: Determine if the graph and txt files should be created
:return: A tuple of the following values:
"""
    global score_calls
    score_calls=0
    convergence=0
    # Generate initial population
    generations=[]
    avg_scores = []  # replace with actual average scores for each generation
    bad_scores = []  # replace with actual bad scores for each generation
    best_scores=[]
    population = generate_initial_population(population_size)
    fitness_scores=[0]*population_size
    for generation in range(max_generations):
        for i,individual in enumerate(population):
            individual,fitness=local_optimum(individual,fitness_scores[i],N,darwin)
            population[i]=individual
            fitness_scores[i]=fitness
        sumFintnesScore=sum(fitness_scores)
        avg_score=sumFintnesScore/len(fitness_scores)
        worst_score=min(fitness_scores)
        best_score=max(fitness_scores)
        generations.append(generation)
        avg_scores.append(avg_score)
        bad_scores.append(worst_score)
        best_scores.append(best_score)
        # Apply selection, crossover, and mutation operators to population
        new_population = evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate,1)
        new_fitness_scores, new_sumFintnesScore, avg_score,worst_score,best_score = calc_score(new_population)
        population=population+new_population
        fitness_scores=fitness_scores+new_fitness_scores
        population,fitness_scores=cut_population(population,fitness_scores)
        sumFintnesScore = sum(fitness_scores)
        avg_score = sumFintnesScore / len(fitness_scores)
        if abs(avg_score-avg_scores[-1]) <= 1:
            convergence += 1
        else:
            convergence=0
        if convergence == 10:
            sol, fitness = chosen_sol(population, fitness_scores)
            return True, sol, fitness, generations, avg_scores, bad_scores,best_scores,score_calls
    sol, fitness = chosen_sol(population, fitness_scores)
    if(isfirst==1):
        graph_and_txt(sol,generations,avg_scores,bad_scores,best_scores,population_size,crossover_rate,mutation_rate,max_generations,score_calls)
    return False, sol, fitness, generations, avg_scores, bad_scores,best_scores,score_calls
def compare():
    """
The compare function is used to compare the performance of different genetic algorithms.
    The function takes no arguments and returns nothing. It simply plots the results of running
    a regular genetic algorithm, a lamarkian genetic algorithm, and a darwinian genetic algorithm
    on the same problem with identical parameters.

:return: The best fitness scores, average fitness scores and the number of calls to score function
"""
    N =[2,5,8,10,15,20]
    crossover_rate = 0.8
    mutation_rate = 0.1
    pop_size = 100
    max_gen=[50,80,100,150,200,250]
    best_fitness_scores = []
    avg_scores = []
    bad_scores = []
    score_calls = []
    #tags=['regular','lamrak','darwin']
    #avg, bad, best, score_call = run_genetic_algorithm_wrapper_to_check_conv(0,0,0,1, pop_size, crossover_rate, mutation_rate,
                                                                             #max_gen)
    #best_fitness_scores.append(best)
    #avg_scores.append(avg)
    #bad_scores.append(bad)
    #score_calls.append(score_call)
    #for i in range(2):
        #avg, bad, best, score_call = run_genetic_algorithm_wrapper_to_check_conv(10,i,1-i,1, pop_size, crossover_rate,
                                                                                # mutation_rate,
                                                                                 #max_gen)
        #best_fitness_scores.append(best)
        #avg_scores.append(avg)
        #bad_scores.append(bad)
        #score_calls.append(score_call)
    for i, param in enumerate(max_gen):
        #avg, bad, best, score_call = run_genetic_algorithm_wrapper_to_check_conv(param,0,1,1 ,pop_size, crossover_rate, mutation_rate,max_gen)
        avg, bad, best, score_call = run_genetic_algorithm_wrapper_to_check_conv(0,0,0,1 ,pop_size, crossover_rate, mutation_rate,param)

        best_fitness_scores.append(best)
        avg_scores.append(avg)
        bad_scores.append(bad)
        score_calls.append(score_call)
    plot_results(max_gen, best_fitness_scores, avg_scores, bad_scores, score_calls)


def plot_results(params, best_fitness_scores, avg_scores, bad_scores, score_calls):
    """
The plot_results function takes in the parameters, best fitness scores, average fitness scores, worst
fitness scores and number of score calls. It then plots these results on a 2x2 grid of subplots. The first
subplot shows the best fitness score for each parameter value over time (i.e., generation). The second subplot
shows the average fitness score for each parameter value over time (i.e., generation). The third subplot shows
the worst fitness score for each parameter value over time (i.e., generation). Finally, the fourth plot shows a bar chart with one bar per run showing

:param param: Set the title of the plots
:param best_fitness_scores: Plot the best fitness scores for each run
:param avg_scores: Store the average fitness scores for each generation
:param bad_scores: Plot the worst fitness scores
:param score_calls: Plot the number of score calls in a bar chart
:return: A plot of the best, average and worst fitness scores
"""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'gray']
    handles = []

    # Plot best fitness scores
    ax = axs[0]
    for i, param in enumerate(params):
        line, = ax.plot(best_fitness_scores[i], label=f'max_gen:{param}', color=colors[i])
        handles.append(line)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best fitness score')
    ax.set_title('Best fitness score')

    # Plot average fitness scores
    ax = axs[1]
    for i, param in enumerate(params):
        line, = ax.plot(avg_scores[i], label=f'max_gen: {param}', color=colors[i])
        handles.append(line)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average fitness score')
    ax.set_title('Average fitness score')

    # Plot worst fitness scores
    ax = axs[2]
    for i, param in enumerate(params):
        line, = ax.plot(bad_scores[i], label=f'max_gen:{param}', color=colors[i])
        handles.append(line)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Worst fitness score')
    ax.set_title('Worst fitness score')

    # Plot number of score calls
    ax = axs[3]
    x_pos = np.arange(len(score_calls))  # Generate x positions for bars
    width = 0.2
    bars = ax.bar(x_pos, score_calls, width, color=colors[:len(x_pos)])
    handles.append(bars)

    ax.set_xlabel('run')
    ax.set_ylabel('Number of score calls')
    ax.set_title('Number of score calls')

    fig.legend(handles=handles, labels=[f' max_gen:{param}' for param in params], loc='center left')

    plt.subplots_adjust(left=0.2, wspace=0.3, hspace=0.4)
    plt.tight_layout()
    plt.show()



def main():

    compare()
    """
The main function is the entry point of the program.
It creates a button that runs the genetic algorithm when clicked.


:return: The best individual
"""
    #button = tk.Button(window, text="Run Genetic Algorithm",
                       #command=lambda: run_genetic_algorithm_wrapper_to_check_conv(N))
    #button.pack()
    #window.mainloop()



if __name__ == '__main__':
    main()
