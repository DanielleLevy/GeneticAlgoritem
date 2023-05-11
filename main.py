import random
import string
import matplotlib.pyplot as plt
import tkinter as tk
# Parameters
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100
score_calls = 0


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

# Create a window
window = tk.Tk()
window.title("Genetic Algorithm Parameters")
window.geometry("300x200")

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


def input_check(population_size, crossover_rate, mutation_rate, max_generations):
    # check population_size
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


# Define a function to get the parameter values from the entry widgets
def get_params():
    pop_size = pop_size_entry.get()
    crossover_rate = crossover_rate_entry.get()
    mutation_rate = mutation_rate_entry.get()
    max_generations = max_generations_entry.get()
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
    window.destroy()
    run_genetic_algorithm(pop_size,crossover_rate,mutation_rate,max_generations)





def fitness_score(candidate):
    """
    Compute the fitness score for a given candidate permutation.
    The score is the sum of the log-probabilities of the English letter and letter pair frequencies in the decoded text.
    """
    # Convert INITIAL_PERMUTATION to a string before using it in str.maketrans()
    initial_permutation_str = ''.join(INITIAL_PERMUTATION)
    try:
        candidate_str = ''.join(candidate)
    except:
        print("")
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


def generate_initial_population(size):
    """
    Generate a random initial population of permutations of the alphabet.
    """
    population = []
    for i in range(size):
        perm = list(INITIAL_PERMUTATION)
        random.shuffle(perm)
        population.append(perm)
    return population


def select_parents(population):
    """
    Select two parents from the population using tournament selection.
    """
    candidates = random.sample(population, 5)
    candidates.sort(key=lambda x: fitness_score(x), reverse=True)
    return candidates[0], candidates[1]


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to produce two offspring.
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
    Mutate a candidate permutation by swapping two random elements.
    """
    pos1, pos2 = random.sample(range(PERMUTATION_SIZE), 2)
    candidate[pos1], candidate[pos2] = candidate[pos2], candidate[pos1]
    return candidate

def calc_score(population):
    global score_calls
    score_calls += 1
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
    return fitness_scores, sumFintnesScore, avg_score, worst_score

def evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate):
    """
    Evolves the given population by applying selection, crossover, and mutation operators.

    Args:
        population (list): A list of individuals in the population.

    Returns:
        list: The new population after applying selection, crossover, and mutation operators.
    """
    # Compute relative fitness scores
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

    # Combine elites, offspring, and mutants to form new population
    new_population = population + offspring + mutants
    return new_population


def cut_population(population, fitness_scores):
    # Sort the population indices by their fitness scores
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

    # Keep only the top 100 indices
    top_indices = sorted_indices[:100]

    # Create a new population containing only the top individuals
    new_population = [population[i] for i in top_indices]

    return new_population

def run_genetic_algorithm(population_size, crossover_rate, mutation_rate, max_generations):
    # Generate initial population
    generations=[]
    avg_scores = []  # replace with actual average scores for each generation
    bad_scores = []  # replace with actual bad scores for each generation
    population = generate_initial_population(population_size)
    # Evolve population
    for generation in range(max_generations):
        fitness_scores, sumFintnesScore, avg_score,worst_score = calc_score(population)
        generations.append(generation)
        avg_scores.append(avg_score)
        bad_scores.append(worst_score)
        # Apply selection, crossover, and mutation operators to population
        population = evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate)
        fitness_scores, sumFintnesScore, avg_score,worst_score = calc_score(population)
        population=cut_population(population,fitness_scores)



    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Set the title and axis labels
    ax.set_title("Average and Bad Scores by Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")

    # Set the bar width
    bar_width = 0.35

    # Plot the average scores as a blue bar
    ax.bar(generations, avg_scores, color='b', width=bar_width, label='Average Scores')

    # Plot the bad scores as a red bar
    ax.bar([g + bar_width for g in generations], bad_scores, color='r', width=bar_width, label='Bad Scores')

    # Add a legend
    ax.legend()

    # Display the chart
    plt.show()


def main():
    # create the button widget and add it to the window
    button = tk.Button(window, text="Run Genetic Algorithm", command=get_params)
    button.pack()

    # Start the window
    window.mainloop()

if __name__ == '__main__':
    main()
