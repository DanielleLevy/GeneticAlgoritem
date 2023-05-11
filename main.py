import math
import random
import string

# Parameters
POPULATION_SIZE = 100
ELITISM_RATIO = 0.1
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MAX_GENERATIONS = 1000
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


def fitness_score(candidate):
    """
    Compute the fitness score for a given candidate permutation.
    The score is the sum of the log-probabilities of the English letter and letter pair frequencies in the decoded text.
    """
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

def calc_score(population):
    global score_calls
    score_calls += 1
    fitness_scores = []
    sumFintnesScore = 0
    # Compute fitness scores
    for individual in population:
        score = fitness_score(individual)
        fitness_scores.append(score)
        sumFintnesScore += score
    avg_score=sumFintnesScore/len(population)
    return fitness_scores,sumFintnesScore,avg_score
def evolve_population(population,fitness_scores,sumFintnesScore):
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
    num_offspring = int(len(population) * CROSSOVER_RATE)
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
    num_mutants = int((len(population)  * MUTATION_RATE))
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


# Generate initial population
population = generate_initial_population(POPULATION_SIZE)
fitness_scores,sumFintnesScore,avg_score=calc_score(population)

# Evolve population
for generation in range(MAX_GENERATIONS):
    # Apply selection, crossover, and mutation operators to population
    population = evolve_population(population,fitness_scores,sumFintnesScore)
    fitness_scores, sumFintnesScore, avg_score = calc_score(population)
    population=cut_population(population,fitness_scores)


print('Finished')
