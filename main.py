import math
import random
import string

# Parameters
POPULATION_SIZE = 100
ELITISM_RATIO = 0.1
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MAX_GENERATIONS = 1000

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
ENGLISH_PAIR_FREQ={}
with open('Letter2_Freq.txt') as f:
    for line in f:
        frequency, characters = line.split()
        ENGLISH_PAIR_FREQ[characters] = float(frequency)

# Define representation
ALPHABET = string.ascii_lowercase  # Include space, newline, and some punctuation marks
PERMUTATION_SIZE = len(ALPHABET)
INITIAL_PERMUTATION = list(ALPHABET)
random.shuffle(INITIAL_PERMUTATION)


def fitness_score(candidate):
    """
    Compute the fitness score for a given candidate permutation.
    The score is the sum of the log-probabilities of the English letter and letter pair frequencies in the decoded text.
    """
    # Build substitution table
    table = str.maketrans(''.join(INITIAL_PERMUTATION), ''.join(candidate))

    # Decode text with substitution table
    decoded_text = CIPHER_TEXT.translate(table)

    # Compute letter and letter pair frequencies
    letter_freq = {}
    pair_freq = {}
    for i in range(len(decoded_text)-1):
        pair = decoded_text[i:i+2]
        if pair.isalpha():
            pair_freq[pair] = pair_freq.get(pair, 0) + 1
        letter = decoded_text[i]
        if letter.isalpha():
            letter_freq[letter] = letter_freq.get(letter, 0) + 1

    # Compute log-probabilities of English letter and letter pair frequencies in decoded text
    score = 0
    for letter, freq in letter_freq.items():
        true_freq = ENGLISH_LETTER_FREQ.get(letter)
        score += freq * math.log(true_freq)
    for pair, freq in pair_freq.items():
        true_freq = ENGLISH_PAIR_FREQ.get(pair)
        score += freq * math.log(true_freq)

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
    if random.random() < CROSSOVER_RATE:
        # Choose a random crossover point
        crossover_point = random.randint(1, PERMUTATION_SIZE - 2)

        # Perform one-point crossover
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        return offspring1, offspring2
    else:
        # If no crossover occurs, return copies of the parents
        return parent1[:], parent2[:]


def mutate(candidate):
    """
    Mutate a candidate permutation by swapping two random elements.
    """
    if random.random() < MUTATION_RATE:
        # Choose two random positions to swap
        pos1, pos2 = random.sample(range(PERMUTATION_SIZE), 2)
        candidate[pos1], candidate[pos2] = candidate[pos2], candidate[pos1]

def evolve_population(population):
    """
    Evolves the given population by applying selection, crossover, and mutation operators.

    Args:
        population (list): A list of individuals in the population.

    Returns:
        list: The new population after applying selection, crossover, and mutation operators.
    """
    # Compute fitness scores
    fitness_scores = [fitness_score(individual) for individual in population]

    # Select individuals for reproduction
    num_elites = int(ELITISM_RATIO * len(population))
    elites = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:num_elites]
    mating_pool = [population[i] for i in elites]

    # Perform crossover
    num_offspring = int((len(population) - num_elites) * CROSSOVER_RATE)
    offspring = []
    for i in range(num_offspring):
        parent1, parent2 = random.sample(mating_pool, 2)
        child = crossover(parent1, parent2)
        offspring.append(child)

    # Perform mutation
    num_mutants = int((len(population) - num_elites) * MUTATION_RATE)
    mutants = []
    for i in range(num_mutants):
        parent = random.choice(mating_pool)
        mutant = mutate(parent)
        mutants.append(mutant)

    # Combine elites, offspring, and mutants to form new population
    new_population = [population[i] for i in elites] + offspring + mutants

    return new_population

# Generate initial population
population = generate_initial_population(POPULATION_SIZE)

# Evolve population
for generation in range(MAX_GENERATIONS):
    # Apply selection, crossover, and mutation operators to population
    population = evolve_population(population)

    # Compute fitness of best candidate in population
    best_candidate = max(population, key=lambda x: fitness_score(x))
    best_fitness = fitness_score(best_candidate)

    # Print status update
    print(f'Generation {generation + 1}, Best fitness: {best_fitness:.2f}, Best candidate: {" ".join(best_candidate)}')

    # Stop if solution is found
    if best_fitness == len(CIPHER_TEXT):
        print('Solution found:')
        print(CIPHER_TEXT.translate(str.maketrans(''.join(best_candidate), ALPHABET)))
        break

print('Finished')

