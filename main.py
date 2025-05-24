"""
Evolutionary Algorithm (EA) Implementation with Artistic Visualization

This module implements a binary-coded evolutionary algorithm to find the maximum of a simple
quadratic function f(x) = -(x-5)² + 100. The algorithm uses tournament selection, single-point
crossover, and bit-flip mutation. The implementation includes comprehensive tracking of the
evolutionary process and creates an artistic visualization of the algorithm's performance.

The visualization includes:
- Population evolution on the fitness landscape
- Fitness convergence over generations
- Population diversity tracking
- Final population distribution
- Algorithm statistics and parameters

Author: [Your Name]
Date: [Current Date]
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import seaborn as sns

# Set the artistic style
plt.style.use('dark_background')
sns.set_palette("viridis")

# --- Your EA Parameters ---
POPULATION_SIZE = 20
INDIVIDUAL_LENGTH = 8
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUMBER_OF_GENERATIONS = 30

# --- Modified EA with tracking ---
def create_individual(length=INDIVIDUAL_LENGTH):
    """
    Creates a single individual in the population as a binary string.
    
    Args:
        length (int): Length of the binary string (default: INDIVIDUAL_LENGTH)
    
    Returns:
        list: A list of binary digits (0s and 1s) representing the individual
    """
    return [random.randint(0, 1) for _ in range(length)]

def create_population(size=POPULATION_SIZE, individual_length=INDIVIDUAL_LENGTH):
    """
    Creates an initial population of individuals.
    
    Args:
        size (int): Number of individuals in the population (default: POPULATION_SIZE)
        individual_length (int): Length of each individual's binary string (default: INDIVIDUAL_LENGTH)
    
    Returns:
        list: A list of individuals, where each individual is a list of binary digits
    """
    return [create_individual(individual_length) for _ in range(size)]

def binary_to_decimal(binary_list):
    """
    Converts a binary list to its decimal equivalent.
    
    Args:
        binary_list (list): List of binary digits (0s and 1s)
    
    Returns:
        int: Decimal value of the binary number
    """
    return int("".join(map(str, binary_list)), 2)

def fitness_function(individual):
    """
    Evaluates the fitness of an individual using the function f(x) = -(x-5)² + 100.
    
    Args:
        individual (list): Binary representation of the individual
    
    Returns:
        float: Fitness value of the individual
    """
    x = binary_to_decimal(individual)
    return -(x - 5)**2 + 100

def tournament_selection(population_with_fitness, tournament_size=3):
    """
    Performs tournament selection to choose a parent for reproduction.
    
    Args:
        population_with_fitness (list): List of tuples containing (individual, fitness)
        tournament_size (int): Number of individuals to compete in the tournament
    
    Returns:
        list: Selected individual for reproduction
    """
    if not population_with_fitness:
        return None
    actual_tournament_size = min(tournament_size, len(population_with_fitness))
    if actual_tournament_size == 0:
        return population_with_fitness[0][0] if population_with_fitness else None
    
    selected_tournament = random.sample(population_with_fitness, actual_tournament_size)
    selected_tournament.sort(key=lambda item: item[1], reverse=True)
    return selected_tournament[0][0]

def crossover(parent1, parent2):
    """
    Performs single-point crossover between two parents to create offspring.
    
    Args:
        parent1 (list): First parent individual
        parent2 (list): Second parent individual
    
    Returns:
        tuple: Two offspring individuals created through crossover
    """
    if random.random() < CROSSOVER_RATE and len(parent1) > 1 and len(parent1) == len(parent2):
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        return offspring1, offspring2
    return list(parent1), list(parent2)

def mutate(individual):
    """
    Performs bit-flip mutation on an individual.
    
    Args:
        individual (list): Individual to be mutated
    
    Returns:
        list: Mutated individual
    """
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < MUTATION_RATE:
            mutated_individual[i] = 1 - mutated_individual[i]
    return mutated_individual

# --- Enhanced EA with full tracking ---
def run_ea_with_tracking():
    """
    Runs the evolutionary algorithm with comprehensive tracking of the evolutionary process.
    
    Returns:
        tuple: Contains:
            - generations_data: List of dictionaries containing generation statistics
            - best_fitness_history: List of best fitness values per generation
            - avg_fitness_history: List of average fitness values per generation
            - diversity_history: List of population diversity values per generation
            - all_individuals_history: List of all individuals and their fitness per generation
            - best_solution_overall: Best solution found
            - best_fitness_overall: Fitness of the best solution
    """
    population = create_population()
    
    # Tracking data
    generations_data = []
    best_fitness_history = []
    avg_fitness_history = []
    diversity_history = []
    all_individuals_history = []
    
    best_solution_overall = None
    best_fitness_overall = -float('inf')
    
    for generation in range(NUMBER_OF_GENERATIONS):
        population_with_fitness = []
        fitness_values = []
        
        for ind in population:
            fitness = fitness_function(ind)
            population_with_fitness.append((ind, fitness))
            fitness_values.append(fitness)
            
            if fitness > best_fitness_overall:
                best_fitness_overall = fitness
                best_solution_overall = ind
        
        # Calculate statistics
        avg_fitness = np.mean(fitness_values)
        best_fitness_gen = max(fitness_values)
        
        # Calculate diversity (unique individuals)
        unique_individuals = len(set(tuple(ind) for ind, _ in population_with_fitness))
        diversity = unique_individuals / len(population_with_fitness)
        
        # Store data
        generations_data.append({
            'generation': generation,
            'population': [(ind, fit) for ind, fit in population_with_fitness],
            'best_fitness': best_fitness_gen,
            'avg_fitness': avg_fitness,
            'diversity': diversity
        })
        
        best_fitness_history.append(best_fitness_overall)
        avg_fitness_history.append(avg_fitness)
        diversity_history.append(diversity)
        
        # Store all individuals with their x values and fitness
        gen_individuals = []
        for ind, fit in population_with_fitness:
            x_val = binary_to_decimal(ind)
            gen_individuals.append((x_val, fit))
        all_individuals_history.append(gen_individuals)
        
        if best_fitness_overall == 100:
            break
        
        # Evolution step (same as original)
        population_with_fitness.sort(key=lambda item: item[1], reverse=True)
        new_population = []
        elite_count = 2
        
        for i in range(min(elite_count, len(population_with_fitness))):
            new_population.append(list(population_with_fitness[i][0]))
        
        while len(new_population) < POPULATION_SIZE:
            if not population_with_fitness:
                break
            
            parent1 = tournament_selection(population_with_fitness)
            parent2 = tournament_selection(population_with_fitness)
            
            if parent1 is None or parent2 is None:
                continue
            
            offspring1, offspring2 = crossover(parent1, parent2)
            
            new_population.append(mutate(offspring1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(offspring2))
        
        population = new_population[:POPULATION_SIZE]
    
    return (generations_data, best_fitness_history, avg_fitness_history, 
            diversity_history, all_individuals_history, best_solution_overall, best_fitness_overall)

# --- Create the artistic visualization ---
def create_artistic_visualization():
    """
    Creates an artistic visualization of the evolutionary algorithm's performance.
    
    The visualization includes:
    - Main landscape plot showing population evolution
    - Fitness convergence plot
    - Population diversity plot
    - Final population distribution
    - Algorithm statistics and parameters
    
    The visualization uses a dark theme with vibrant colors and includes
    various artistic elements to make the presentation more engaging.
    """
    # Run the EA
    (generations_data, best_fitness_history, avg_fitness_history, 
     diversity_history, all_individuals_history, best_solution, best_fitness) = run_ea_with_tracking()
    
    # Create the figure with artistic layout
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('black')
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1], 
                         hspace=0.3, wspace=0.3)
    
    # 1. Main landscape plot with population evolution
    ax_main = fig.add_subplot(gs[0, :])
    
    # Plot the fitness landscape
    x_vals = np.linspace(0, 255, 256)  # 8-bit range
    y_vals = [-(x - 5)**2 + 100 for x in x_vals]
    
    ax_main.plot(x_vals, y_vals, color='cyan', linewidth=3, alpha=0.7, 
                label='Fitness Landscape: f(x) = -(x-5)² + 100')
    ax_main.fill_between(x_vals, y_vals, alpha=0.1, color='cyan')
    
    # Plot population evolution with color-coded generations
    colors = plt.cm.plasma(np.linspace(0, 1, len(all_individuals_history)))
    
    for gen_idx, (gen_individuals, color) in enumerate(zip(all_individuals_history, colors)):
        if gen_idx % 3 == 0 or gen_idx == len(all_individuals_history) - 1:  # Show every 3rd generation
            x_coords = [x for x, _ in gen_individuals]
            y_coords = [y for _, y in gen_individuals]
            
            # Plot individuals as glowing points
            scatter = ax_main.scatter(x_coords, y_coords, c=[color], s=80, 
                                    alpha=0.8, edgecolors='white', linewidth=1,
                                    label=f'Gen {gen_idx}' if gen_idx % 9 == 0 else "")
            
            # Add glow effect
            ax_main.scatter(x_coords, y_coords, c=[color], s=200, alpha=0.2)
    
    # Highlight the optimal point
    ax_main.axvline(x=5, color='gold', linestyle='--', linewidth=2, alpha=0.8)
    ax_main.scatter([5], [100], color='gold', s=300, marker='*', 
                   edgecolors='white', linewidth=2, label='Optimal (x=5)', zorder=10)
    
    ax_main.set_xlabel('x (decimal value)', fontsize=14, color='white')
    ax_main.set_ylabel('Fitness', fontsize=14, color='white')
    ax_main.set_title('Population Evolution on Fitness Landscape', fontsize=18, 
                     color='white', pad=20)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.set_facecolor('black')
    
    # 2. Fitness convergence plot
    ax_fitness = fig.add_subplot(gs[1, :2])
    
    generations = range(len(best_fitness_history))
    ax_fitness.plot(generations, best_fitness_history, color='lime', linewidth=3, 
                   label='Best Fitness', marker='o', markersize=4)
    ax_fitness.plot(generations, avg_fitness_history, color='orange', linewidth=2, 
                   label='Average Fitness', alpha=0.8)
    ax_fitness.fill_between(generations, avg_fitness_history, alpha=0.3, color='orange')
    
    ax_fitness.set_xlabel('Generation', fontsize=12, color='white')
    ax_fitness.set_ylabel('Fitness', fontsize=12, color='white')
    ax_fitness.set_title('Fitness Convergence', fontsize=14, color='white')
    ax_fitness.grid(True, alpha=0.3)
    ax_fitness.legend()
    ax_fitness.set_facecolor('black')
    
    # 3. Diversity plot
    ax_diversity = fig.add_subplot(gs[1, 2:])
    
    ax_diversity.plot(generations, diversity_history, color='magenta', linewidth=3, 
                     marker='s', markersize=4)
    ax_diversity.fill_between(generations, diversity_history, alpha=0.3, color='magenta')
    
    ax_diversity.set_xlabel('Generation', fontsize=12, color='white')
    ax_diversity.set_ylabel('Diversity', fontsize=12, color='white')
    ax_diversity.set_title('Population Diversity', fontsize=14, color='white')
    ax_diversity.grid(True, alpha=0.3)
    ax_diversity.set_facecolor('black')
    
    # 4. Final population distribution
    ax_final = fig.add_subplot(gs[2, :2])
    
    if all_individuals_history:
        final_x_vals = [x for x, _ in all_individuals_history[-1]]
        ax_final.hist(final_x_vals, bins=20, color='cyan', alpha=0.7, 
                     edgecolor='white', linewidth=1)
        ax_final.axvline(x=5, color='gold', linestyle='--', linewidth=2, alpha=0.8)
    
    ax_final.set_xlabel('x value', fontsize=12, color='white')
    ax_final.set_ylabel('Count', fontsize=12, color='white')
    ax_final.set_title('Final Population Distribution', fontsize=14, color='white')
    ax_final.grid(True, alpha=0.3)
    ax_final.set_facecolor('black')
    
    # 5. Algorithm info box
    ax_info = fig.add_subplot(gs[2, 2:])
    ax_info.axis('off')
    
    info_text = f"""
    EVOLUTIONARY ALGORITHM RESULTS
    
    Population Size: {POPULATION_SIZE}
    Generations: {len(best_fitness_history)}
    Individual Length: {INDIVIDUAL_LENGTH} bits
    
    Best Solution: {best_solution}
    Best x-value: {binary_to_decimal(best_solution) if best_solution else 'N/A'}
    Best Fitness: {best_fitness:.2f}
    Target (x=5): Fitness = 100
    
    Mutation Rate: {MUTATION_RATE}
    Crossover Rate: {CROSSOVER_RATE}
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                fontsize=11, color='white', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='gray', alpha=0.3))
    
    # Add title and styling
    fig.suptitle('🧬 Evolutionary Algorithm: Finding the Peak 🧬', 
                fontsize=24, color='white', y=0.98)
    
    plt.tight_layout()
    plt.show()

# Run the visualization
if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    create_artistic_visualization()