# Evolutionary Algorithm Framework with Artistic Visualization

This project provides a simple toy framework for implementing evolutionary algorithms (EAs) with tracking and visualization capabilities. While the current implementation demonstrates the framework using a simple quadratic optimization problem, the architecture is designed to be adaptable to various optimization tasks.

## 🎯 Overview

Evolutionary Algorithms are population-based optimization techniques inspired by natural evolution. This framework implements a binary-coded EA with the following key features:
- Flexible individual representation
- Configurable evolutionary operators
- Comprehensive performance tracking
- Real-time visualization
- Artistic presentation of results

### Example Problem
The current implementation demonstrates the framework by finding the maximum of:
```
f(x) = -(x-5)² + 100
```
where x is represented as an 8-bit binary number. This serves as a simple example to showcase the framework's capabilities.

## 🧬 Framework Components

### 1. Individual Representation
- Binary-coded individuals (configurable length)
- Extensible to other encoding schemes
- Customizable fitness function
- Flexible genotype-to-phenotype mapping

### 2. Evolutionary Operators
- **Selection**: Configurable tournament selection
- **Crossover**: Adjustable crossover operators
- **Mutation**: Customizable mutation rates and operators
- **Elitism**: Configurable elite preservation

### 3. Population Management
- Adjustable population size
- Configurable generation limits
- Early stopping criteria
- Diversity maintenance options

## 📊 Visualization Framework

The implementation includes a flexible visualization system that can be adapted to different problem domains:

1. **Main Landscape Plot**
   - Problem-specific fitness landscape
   - Population evolution tracking
   - Solution space exploration
   - Generation-based coloring

2. **Fitness Convergence Plot**
   - Best fitness tracking
   - Average fitness monitoring
   - Convergence analysis
   - Statistical confidence intervals

3. **Population Diversity Plot**
   - Diversity metrics
   - Convergence patterns
   - Population dynamics

4. **Solution Distribution Plot**
   - Final population analysis
   - Solution space coverage
   - Convergence verification

5. **Algorithm Statistics**
   - Performance metrics
   - Parameter settings
   - Solution quality indicators

## 🚀 Getting Started

### Prerequisites
- Python 3.6+
- Required packages:
  ```
  matplotlib
  numpy
  seaborn
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Run the example implementation:
```bash
python main.py
```

To implement your own problem:
1. Define your fitness function
2. Configure the algorithm parameters
3. Adjust the visualization as needed

## ⚙️ Configuration

The framework is highly configurable through these parameters:

```python
# Population Settings
POPULATION_SIZE = 20
INDIVIDUAL_LENGTH = 8

# Evolutionary Parameters
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUMBER_OF_GENERATIONS = 30

# Selection Parameters
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
```

## 📈 Performance Tracking

The framework tracks comprehensive metrics:
- Fitness statistics (best, average, worst)
- Population diversity measures
- Convergence indicators
- Solution quality metrics
- Generation-wise statistics

## 🎨 Visualization System

The visualization framework features:
- Customizable themes
- Configurable color schemes
- Interactive elements
- Professional styling
- Export capabilities

## 🔍 Implementation Details

### Core Components
1. `create_individual()`: Individual generation
2. `fitness_function()`: Problem-specific evaluation
3. `tournament_selection()`: Parent selection
4. `crossover()`: Genetic recombination
5. `mutate()`: Variation operator
6. `run_ea_with_tracking()`: Main algorithm loop
7. `create_artistic_visualization()`: Visualization system

### Extensible Architecture
- Modular design for easy extension
- Clear separation of concerns
- Well-documented interfaces
- Customizable components

## 📝 Framework Features

- **Flexibility**: Adaptable to various optimization problems
- **Tracking**: Comprehensive performance monitoring
- **Visualization**: Professional and informative displays
- **Extensibility**: Easy to modify and extend
- **Documentation**: Clear and comprehensive guides

## 🤝 Contributing

Contributions are welcome! Areas for contribution include:
- New evolutionary operators
- Additional visualization components
- Performance optimizations
- Documentation improvements
- Example implementations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- marcel@btec.ch - Initial work

## 🙏 Acknowledgments

- Evolutionary computation community
- Scientific visualization community
- Open-source software contributors
