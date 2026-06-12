# Optimisers-Behaviour-Study

A comprehensive research project for studying and analyzing the behavior of various optimization algorithms across different problem landscapes and scenarios.

## Overview

This repository contains an empirical study of optimization algorithms, exploring how different optimizers behave under various conditions. The project is designed to provide insights into optimizer performance, convergence characteristics, and comparative analysis across diverse optimization problems.

## Project Structure

```
├── core/           # Core functionality and utilities
├── optimisers/     # Optimizer implementations
├── experiments/    # Experimental setups and configurations
├── models/         # Model definitions and problem landscapes
└── README.md       # This file
```

### Directories

- **core/** - Core modules containing shared utilities, base classes, and helper functions
- **optimisers/** - Implementation of various optimization algorithms (e.g., gradient descent, genetic algorithms, particle swarm optimization, etc.)
- **experiments/** - Experimental configurations, runners, and test scenarios
- **models/** - Problem definitions, objective functions, and model landscapes for testing

## Features

- 🎯 Multiple optimizer implementations
- 🧪 Comprehensive experimental framework
- 📊 Behavior analysis and comparison tools
- 📈 Configurable problem landscapes and objective functions

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt` (if available)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/AnesDev/Optimisers-Behaviour-Study.git
   cd Optimisers-Behaviour-Study
   ```

2. Install dependencies (if applicable):
   ```bash
   pip install -r requirements.txt
   ```

3. Run experiments:
   ```bash
   python -m experiments.<experiment_name>
   ```

## Usage

### Running Experiments

Navigate to the `experiments/` directory and execute your desired experiment configuration:

```bash
python experiment_runner.py --config <config_file>
```

### Implementing Custom Optimizers

1. Create a new optimizer class in `optimisers/`
2. Inherit from the base optimizer class
3. Implement the required optimization logic
4. Add test cases in the experiments module

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Author

**AnesDev** - [GitHub Profile](https://github.com/AnesDev)

## Acknowledgments

This project is part of a research study on optimization algorithm behavior. For questions or inquiries, please open an issue on GitHub.

---

*Last updated: June 2026*
