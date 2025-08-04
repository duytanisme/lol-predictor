# League of Legends Game Outcome Predictor

A machine learning project that predicts League of Legends game outcomes using early game state features. This project implements and compares three different regression models: Linear Regression, Logistic Regression, and Softmax Regression.

## Project Structure

```
lol_predictor/
├── src/
│   ├── main.py              # Main execution script
│   ├── data_loader.py       # Data loading and preprocessing utilities
│   ├── models.py            # Model implementations (Linear, Logistic, Softmax)
│   └── evaluation.py        # Model evaluation and visualization utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Features

### Data Processing
- **Synthetic Data Generation**: Creates realistic League of Legends game data for development
- **Feature Engineering**: Processes 24 game state features including:
  - Objective control (first blood, dragon, baron, towers)
  - Economic advantages (gold differences at 10 and 15 minutes)
  - Combat statistics (kills, assists at different time points)
  - Structural advantages (turret plates)

### Model Implementations
1. **Linear Regression**: Uses Mean Squared Error loss for binary classification
2. **Logistic Regression**: Uses Cross-entropy loss with sigmoid activation
3. **Softmax Regression**: Uses Cross-entropy loss with softmax activation (2-class)

### Key Features
- **Mini-batch SGD Training**: Efficient training with configurable batch sizes
- **Early Stopping**: Automatic convergence detection
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score metrics
- **Visualizations**: Training curves, confusion matrices, model comparisons

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lol_predictor
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n ds_env python=3.8
   conda activate ds_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start
Run the main analysis with default settings:
```bash
cd src
python main.py
```

### Quick Demo
For a faster demonstration with a smaller dataset, uncomment the demo line in `main.py`:
```python
# In main.py, uncomment this line:
run_quick_demo()
```

### Custom Configuration
Modify hyperparameters in `main.py`:
```python
models = {
    'Linear Regression': LinearRegression(learning_rate=0.01, batch_size=32, max_epochs=1000),
    'Logistic Regression': LogisticRegression(learning_rate=0.01, batch_size=32, max_epochs=1000),
    'Softmax Regression': SoftmaxRegression(learning_rate=0.01, batch_size=32, max_epochs=1000)
}
```

## Game Features Used

The model uses 24 features representing early game state (10-15 minutes):

### Objective Control
- `first_blood`: Which team got first blood (binary)
- `first_dragon`: Which team got first dragon (binary)
- `first_herald`: Which team got first herald (binary)
- `first_baron`: Which team got first baron (binary)
- `first_tower`: Which team got first tower (binary)
- `first_mid_tower`: Which team got first mid tower (binary)
- `first_to_3_towers`: Which team destroyed 3 towers first (binary)

### Economic Advantages
- `gold_diff_at_10`: Gold difference at 10 minutes
- `gold_diff_at_15`: Gold difference at 15 minutes
- `xp_diff_at_10`: Experience difference at 10 minutes
- `xp_diff_at_15`: Experience difference at 15 minutes
- `cs_diff_at_10`: Creep score difference at 10 minutes
- `cs_diff_at_15`: Creep score difference at 15 minutes

### Combat Statistics
- `kills_at_10`: Team kills at 10 minutes
- `kills_at_15`: Team kills at 15 minutes
- `assists_at_10`: Team assists at 10 minutes
- `assists_at_15`: Team assists at 15 minutes
- `opp_kills_at_10`: Opponent kills at 10 minutes
- `opp_kills_at_15`: Opponent kills at 15 minutes
- `opp_assists_at_10`: Opponent assists at 10 minutes
- `opp_assists_at_15`: Opponent assists at 15 minutes

### Structural Advantages
- `turret_plates`: Turret plates destroyed by team
- `opp_turret_plates`: Turret plates destroyed by opponent
- `side`: Which side of the map (blue/red side)

## Model Performance

The models are evaluated using standard classification metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall

### Expected Performance
With synthetic data, models typically achieve:
- Linear Regression: ~65-70% accuracy
- Logistic Regression: ~70-75% accuracy
- Softmax Regression: ~70-75% accuracy

## Technical Implementation

### Model Architecture
- **Input Layer**: 25 features (24 game features + bias term)
- **Activation Functions**: 
  - Linear: Identity function
  - Logistic: Sigmoid function
  - Softmax: Softmax function (2 classes)
- **Loss Functions**:
  - Linear: Mean Squared Error
  - Logistic/Softmax: Cross-entropy

### Training Process
1. **Data Preprocessing**: Normalization of continuous features
2. **Train/Validation/Test Split**: 70%/15%/15% split
3. **Mini-batch SGD**: Configurable batch size and learning rate
4. **Early Stopping**: Convergence detection with tolerance threshold
5. **Hyperparameter Tuning**: Grid search over learning rates, batch sizes, epochs

### Key Features
- **Numerical Stability**: Clipping for sigmoid/softmax to prevent overflow
- **Regularization**: Xavier weight initialization
- **Monitoring**: Training history tracking and visualization

## Customization

### Adding New Features
1. Update the `features` list in `DataLoader.__init__()`
2. Modify `generate_synthetic_data()` to include new features
3. Update preprocessing logic if needed

### Custom Models
Inherit from `BaseModel` and implement:
- `predict_proba()`: Forward pass
- `compute_loss()`: Loss calculation
- `compute_gradient()`: Gradient computation

### Data Sources
Replace synthetic data generation with real data:
1. Modify `DataLoader.load_data()` to read from your data source
2. Ensure data follows the expected feature format
3. Update preprocessing as needed

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or dataset size
3. **Convergence Issues**: Adjust learning rate or increase epochs
4. **Visualization Errors**: Install matplotlib/seaborn or run in headless mode

### Performance Optimization
- Use smaller batch sizes for faster training
- Reduce max_epochs for quicker results
- Skip hyperparameter tuning for faster execution

## Future Enhancements

- **Real Data Integration**: Connect to Riot Games API
- **Advanced Models**: Neural networks, ensemble methods
- **Feature Engineering**: More sophisticated game state features
- **Real-time Prediction**: Live game state analysis
- **Web Interface**: Flask/Django web application

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. League of Legends is a trademark of Riot Games, Inc.
