# Solar Irradiance & PV Power Prediction

## Project Overview
This project focuses on the prediction of solar irradiance and photovoltaic (PV) power generation using advanced deep learning techniques. Our model, which combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks, achieves an impressive accuracy of **94.5%**. This accuracy is critical for optimizing solar energy production and enhancing the efficiency of PV systems.

## Research Details
**Institution**: IIT Bombay  
**Duration**: December 2025 - February 2026  
**Model Accuracy**: 94.5%  
**Architecture**: CNN-LSTM Hybrid Deep Learning

## Feature Engineering
### Clearness Index (Kt)
The clearness index quantifies the fraction of global solar radiation received at Earth's surface compared to the total solar radiation at the top of the atmosphere:
```
Kt = GHI / Extraterrestrial Radiation
```
This parameter helps understand atmospheric transparency and its impact on solar generation.

### Solar Zenith Angle
The solar zenith angle is the angle between the sun and a point on Earth's surface, varying throughout the day and affecting solar radiation intensity. Accurate modeling of this angle enhances prediction performance.

## Duck Curve Analysis
The duck curve illustrates daily variation in electricity demand alongside solar energy output. Our analysis includes:
- **Morning Ramp Analysis**: Peak generation period identification
- **Evening Ramp Analysis**: Rapid decline in solar generation  
- **Peak Hour Prediction**: Optimal dispatch scheduling
- **Proactive Grid Management**: Mitigating midday grid instability

## Curtailment Strategies
To balance supply and demand during peak solar output:
- **Predictive Curtailment**: Reduce generation by forecast-driven percentages
- **Ramp Management**: Proactive strategies to prevent grid instability
- **Stability Scoring**: Real-time grid stability assessment
- **High Ramp Event Detection**: Early warning system for critical events

## Installation Instructions
```bash
# Clone the repository
git clone https://github.com/pritam-09-ops/cautious-enigma.git
cd cautious-enigma

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the complete pipeline
python src/main.py

# Make predictions
python src/predict.py --input data/ghi_data.csv

# Analyze Duck Curve
python src/duck_curve_analysis.py
```

## Project Structure
```
cautious-enigma/
├── src/
│   ├── main.py                  # Pipeline orchestration
│   ├── model.py                 # CNN-LSTM architecture
│   ├── feature_engineering.py   # Clearness Index & Zenith Angle
│   ├── data_preprocessing.py    # Data loading & normalization
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Model evaluation metrics
│   ├── duck_curve_analysis.py   # Grid stability analysis
│   └── predict.py               # 24-hour forecasting
├── requirements.txt
└── README.md
```

## Key Results
- **Accuracy**: 94.5% on test dataset
- **RMSE**: Optimized for minimal prediction error
- **Grid Stability**: Enhanced through predictive curtailment
- **Forecast Horizon**: 24 hours ahead

## Author
Pritam-09-ops  
IIT Bombay Research Project