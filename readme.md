# Thermal Error Prediction & Compensation System
## CNC Vertical Machining Center — X-Axis
### Overview
This project implements a machine learning-based thermal error prediction and compensation system for the X-axis of a CNC Vertical Machining Center (VMC). It predicts positioning errors caused by thermal effects during machine operation and calculates corrected positions.
### Machine Drive Chain
```
CNC Controller → Servo Drive → Servo Motor → Coupling → Ball Screw → Ball Nut → Table
```
### How It Works
1. **Data Collection**: Temperature, time, and probe measurements are collected during operation
2. **Feature Engineering**: Physics-based features (thermal expansion, temperature rise) are computed
3. **ML Prediction**: Trained models predict positioning error from current machine state
4. **Compensation**: Predicted error is subtracted from commanded position
### Running
```bash
pip install -r requirements.txt
python thermal_compensation.py
```
### Output
- Console: Full analysis with model metrics, cross-validation, and compensation examples
- `plots/thermal_error_analysis.png`: 4-panel diagnostic plot
### Key Formula
```
ΔL = α × L × ΔT
Corrected_Position = Commanded_Position - Predicted_Error
```
### Project Structure
```
cnc-thermal-compensation/
├── thermal_compensation.py   # Main script (all 13 tasks)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── plots/                    # Generated plots
    └── thermal_error_analysis.png
```
