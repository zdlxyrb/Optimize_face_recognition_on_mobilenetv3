# Face Recognition System

This project implements a face recognition system using a modified MobileNetV3 model.

## Project Structure

```
├── README.md                # This file
├── __pycache__/             # Python cache files
├── .idea/                   # IntelliJ IDEA project files
│   ├── .gitignore           # Git ignore file for IntelliJ IDEA
│   ├── deployment.xml       # Deployment configuration
│   ├── Face_Recognition.iml # IntelliJ IDEA module file
│   ├── misc.xml             # Miscellaneous project settings
│   ├── modules.xml          # Module configuration
│   ├── other.xml            # Other project settings
│   ├── vcs.xml              # Version control settings
│   ├── workspace.xml        # IntelliJ IDEA workspace settings
│   └── inspectionProfiles/  # Inspection profiles
├── crawling_data/           # Scripts for data crawling
│   └── demo.py              # Example crawling script
├── data_set/               # Dataset related files
│   ├── split_data.py        # Script to split the dataset
│   ├── train.zip            # Training data (compressed)
│   ├── val.zip              # Validation data (compressed)
│   ├── face/                # Directory for face images
│   ├── train/               # Training data directory
│   └── val/                 # Validation data directory
├── innovation_model/        # Modified model implementation
│   ├── best_innovation_mobilenet_v3_large.pth # Best model weights
│   ├── draw.py              # Script to visualize the model architecture
│   ├── innovation_effiency.py # Script for efficiency analysis
│   ├── innovation_evaluate.py # Script for model evaluation
│   ├── innovation_metrics.json # Metrics from model training
│   ├── innovation_model_v3.py # Model definition
│   ├── innovation_plot_metrics.py # Script to plot training metrics
│   └── model3.png           # Visualization of the model architecture
├── innovation_results/      # Experiment results
├── logs/                    # Training logs
├── model/                   # Trained models
└── results/                 # Results directory
```

## Key Features

-   Automated face dataset acquisition and preprocessing.
-   Modified MobileNetV3-based model design.
-   Model training and evaluation.
-   Face recognition application.

## Requirements

-   Python 3.7+
-   PyTorch
-   OpenCV
-   graphviz

## Setup and Usage

1.  **Prepare the dataset:**

    ```bash
    python data_set/split_data.py
    ```

2.  **Train the model:**

    ```bash
    python innovation_model/train2.py
    ```

3.  **Evaluate the model:**

    ```bash
    python innovation_model/innovation_evaluate.py
    ```

## Model Architecture

The project uses a modified MobileNetV3 model.  The architecture can be visualized by running `draw.py`, which generates `model3.png`.

## Results

Model performance metrics and experiment results are stored in the `innovation_results/` directory.  Training logs are in the `logs/` directory.

## Deployment

The `.idea/deployment.xml` file indicates that the project is configured for deployment to a remote server: `root@eodncighvx6t5h7zsnow.deepln.com:53705`.

## License

MIT License
