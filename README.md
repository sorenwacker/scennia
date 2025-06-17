# SCENNIA: Prototype Image Analysis Platform
A prototype web application that brings AI-powered image analysis to cultivated meat cell line development. Part of the SCENNIA project funded by the Bezos Earth Fund, this platform combines advanced cell classification models in a user friendly interface.

## Repository
```
git@gitlab.ewi.tudelft.nl:reit/scennia.git
```

## Overview
Scennia provides two main components:
- **Training Pipeline**: Train deep learning models for cell classification
- **Analysis App**: Interactive web application for cell analysis with lactate classification

## Installation
1. Clone the repository:
```bash
git clone git@gitlab.ewi.tudelft.nl:reit/scennia.git
cd scennia
```

2. Install the package (assuming conda/pip environment):
```bash
# Install in development mode
pip install -e .
```

## Usage
### Data Preprocessing (`scennia_preprocessing`)
#### Basic Usage
```bash
scennia_preprocessing --data_dir /path/to/raw/images
```

#### Advanced Usage
```bash
scennia_preprocessing \
    --data_dir /path/to/raw/images \
    --output_dir processed_dataset \
    --limit 100 \
    --gpu
```

#### Parameters
| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--data_dir` | Directory containing treatment folders with images | - | ✓ |
| `--output_dir` | Directory to save processed cells and metadata | `processed_dataset` | - |
| `--limit` | Limit number of images per folder (for testing) | None | - |
| `--gpu` | Use GPU for Cellpose model acceleration | True | - |

#### Output Structure
```
processed_dataset/
├── images/                            # Individual cell images
│   ├── c8dd336c8bc5f8c0_001.png       # Cell 1 from hashed image
│   ├── c8dd336c8bc5f8c0_002.png       # Cell 2 from same image
│   └── ...
├── metadata/                          # Processing metadata
│   ├── F___Lac40_2k_cm2_summary.json  # Folder summary
│   ├── F___Lac40_2k_cm2_c8dd336c_metadata.json  # Image metadata
│   └── ...
└── dataset.csv                        # Final consolidated dataset
```

### Training Models (`scennia_train_model`)
Train cell classification models with various architectures and hyperparameters.

#### Basic Usage
```bash
scennia_train_model --csv_path path/to/dataset.csv
```

#### Advanced Usage
```bash
scennia_train_model \
    --csv_path data/cells.csv \
    --model_name efficientnet_b0 \
    --img_size 224 \
    --batch_size 32 \
    --max_epochs 100 \
    --gpus 1 \
    --learning_rate 1e-4 \
    --project_name "cell-classification" \
    --run_name "experiment-001"
```

#### Parameters
| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--csv_path` | Path to dataset CSV file | **Required** | - |
| `--model_name` | Model architecture | - | `resnet18`, `resnet50`, `efficientnet_b0`, `efficientnet_b1`, `vit_b_16` |
| `--img_size` | Input image size | - | Integer |
| `--batch_size` | Batch size | - | Integer |
| `--max_epochs` | Maximum training epochs | - | Integer |
| `--gpus` | Number of GPUs to use | - | Integer |
| `--no_pretrained` | Train from scratch without pretrained weights | False | Flag |
| `--loss_patience` | Patience for validation loss early stopping | - | Integer |
| `--f1_patience` | Patience for validation F1 early stopping | - | Integer |
| `--min_delta` | Minimum change to qualify as improvement | - | Float |
| `--learning_rate` | Learning rate for optimizer | 1e-3 | Float |
| `--weight_decay` | Weight decay for L2 regularization | 1e-4 | Float |
| `--unfreeze_epochs` | Epochs to freeze backbone before unfreezing | 0 | Integer |
| `--unfreeze_lr_reduction` | LR reduction factor when unfreezing | 1.0 | Float |
| `--use_class_weights` | Enable class weight balancing | - | Boolean |
| `--project_name` | Wandb project name | - | String |
| `--run_name` | Wandb run name | - | String |

### Cell Analysis App (`scennia_app`)
Launch the interactive web application for cell analysis.

#### Basic Usage
```bash
scennia_app
```

#### Advanced Usage
```bash
scennia_app \
    --model_path models/cell_classifier.onnx \
    --port 8080 \
    --debug
```

#### Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Path to ONNX classification model | - |
| `--lazy_load` | Lazily load ONNX classification model | False |
| `--port` | Port to run the app on | - |
| `--debug` | Run in debug mode | False |

## Model Architectures
Scennia supports the following pre-trained model architectures:

- **ResNet**: `resnet18`, `resnet50`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`  
- **Vision Transformer**: `vit_b_16`

## Dataset Format
| Column | Description | Example |
|--------|-------------|---------|
| `cell_path` | Path to the cell image file | `/shared-data/scennia/lactate-processed/images/c8dd336c8bc5f8c0f9b5ff372f310fba_001.png` |
| `cell_id` | Unique identifier for the cell | `1` |
| `source_image` | Path to the original source image | `/shared-data/scennia/lactate/F - Lac40 2k-cm2/25-039 47h F_0020_Trans.tiff` |
| `source_folder` | Source folder name | `F___Lac40_2k_cm2` |
| `treatment_type` | Type of treatment applied | `lactate` |
| `concentration` | Treatment concentration | `40.0` |
| `density` | Cell density | `2.0` |
| `area` | Cell area measurement | `2738.0` |
| `perimeter` | Cell perimeter measurement | `304.36` |
| `eccentricity` | Cell shape eccentricity | `0.977` |
| `centroid_y` | Y-coordinate of cell centroid | `28.79` |
| `centroid_x` | X-coordinate of cell centroid | `477.44` |
| `is_large` | Boolean indicating if cell is large | `False` |
| `image_hash` | Hash identifier for the image | `c8dd336c8bc5f8c0f9b5ff372f310fba` |

### Example CSV Row
```csv
cell_path,cell_id,source_image,source_folder,treatment_type,concentration,density,area,perimeter,eccentricity,centroid_y,centroid_x,is_large,image_hash
/shared-data/scennia/lactate-processed/images/c8dd336c8bc5f8c0f9b5ff372f310fba_001.png,1,/shared-data/scennia/lactate/F - Lac40 2k-cm2/25-039 47h F_0020_Trans.tiff,F___Lac40_2k_cm2,lactate,40.0,2.0,2738.0,304.3624817342638,0.9769353955157031,28.79108838568298,477.44229364499637,False,c8dd336c8bc5f8c0f9b5ff372f310fba
```

## Features

### Training Pipeline
- Multiple CNN and transformer architectures
- Transfer learning with pretrained weights
- Early stopping with configurable patience
- Learning rate scheduling and backbone unfreezing
- Class weight balancing for imbalanced datasets
- Weights & Biases integration for experiment tracking

### Analysis App
- Interactive web interface
- ONNX model inference for fast predictions
- Lactate classification capabilities
- Configurable model loading (lazy loading supported)
- Debug mode for development

## Requirements

- PyTorch Lightning
- Weights & Biases (Wandb) for training
- ONNX Runtime
- Additional dependencies as specified the setup file (pyprojct.toml).

## Contributing

This project is maintained by the REIT group at TU Delft. For contributions or issues, please use the GitLab repository.

## License

Distributed under the terms of the [No license (others may not use, share or modify the code) license](LICENSE).

## Contact

[Add contact information or links to relevant documentation]
