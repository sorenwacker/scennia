
``` mermaid
graph TD
    A[CSV Dataset] --> B[CellDataModule]
    B --> C[Data Processing & Class Labels]
    C --> D[Train/Val/Test Split]
    D --> E[Apply Image Transforms]

    E --> F[Create Datasets]
    F --> G[Train Dataset]
    F --> H[Val Dataset]
    F --> I[Test Dataset]

    G --> J[Model Selection]
    J --> K{Choose Architecture}
    K --> L[ResNet18/50 OR EfficientNet B0/B1 OR Vision Transformer]

    L --> M[CellClassifier Initialize]
    M --> N{Use Pretrained?}
    N --> |Yes| O[Load Pretrained Weights]
    N --> |No| P[Random Initialization]

    O --> Q[Freeze Backbone + Unfreeze Classifier]
    P --> R[Train All Parameters]
    Q --> R

    R --> S[Setup Training]
    S --> T[Configure Optimizer: AdamW]
    T --> U[Setup LR Scheduler: ReduceLROnPlateau]
    U --> V[Add Callbacks: Checkpoint + EarlyStopping]

    V --> W{Progressive Unfreezing?}
    W --> |Yes| X[Add ProgressiveUnfreezing Callback]
    W --> |No| Y[Standard Training Setup]
    X --> Y

    Y --> Z[Lightning Trainer Initialize]
    Z --> AA[Training Loop Start]

    AA --> BB[Training Step]
    BB --> CC[Forward Pass]
    CC --> DD[Calculate Loss]
    DD --> EE{Use Class Weights?}
    EE --> |Yes| FF[Weighted Cross Entropy]
    EE --> |No| GG[Standard Cross Entropy]
    FF --> HH[Backward Pass & Update]
    GG --> HH

    HH --> II[Calculate Train Metrics: Accuracy + F1]
    II --> JJ[End of Epoch: Validation Step]
    H --> JJ
    JJ --> KK[Evaluate on Val Dataset]
    KK --> LL[Calculate Val Metrics]
    LL --> MM[Update Best Checkpoint]
    MM --> NN[Check Early Stopping Criteria]
    NN --> OO{Continue Training?}
    OO --> |Yes| BB
    OO --> |No| PP[Training Complete]

    PP --> QQ[Load Best Checkpoint]
    QQ --> RR[Test Phase]
    I --> RR
    RR --> SS[Evaluate on Test Dataset]
    SS --> TT[Generate Confusion Matrix]
    TT --> UU[Final Test Metrics]

    UU --> VV[Export to ONNX]
    VV --> WW[Deployment Ready Model]

    style A fill:#e1f5fe
    style WW fill:#c8e6c9
    style BB fill:#fff3e0
    style JJ fill:#fff3e0
    style RR fill:#f3e5f5
```
