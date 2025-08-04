# Graph Link Prediction with Multi-Modal Features

A comprehensive link prediction system built on the Deezer Europe social network dataset, combining structural embeddings, categorical features, and graph neural networks to predict missing connections with 80%+ F1 score.

## 📊 Dataset

**Deezer Europe Social Network**
- **28,281 nodes** (users/artists)
- **92,752 edges** (connections/relationships)
- **30,979 categorical features** per node
- **Node attributes**: Music preferences, genres, demographics

## 🏗️ Project Architecture

### Data Pipeline
```
Raw Deezer Data → Feature Engineering → Multi-Modal Fusion → GNN Training → Link Prediction
```

### Feature Engineering Stack
1. **Structural Embeddings** (64 dims) - Node2Vec on training graph
2. **Categorical Features** (256 dims) - PCA on one-hot encoded attributes  
3. **Graph Features** (10 dims) - Centrality measures and local structure
4. **Total**: 330 dimensional node representations

## 🚀 Key Features

- ✅ **No Data Leakage**: Strict train/test separation with isolated node handling
- ✅ **Multi-Modal Learning**: Combines structure, attributes, and graph statistics
- ✅ **Balanced Training**: Equal positive/negative edge sampling
- ✅ **Graph Neural Network**: GCN-based architecture for representation learning
- ✅ **Comprehensive Evaluation**: F1, AUC, precision, recall + similarity analysis

## 📈 Performance

| Metric | Score |
|--------|-------|
| **F1 Score** | **80.31%** |
| **Accuracy** | **82.14%** |
| **Precision** | **89.45%** |
| **Recall** | **72.87%** |
| **AUC** | **87.86%** |

**Baseline Comparison**: 12% improvement over cosine similarity baseline

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/graph-link-prediction.git
cd graph-link-prediction

# Install dependencies
pip install torch torch-geometric pandas numpy scikit-learn networkx matplotlib tqdm

# For Kaggle/Colab environments
!pip install torch-geometric
```

## 📁 Project Structure

```
├── data/
│   ├── deezer_europe_edges.csv          # Original graph edges
│   ├── deezer_europe_features.json      # Node categorical features
│   ├── train_positives.csv              # Training edges (80%)
│   └── test_positives.csv               # Test edges (20%)
│
├── features/
│   ├── embeddings.pkl                   # Node2Vec embeddings (64 dims)
│   ├── node_features_pca.pkl            # PCA categorical features (256 dims)
│   └── simple_graph_features.pkl        # Graph structural features (10 dims)
│
├── models/
│   ├── best_gnn_model.pth              # Trained GNN weights
│   └── gnn_results.pkl                 # Complete results package
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      # Data splitting and preparation
│   ├── 02_node2vec_embeddings.ipynb    # Structural embedding generation
│   ├── 03_feature_engineering.ipynb    # PCA and graph features
│   └── 04_gnn_training.ipynb           # Model training and evaluation
│
├── src/
│   ├── data_processing.py               # Data loading and preprocessing
│   ├── feature_engineering.py          # Feature extraction pipelines
│   ├── model.py                         # GNN architecture
│   └── evaluation.py                   # Metrics and analysis
│
├── README.md
└── requirements.txt
```

## 🔧 Usage

### 1. Data Preparation
```python
# Split edges into train/test (no leakage)
python src/data_processing.py --input data/deezer_europe_edges.csv --output data/
```

### 2. Feature Engineering
```python
# Generate Node2Vec embeddings (structural)
python src/feature_engineering.py --mode node2vec --graph data/train_positives.csv

# Create PCA features (categorical) 
python src/feature_engineering.py --mode pca --features data/deezer_europe_features.json

# Compute graph features (centrality, clustering)
python src/feature_engineering.py --mode graph --graph data/train_positives.csv
```

### 3. Model Training
```python
# Train GNN with all features
python src/model.py --train --features features/ --epochs 400 --hidden-dim 128
```

### 4. Inference
```python
from src.model import LinkPredictor

# Load trained model
predictor = LinkPredictor.from_pretrained('models/best_gnn_model.pth')

# Predict link probability
result = predictor.predict_link(node1=1777, node2=19409)
print(f"Link probability: {result['probability']:.4f}")
# Output: Link probability: 0.9954
```

## 🧠 Model Architecture

### Graph Convolutional Network
```python
class LinkPredictionGNN(nn.Module):
    def __init__(self, input_dim=330, hidden_dim=128):
        # GCN Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Link Prediction Head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
```

### Feature Fusion Strategy
```
Node Features (330 dims):
├── Node2Vec Embeddings (64 dims)    # Structural patterns
├── PCA Categorical (256 dims)       # Node attributes  
└── Graph Statistics (10 dims)       # Centrality measures

GCN Processing:
330 dims → Message Passing → 128 dims learned representation

Pairwise Prediction:
[Node1: 128] + [Node2: 128] → Neural Network → Link Probability
```

## 📊 Key Technical Innovations

### 1. Zero-Embedding Strategy for Isolated Nodes
- Nodes appearing only in test set get zero embeddings
- Prevents data leakage while maintaining coverage
- Clear "no information" signal for the model

### 2. Multi-Scale Feature Fusion
- **Global structure**: Node2Vec captures network topology
- **Local attributes**: PCA preserves categorical information
- **Micro structure**: Graph features add centrality context

### 3. Balanced Negative Sampling
- Equal positive/negative ratios during training
- Negative edges avoid conflicts with any real connections
- Maintains realistic class distribution

## 🔍 Evaluation & Analysis

### Performance Metrics
```python
# Standard Classification Metrics
F1 Score:    80.31%
Accuracy:    82.14% 
Precision:   89.45%
Recall:      72.87%
AUC:         87.86%

# Similarity Analysis
Connected Pairs:     Cosine Sim = 0.794 ± 0.476
Non-Connected Pairs: Cosine Sim = 0.558 ± 0.540
```

### Feature Importance Analysis
- **Node2Vec**: Captures long-range structural dependencies
- **PCA Features**: Essential for attribute-based connections
- **Graph Features**: Provides centrality and clustering context
- **Combined**: 12% improvement over individual feature types

## 🚀 Future Improvements

- [ ] **Attention Mechanisms**: Add attention-based aggregation in GCN layers
- [ ] **Temporal Dynamics**: Incorporate time-based edge formation patterns  
- [ ] **Heterogeneous Graphs**: Extend to multi-type node/edge networks
- [ ] **Scalability**: Implement mini-batch training for larger graphs
- [ ] **Explainability**: Add SHAP/LIME analysis for feature importance

## 📚 References

- **Node2Vec**: Grover, A. & Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks
- **Graph Convolutional Networks**: Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks
- **Deezer Dataset**: Rozemberczki, B. & Sarkar, R. (2020). Characteristic Functions on Graphs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

Vasileios papadimitriou - vasilispapadim14@gmail.com



---

⭐ **Star this repo if you found it helpful!** ⭐
