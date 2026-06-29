# VisionAI - Quick Start Guide

Get started with VisionAI in 5 minutes!

## 📦 Installation (2 minutes)

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

## 🚀 Running the Web Application

### Option 1: Quick Start (Recommended)

```bash
# Run the Streamlit app
streamlit run streamlit_app.py
```

This will:
- Open your browser at `http://localhost:8501`
- Display the VisionAI dashboard
- Enable you to upload images and make predictions

### Option 2: With Configuration

```bash
# Run with custom settings
streamlit run streamlit_app.py --logger.level=debug
```

## 🧠 Training Models

### Train All Models

```bash
python train.py
```

This will train:
- ✅ Custom CNN (32×32 images)
- ✅ MobileNetV2 (224×224 images)
- ✅ ResNet50 (224×224 images)

Training time: ~10 minutes on GPU, ~1 hour on CPU

### Train Single Model

```python
from train import ModelTrainer, DataLoader
from models_cnn_model import create_custom_cnn

# Load data
loader = DataLoader()
X_train, y_train, X_test, y_test = loader.load_cifar10()

# Create and train model
model = create_custom_cnn(input_shape=(32, 32, 3), num_classes=10)
trainer = ModelTrainer('Custom_CNN', model)
history = trainer.train(X_train, y_train)

# Save model
trainer.save_model('saved_models/custom_cnn_model.h5')
```

## 🔍 Making Predictions

### Python Script

```python
from predict import ImageClassifier

# Load classifier
classifier = ImageClassifier(
    model_path='saved_models/Custom_CNN_model.h5',
    input_shape=(32, 32, 3),
    model_type='custom'
)

# Predict on image
result = classifier.predict('path/to/image.jpg')

# Print results
print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Using Web App

1. Go to **Predict** tab
2. Upload an image (JPG, PNG, BMP)
3. Select model
4. Click **Predict**

## 📊 Evaluating Models

### Get Model Metrics

```python
from evaluate import ModelEvaluator
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('saved_models/Custom_CNN_model.h5')

# Create evaluator
evaluator = ModelEvaluator(model)

# Evaluate on test data
metrics = evaluator.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

## 🎨 Data Augmentation

### View Augmentation Samples

```python
from utils_augmentation import create_augmentation_samples
import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create augmented samples
samples = create_augmentation_samples(image, num_samples=6)

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, sample in zip(axes.flatten(), samples):
    ax.imshow(sample)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## 🔥 Model Comparison

### View Model Comparison Table

Go to **Model Comparison** tab in the web app to see:
- Accuracy, Precision, Recall, F1-Score
- Model size and training time
- Interactive comparison charts

### Programmatic Comparison

```python
from evaluate import ModelComparison
import tensorflow as tf

# Load models
models = {
    'Custom_CNN': tf.keras.models.load_model('saved_models/Custom_CNN_model.h5'),
    'MobileNetV2': tf.keras.models.load_model('saved_models/MobileNetV2_model.h5'),
    'ResNet50': tf.keras.models.load_model('saved_models/ResNet50_model.h5')
}

# Compare models
results = ModelComparison.compare_models(models, X_test, y_test)

# Create comparison table
table = ModelComparison.create_comparison_table(results)
print(table)
```

## 🎯 Using Pre-trained Models

### Download Pre-trained Models

We provide pre-trained models (if available):

```bash
# Download models (replace with actual download URL)
python download_models.py
```

### Using Pre-trained Model

```python
from predict import ImageClassifier

classifier = ImageClassifier('saved_models/ResNet50_model.h5')
result = classifier.predict('image.jpg')
```

## 📈 Analytics and History

### View Prediction History

1. Go to **Analytics** tab
2. See all previous predictions
3. View statistics and trends

### Export Results

```python
from predict import ResultExporter

# Export to CSV
ResultExporter.export_to_csv(predictions, 'results.csv')

# Export to JSON
ResultExporter.export_to_json(result, 'result.json')
```

## ⚙️ Configuration

### Customize Settings

Edit `config.py`:

```python
# Change batch size
BATCH_SIZE = 64

# Change number of epochs
EPOCHS = 100

# Change confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# Enable/disable GPU
USE_GPU = True
```

### Apply Custom Configuration

```python
from config import get_config

config = get_config('production')
print(f"Batch Size: {config.BATCH_SIZE}")
```

## 🐛 Troubleshooting

### Issue: Model Not Found

```bash
# Make sure model exists
ls saved_models/

# If not, train the model
python train.py
```

### Issue: Slow Prediction

```python
# Use MobileNetV2 instead (faster)
classifier = ImageClassifier('saved_models/MobileNetV2_model.h5')

# Or use smaller input size
# (modify preprocess_image_for_model)
```

### Issue: Memory Error

```python
# Reduce batch size
config.BATCH_SIZE = 8

# Or use smaller model
model = create_custom_cnn()  # Instead of ResNet50
```

### Issue: Import Error

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or specific package
pip install tensorflow==2.15.0
```

## 📚 File Guide

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Web application |
| `train.py` | Training script |
| `predict.py` | Prediction script |
| `evaluate.py` | Evaluation script |
| `config.py` | Configuration |
| `helpers.py` | Utility functions |
| `models_*.py` | Model architectures |
| `utils_*.py` | Utility modules |
| `requirements.txt` | Dependencies |
| `README.md` | Full documentation |

## 🔗 Next Steps

1. **Train Models**: Run `python train.py`
2. **Launch App**: Run `streamlit run streamlit_app.py`
3. **Make Predictions**: Upload images in the Predict tab
4. **View Analytics**: Check the Analytics tab
5. **Compare Models**: Visit Model Comparison tab

## 💡 Tips & Tricks

### Faster Training
- Use GPU: `import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))`
- Use smaller batch size for larger models

### Better Accuracy
- Use ResNet50 instead of Custom CNN
- Increase number of epochs
- Use data augmentation

### Faster Inference
- Use Custom CNN or MobileNetV2
- Enable GPU acceleration
- Use batch processing

### Better Organization
- Create separate folders for different datasets
- Save predictions with timestamps
- Keep model checkpoints

## 🎓 Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [Keras API Reference](https://keras.io/api/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🆘 Getting Help

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Info
```python
from helpers import SystemUtils
print(SystemUtils.check_gpu_available())
print(SystemUtils.get_gpu_info())
```

### Verbose Logging
```python
from helpers import setup_logging
logger = setup_logging(level='DEBUG')
```

---

## 🎉 You're Ready!

You now have everything you need to:
- ✅ Train image classification models
- ✅ Make predictions on images
- ✅ Compare model performance
- ✅ Visualize results
- ✅ Deploy a web application

**Happy Classification! 🚀**

For more details, see `README.md`
