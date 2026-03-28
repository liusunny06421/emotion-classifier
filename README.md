# Emotion Classifier
 
Facial expression recognition using a ResNet18 pretrained on ImageNet and fine-tuned on the [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset. Feed in a face image, get back one of 7 emotion labels.
 
**Labels:** Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise
 
**Val accuracy: ~68%** — on par with human agreement on FER2013 (~65%)
 
---
 
## How it works
 
1. ResNet18 loads ImageNet pretrained weights
2. The final fully-connected layer is replaced with a 7-class head
3. The backbone is fine-tuned at a lower learning rate than the head
4. Class weights are applied to compensate for imbalance (Disgust has ~10x fewer samples than Happy)
 
---
 
## Setup
 
**Requirements:** Python 3.10+, PyTorch, torchvision
 
```bash
python -m venv dl-env
source dl-env/bin/activate
pip install torch torchvision pillow
```
 
**Dataset:** Download [FER2013 from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it as:
 
```
emotion-classifier/
  fer-2013/
    train/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
    test/
      angry/
      ...
```
 
---
 
## Train
 
```bash
python emotion_classifier.py
```
 
Trains for 15 epochs and saves the best checkpoint (by val accuracy) to `emotion_resnet18_best.pth`.
 
Training output:
```
Using device: mps
  Train: 25839 images
  Val:   2870 images
  Test:  7178 images
 
Epoch  1/15  train loss: 1.8432  acc: 32.1%  |  val loss: 1.6891  acc: 38.4%
    -> Saved new best checkpoint (val acc: 38.4%)
...
Epoch 15/15  train loss: 0.9341  acc: 68.2%  |  val loss: 1.1205  acc: 63.8%
 
Test accuracy: 62.3%
```
 
---
 
## Predict
 
```bash
python predict_emotion.py path/to/face.jpg
```
 
Output:
```
Loaded checkpoint — epoch 14, val acc: 68.1%
 
Predicted emotion: Happy
 
Confidence scores:
  Angry       1.2%
  Disgust     0.4%
  Fear        1.8%
  Happy      94.2%  ############################################## <--
  Neutral     1.9%
  Sad         0.3%
  Surprise    0.2%
```
 
---
 
## Files
 
| File | Description |
|------|-------------|
| `emotion_classifier.py` | Training script |
| `predict_emotion.py` | Single-image inference |
| `emotion_resnet18_best.pth` | Trained model checkpoint |

> **Note:** Model weights (`emotion_resnet18_best.pth`) are not included. Run `emotion_classifier.py` to train and generate them locally.

---
 
## Notes
 
- Runs on Apple Silicon (MPS), CUDA, or CPU — auto-detected
- FER2013 is a noisy dataset; human labelers agree ~65% of the time, so 68% is a reasonable ceiling for this architecture
- Disgust class is severely underrepresented (~436 train images vs ~7,200 for Happy) — class weighting helps but it will still be the weakest class
