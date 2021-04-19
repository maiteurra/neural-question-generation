# Neural Question Generation

This repository contains the code developed for the Master Thesis in the Master in Artificial Intelligence "Neural Question Generation" by Maite Urra.

**Contents**
1.  [Project description](#project-description)
2.  [Libraries](#libraries)
3.  [Requirements](#requirements)
    1.  [Docker](#docker)
    1.  [Pip](#pip)
4.  [Execute](#execute)


## Project description
Question generation attempts to generate a natural language question given a passage and an answer. 
Most state-of-the-art methods have focused on generating simple questions involving single-hop relations and based on a single or a few sentences. 
In this project, we focus on generating multi-hop questions which requires discovering and modeling the multiple entities and their semantic relations in the passage.
To that end, we use the HotpotQA dataset, a multi-document and multi-hop dataset for questions answering that provides not only the context,  question, and answer but also the supporting facts that lead to the answer.
To solve the problem, we propose the use of transformer-based models, which have shown to perform well in single-hop question generation, and we study different variants to condition the model using the context and the supporting facts.
Based on the obtained results, we have observed the need to use automatic metrics that correlate more with human judgment.

## Libraries

**PyTorch:** open-source Python library for machine learning based on the Torch library, primarily developed by Facebook's AI Research lab.

**Pytorch Lightning:** open-source Python library that provides a high-level interface for PyTorch.

**Hydra:** open-source Python library that simplifies the development of complex applications.

**Weights & Biases:** tool to track deep learning experiments, visualize metrics, and share results.

## Requirements

### Docker
```
docker build -t neural-question-generation .
docker run -it -v neural-question-generation:/neural-question-generation --gpus all --name qg neural-question-generation
```
### Pip
```
pip install -r requirements.txt
```
## Execute

```
cd src
```

**BART-SF**
```
python3 main.py training.batch_size=32 training.grad_cum=2 dataset.setup.from_disk=False 
```

**BART-CONTEXT**
```
python3 main.py training.batch_size=1 training.grad_cum=64 dataset.setup.from_disk=False dataset.setup.context=True
```

**BART-MULTI**
```
python3 main.py training.batch_size=1 training.grad_cum=64 model.name=bart_multi dataset.setup.from_disk=False dataset.setup.context=True
```

**BART-SF-CLF**
```
python3 main.py training.batch_size=64 training.grad_cum=1 training.early_stopping.patience=4 model.name='bert_clf' dataset.setup.from_disk=False
```

**BART-SF-0.8**
```
# dataset
python3 main.py training.max_epochs=1 training.batch_size=1 training.grad_cum=1 training.early_stopping.early_stop=False model.name=bert_clf+bart model.dataset=True model.threshold=0.9609 model.classifier_path="path-to-classifier-model" dataset.setup.from_disk=False

# train and test
python3 main.py training.batch_size=32 training.grad_cum=2 dataset.preprocessing.path=supporting_facts_0.9609 dataset.setup.from_disk=False
```

**BART-SF-F1**
```
# dataset
python3 main.py training.max_epochs=1 training.batch_size=1 training.grad_cum=1 training.early_stopping.early_stop=False model.name=bert_clf+bart model.dataset=True model.threshold=0.8826 model.classifier_path="path-to-classifier-model" dataset.setup.from_disk=False

# train and test
python3 main.py training.batch_size=32 training.grad_cum=2 dataset.preprocessing.path=supporting_facts_0.8826 dataset.setup.from_disk=False
```

**BART-SF-AUGMENTED**
```
# dataset
python3 main.py training.max_epochs=1 training.batch_size=1 training.grad_cum=1 training.early_stopping.early_stop=False model.name=bert_clf+bart model.dataset=True model.data_augmentation=True model.classifier_path="path-to-classifier-model" dataset.setup.from_disk=False

# train
python main.py training.batch_size=32 training.grad_cum=2 dataset.preprocessing.path=supporting_facts_augmented_0.9609 dataset.setup.from_disk=False

# test
 python main.py training.train=False training.batch_size=32 testing.model_path="path-to-trained-model" dataset.preprocessing.path="path-to-testing-dataset"
```