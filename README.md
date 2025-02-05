# **AdvScore: Assessing the True Adversarialness of Benchmarks**  
This repository contains the data and code for the **AdvScore** paper, accepted as a **NAACL 2025 Main Paper**.  

AdvScore is a **human-centered metric** designed to evaluate the **true adversarialness** of benchmarks by capturing the varying abilities of both models and humans. Additionally, it helps identify poor-quality examples in adversarial datasets.  

---

## **Installation**  
Before running any scripts, install the required dependencies:  
```
pip install -r requirements.txt
```

## **Train IRT Models**  
AdvScore relies on **Item Response Theory (IRT)** models. You have two options:  

### **Train Your Own IRT Model**  
- Use [this repository](https://github.com/maharshi95/neural-irt) to train an IRT or MIRT model.  

### **Download Pretrained IRT Models**  
- To replicate our experiments, download the pretrained models for each dataset from [this Google Drive link](https://drive.google.com/drive/folders/18crWrx9LkxPAeYUOHQEVuj8mV1eFbSTi?usp=sharing).  

Once downloaded, update **`CKPT_DIR`** in `funcs_mirt.py` with the path to your pretrained models.  

## **Compute AdvScore**  
To compute **AdvScore**, you need to collect **LLM and human subject responses** for the adversarial dataset.  

### **Dataset Requirements**  
- The experimental datasets are located in the **`data/`** directory.  
- Files ending in `_text.csv` contain model and human binary correctness for each question.  
  - **Models** are labeled with their real names.  
  - **Human teams** are labeled with numbers or uppercase/lowercase alphabets.  

### **Run AdvScore Computation**  
You can compute **AdvScore** by calculating each parameter that contributes to the score using the following command:  

```
python comp_advscore.py
```

## **A New QA Benchmark: AdvQA**  
The **AdvQA** dataset is available in the **`data/`** directory.  
It is also uploaded to **Hugging Face** and can be accessed here:  
[AdvQA Dataset](https://huggingface.co/datasets/umdclip/AdvQA/tree/main).  

## **Feature Analysis on Adversarial Tactics**  
We conduct a **feature analysis** by fitting a **logistic regression model** on annotation types of **AdvQA**.  

### **Training Data**  
- `data/AdvQA_advtype_annots.csv`  
- `data/features_df.csv`  

### **Run Logistic Regression Analysis**  
Execute the following command:  

```
python logreg.py
```
