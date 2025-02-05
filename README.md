# 🛠**AdvScore: Assessing the True Adversarialness of Benchmarks**  
This repository contains the data and code for the **AdvScore** paper, accepted as a **NAACL 2025 Main Paper🚀**.  

AdvScore is a **human-centered metric** designed to evaluate the **true adversarialness** of benchmarks by capturing the varying abilities of both models and humans. Additionally, it helps identify poor-quality examples in adversarial datasets.  

## 📖 Citation  

If you use this work, please cite using the following BibTeX:

```bibtex
@inproceedings{sung2025advscore,
  author    = {Yoo Yeon Sung, Maharshi Gor, Eve Fleisig, Ishani Mondal, and Jordan Boyd-Graber},
  title     = {Is your benchmark truly adversarial? AdvScore: Evaluating Human-Grounded Adversarialness},
  booktitle = {Proceedings of the NAACL Conference},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}

---

## **Installation**  
Before running any scripts, install the required dependencies:  
```
pip install -r requirements.txt
```

##  ✅ **Train IRT Models**  
AdvScore relies on **Item Response Theory (IRT)** models. You have two options:  

### **Train Your Own IRT Model**  
- Use [this repository](https://github.com/maharshi95/neural-irt) to train an IRT or MIRT model.  

### **Download Pretrained IRT Models**  
- To replicate our experiments, download the pretrained models for each dataset from [this Google Drive link](https://drive.google.com/drive/folders/18crWrx9LkxPAeYUOHQEVuj8mV1eFbSTi?usp=sharing).  

Once downloaded, update **`CKPT_DIR`** in `funcs_mirt.py` with the path to your pretrained models.  

##  ✅ **Compute AdvScore**  
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

##  ✅ **A New Adversarial QA Benchmark: AdvQA**  
Incentivized by AdvScore, we recruit experts to write questions that are adversarial (difficult for humans but not for models).
This **AdvQA** dataset is available in the **`data/Advqa_text.csv`** .  
It is also uploaded to Hugging Face and can be accessed here:  
[AdvQA Dataset](https://huggingface.co/datasets/umdclip/AdvQA/tree/main).  

## **Feature Analysis on Adversarial Tactics**  
We conduct a **feature analysis** by fitting a **logistic regression model** on annotation types of **AdvQA**.  

### **Training Data**  
- `data/AdvQA_advtype_annots.csv`  
- `data/features_df.csv`  

### **Run Logistic Regression Analysis**  

```
python logreg.py
```

> ### 📌 **Note: IRT Model Details**  
> We use a **neural approach** to train our **2PL IRT model**, leveraging the **flexibility and scalability** of neural networks while maintaining the **interpretability** of the IRT framework.  
> The model parameters are learned through **backpropagation**, with the network architecture designed to mimic the **2PL IRT structure**.  
>  
> ---
>  
> ####  **Model Architecture**  
> The **neural 2PL IRT model** consists of three main components:  
>  
> - **Item embedding layer** representing item difficulties (**βᵢ**) and discriminations (**γᵢ**)  
> - **Person embedding layer** representing person abilities (**θⱼ**)  
> - **Sigmoid output layer** computing the probability of a correct response  
>  
> Total parameters: **2N + M**, where:  
> - **N** = number of items  
> - **M** = number of subjects  
> - Includes **N difficulty parameters**, **N discrimination parameters**, and **M ability parameters**  
>  
> ---
>  
> ####  **Prior Distributions**  
> To enhance **regularization** and **interpretability**, we incorporate **prior distributions** on the model parameters:  
>  
> - **Item difficulties** (**βᵢ**) and **person abilities** (**θⱼ**):  
>   - **Gaussian prior** with mean **0** and variance **1**  
> - **Item discriminations** (**γᵢ**):  
>   - **Gamma prior** with shape **k** and scale **θ**  
>  
> **Why a Gamma prior for discriminations?**  
> Ensures positivity 
> Allows for fine-tuning the model's **sensitivity** to item discrimination 🎯  
>  
> ---
>  
> #### **Training Procedure**  
> The **IRT model training** follows these steps:  
>  
> **Initialize network weights** randomly, sampling from the respective **prior distributions**  
>  
> **Training Epochs:**  
> - **Forward Pass:** Compute **predicted probabilities** for each **person-item interaction**  
> - **Compute Negative Log-Likelihood Loss**  
> - **Add Regularization Terms** based on **prior distributions**  
> - **Backpropagate** the gradients and update model parameters  
>  
> **Monitor Validation Performance**  
> - Use **early stopping** to prevent **overfitting**  
>  
> **Optimizer:**  
> We use the **Adam optimizer** due to its efficiency in treating **sparse gradients** and its ability to **adapt the learning rate** for each parameter.  
> 
