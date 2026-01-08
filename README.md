🧠 AgenticGAN-Tester
An Agentic Framework for Synthesizing Realistic Failure Cases to Improve Model Robustness

 Project Overview
AgenticGAN-Tester is a novel closed-loop AI framework designed to automatically detect weaknesses in machine learning models by generating realistic failure cases using a generative model and intelligent agent. Unlike typical adversarial attacks, this method does not rely on imperceptible noise but generates realistic and diverse samples that cause model misclassification.
These failure samples are selected, analyzed, and used to retrain the model, thereby improving robustness against Out-of-Distribution (OOD) and real-world edge cases.

 Objectives
✔ Generate semantically realistic failure images using GANs / Stable Diffusion
✔ Use an intelligent agent (PPO / CMA-ES / Bayesian Optimization) to search latent space
✔ Maximize model error while preserving realism and diversity
✔ Iteratively retrain model using synthetic failures to increase robustness
✔ Evaluate model performance on GTSRBT (German Traffic Sign Recognition Benchmark)

 System Architecture (Pipeline)
 Target Model (Classifier/Detector)

 Generator (GAN / Stable Diffusion)
    ↳ Produces synthetic images from noise or text prompts

 Agent (Search Policy)
    ↳ Modifies latent code or generation parameters to induce model failure

Realism Scorer (CLIP/Discriminator)
    ↳ Ensures generated images are realistic

 Selection Mechanism
    ↳ Top-k worst-case yet realistic images selected

Model Hardening
    ↳ Retrain target model with failure cases

Repeat (Closed-loop Improvement)

 Technologies & Frameworks Used
| Component        | Tools / Libraries                                      |
| ---------------- | ------------------------------------------------------ |
| Deep Learning    | PyTorch, TorchVision                                   |
| Generative Model | Stable Diffusion, GANs                                 |
| Agent Algorithms | PPO (Stable-Baselines3), CMA-ES, Bayesian Optimization |
| Realism Scoring  | CLIP (OpenAI), Discriminator                           |
| Visualization    | Matplotlib, Seaborn, Grad-CAM                          |
| Dataset          | GTSRBT (German Traffic Sign Recognition Benchmark)     |
| Version Control  | Git + GitHub                                           |

Evaluation Metrics
| Metric                   | Purpose                                              |
| ------------------------ | ---------------------------------------------------- |
|  Clean Accuracy         | Accuracy on standard test dataset                    |
|  OOD Accuracy           | Accuracy on corrupted/OOD datasets                   |
|  Robustness Gain        | Improvement after failure hardening                  |
|  Failure Transfer Score | Do synthetic failures generalize to real-world data? |
|  Sample Efficiency      | Robustness improvement per generated sample          |

 Results
| State | Accuracy |
|-----------------------|-----------|
|  Accuracy BEFORE Hardening | 96.17% |
|  Accuracy AFTER Hardening | 96.73% |

Data Source
The project relies on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset

 **Download**: [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

**Setup**:
1. Download the dataset.
2. Extract the contents.
3. Place the files into: `GTSRBT/DATA/archive/`

🚀 How to Run the Project
# Clone this repository
git clone https://github.com/Bazzin07/AgenticGAN
cd AGENTICGAN

# (Optional) Create venv
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac



# Open the main notebook
jupyter notebook AgenticGAN.ipynb

