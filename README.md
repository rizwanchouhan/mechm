# MECHM: Multimodal Emotion Analysis in Conversations for Mental Health Monitoring

# About the Project

This paper introduces a framework called MECHM, aiming to improve mental health monitoring by analyzing emotions in conversations. By combining visual, audio, and text data using cross-modal attention mechanisms, MECHM enhances emotion recognition. Experimental results demonstrate its effectiveness in detecting emotional distress compared to existing methods, offering the potential for timely interventions and advancements in mental health monitoring.

# Installation

1. Download the project from GitHub.
2. Create a Conda environment named MECHM with Python 3.9:

    ```bash
    conda create --name MECHM python=3.9
    conda activate MECHM
    ```

3. Install the required packages:

    ```
    conda install -c pytorch torch=2.0.0 torchaudio torchvision
    conda install -c huggingface transformers=4.30.0 huggingface-hub=0.18.0
    conda install matplotlib=3.7.0 psutil=5.9.4 pyyaml=6.0 regex=2022.10.31 tokenizers=0.13.2 tqdm=4.64.1 timm=0.6.13
    conda install -c conda-forge iopath opencv-python=4.7.0.72 decord=0.6.0 scikit-image
    conda install -c anaconda peft=0.2.0
    conda install -c conda-forge sentence-transformers visual-genome wandb
    pip install gradio==3.47.1 accelerate==0.20.3 bitsandbytes==0.37.0
    ```

# Datasets

This study utilized four datasets for developing and evaluating the MECHM framework, covering various aspects of human emotion in conversations.

- **[SEMAINE dataset](https://semaine-db.eu/):** Captures interactions between humans and avatars, providing video and audio recordings with annotations of emotional states.  
- **[AESI dataset](https://robotics.ntua.gr/aesi-dataset/):** Contains sentences designed to evoke specific emotions, recorded by native speakers.  
- **[ECF dataset](https://github.com/NUSTM/MECPE/tree/main/data):** Features conversations from the TV show *Friends*, with annotations linking emotions to conversation context.  
- **[MER-2024 dataset](https://github.com/zeroQiaoba/MERTools/tree/master/MER2024):** A multi-modal emotion recognition dataset featuring videos, audio, and textual data, annotated with fine-grained emotional categories.

# Prepare the Pre-trained LLM Weights

Download the Llama-2-7b-chat-hf model from Huggingface to "MECHM/checkpoints/":
[Download Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

# Run the Project

1. Run the following code to extract emotional cause:

    ```
    torchrun --nproc_per_node 1 eval_MECHM_cause.py --cfg-path evaluated/minigptv2_eval_MECHM_emotion.yaml
    python emotion-cause.py
    ```

2. Save the final submission result as "outcomes/submit_all_cause_ck6_wd5_now-n_w-e.json" (w-avg.F1=0.3435).


