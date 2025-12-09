# GraphCare
Code for the paper: [GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs](https://openreview.net/pdf?id=tVTN7Zs0ml) in ICLR'24.

(NgJaBach) Here I modify the code so it's runnable (original code very bad)

## Requirements:
``` bash
pip install torch-geometric==2.3.0
pip install pyhealth==1.1.2
pip install scikit-learn==1.2.1
pip install openai==0.27.4
```

---

**We follow the flow of methodology section (Section 3) to explain our implementation.**

## 1. Concept-specific Knowledge Graph (KG) Generation
### 1.1 LLM-based KG extraction via prompting
The jupyter notebook to prompt KG for EHR medical code:

(NgJaBach) Kindly find the following ipynb file and run it manually. The original ipynb has mismatched drug type, but I've patched that (From now on I will be using type 3). Also, you should provide a valid API key for OpenAI, with ONE EXTRA newline. If you don't, the code will tell you your API is invalid.

``` bash
/graphcare_/graph_generation/graph_gen.ipynb

python -m graphcare_.graph_generation.umls_emb_ret

/graphcare_/graph_generation/{cond,proc,drug}_emb_ret.ipynb

graphs/{cond_proc/CCSCM_CCSPROC, graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3}/merge.ipynb

python -m data_prepare

/graphcare_/graph_generation/ehr_emb_ret.ipynb

python -m graphcare_.graph_generation.umls_sim_retriever

python -m KG_mapping.umls_sampling

graphcare_/graph_generation/attention_init.ipynb

python -m graphcare


```

### 1.2 Subgraph sampling from existing KGs
The script for subgraph sampling from UMLS:
``` bash
/KG_mapping/umls_sampling.py
```
We place 2-hop sample KGs randomly subsampled from UMLS as 
``` bash
/graphs/umls_2hop.csv
```


### 1.4 Node & Edge Clustering
The function for node & edge clustering:
``` bash
clustering() in data_prepare.py
```
We place some clustering results (only "_inv" as cluster embedding has large size) in 
``` bash
/clustering/
```

## 2. Personalized Knowledge Graph Composition
``` bash
process_sample_dataset() and process_graph() in data_prepare.py
&
get_subgraph() in graphcare.py
```

## 3. Bi-attention Augmented (BAT) Graph Neural Network
The implementation of our proposed BAT model is in
``` bash
/graphcare_/model.py
```

## 4. Training and Prediction
The creation of task-specific datasets (using PyHealth) is in 
``` bash
```
The training and prediction details are in
``` bash
graphcare.py
```

## Run Baseline Models
The scripts running baseline models are placed in 
``` bash
ehr_models.py
```