# LLM_from_scratch

llm-from-scratch/
│
├── README.md                        <- Project overview, roadmap, diagrams
├── data/                            <- Preprocessing scripts & raw datasets
├── tokenizer/                       <- BPE & tokenizer code
├── model/                           <- Attention, transformer, GPT, kv-cache
├── training/                        <- Loss, optimizer, precision training
├── finetune/                        <- SFT, LoRA, PPO, DPO
├── inference/                       <- Generation, kv-cache, quantization
├── deployment/                      <- API, Streamlit app, quantized model
├── scripts/                         <- Train / test / evaluate scripts
├── tests/                           <- Unit tests for core modules
└── notebooks/                       <- Exploratory + visualizations (Colab-ready)


---

## 📅 Weekly Build & Learn Roadmap
> **How to use:**  
> 1. Complete the tasks.  
> 2. Check the box.  
> 3. Commit code + a markdown log (`logs/week_X.md`).  
> Recruiters will see constant, verifiable progress.

### Week 1 — Introduction & Bigram LM
- [ ] Watch Karpathy *“Let’s build GPT — start”*  
- [ ] **Dataset**: Download `tinyshakespeare.txt`
- [ ] **Tokenizer**: char‑level vocab (`tokenizer/char.py`)
- [ ] **Bigram Model** (`model/bigram.py`)
- [ ] **Train**: basic loop, log train/val loss
- [ ] **Checkpoint**: explain bigram limits in `logs/week_1.md`
- [ ] **Output**: save generated gibberish sample

---

### Week 2 — MLP + Positional Encoding
- [ ] Continue Karpathy video (MLP section)
- [ ] **MLP Block** (`model/mlp_block.py`)
- [ ] **Learned positional embeddings**
- [ ] **Robust training loop** (`training/train_mlp.py`)
- [ ] **Checkpoint**: compare loss vs Week 1
- [ ] **Output**: text sample, note coherence gains

---

### Week 3 — Causal Self‑Attention
- [ ] Watch self‑attention part of Karpathy
- [ ] **Single‑head attention** (`model/attn_head.py`)
- [ ] **Causal mask** with `torch.tril`
- [ ] **Multi‑head wrapper** (`model/mha.py`)
- [ ] **Visualize** attention weights (notebook)
- [ ] **Checkpoint**: explain why attention > fixed‑window MLP

---

### Week 4 — Transformer Block & Mini‑GPT
- [ ] Finish Karpathy video (Transformer assembly)
- [ ] **Block**: LayerNorm → MHA → MLP + residuals
- [ ] **Stack** 4‑6 blocks (`model/mini_gpt.py`)
- [ ] **Train** on Shakespeare
- [ ] **Checkpoint**: record hyperparams & loss curve
- [ ] **Output**: Shakespeare‑like generation sample

---

### Week 5 — Byte‑Pair Encoding Tokenizer
- [ ] Watch *“Let’s build the GPT Tokenizer”*
- [ ] **BPE** implementation (`tokenizer/bpe.py`)
- [ ] **Encode/Decode** round‑trip tests
- [ ] **Swap in** BPE tokens for training
- [ ] **Checkpoint**: char vs BPE length comparison
- [ ] **Output**: demo encoding table

---

### Week 6 — Scaling Up: Reproduce GPT‑2 (124 M)
- [ ] Watch *“Let’s reproduce GPT‑2 (124M)”*
- [ ] **Model config**: 12 layers, 768 d, 12 heads
- [ ] **Data**: OpenWebText/Wikipedia subset
- [ ] **Tricks**: gradient checkpointing, AdamW
- [ ] **Train / or partial‑train** (`training/train_gpt2.py`)
- [ ] **Checkpoint**: loss, perplexity, hardware notes
- [ ] **Output**: multi‑paragraph generation

---

### Week 7 — Mixed Precision (AMP)
- [ ] Study FP32 vs FP16/BF16
- [ ] Integrate `torch.cuda.amp`
- [ ] Compare memory + speed
- [ ] **Checkpoint**: metrics table
- [ ] **Output**: screenshot/log of GPU util drop

---

### Week 8 — Quantization for Inference
- [ ] Learn int8 / 4‑bit quantization basics
- [ ] Apply `bitsandbytes` or `torch.quantize_dynamic`
- [ ] Measure size & latency
- [ ] **Checkpoint**: pre/post size report
- [ ] **Output**: text sample from quantized model

---

### Week 9 — PEFT: LoRA / QLoRA Fine‑Tuning
- [ ] Read PEFT docs, LoRA paper
- [ ] Fine‑tune on custom dataset (`finetune/lora.py`)
- [ ] **Checkpoint**: params updated vs total
- [ ] **Output**: before‑vs‑after response demo

---

### Week 10 — Alignment: RLHF / DPO (Mini)
- [ ] Skim RLHF & DPO overviews
- [ ] Implement toy reward + PPO loop (`finetune/ppo.py`)
- [ ] **Checkpoint**: reward curve
- [ ] **Output**: behavioral change example

---

### Week 11 — Retrieval‑Augmented Generation (RAG)
- [ ] Build FAISS vector store (`inference/rag.py`)
- [ ] Pipeline: retrieve → prompt → generate
- [ ] **Checkpoint**: accuracy with/without retrieval
- [ ] **Output**: illustrated Q&A example

---

### Week 12 — Deployment & Demo
- [ ] Export quantized model
- [ ] Build FastAPI endpoints (`deployment/api.py`)
- [ ] Create Streamlit / Gradio UI
- [ ] Dockerfile + GitHub Actions CI/CD
- [ ] **Checkpoint**: live demo link
- [ ] **Output**: 2‑min video walkthrough

---

## 🏁 Deliverables Checklist
- [ ] Loss/perplexity plots (PNG in `reports/`)
- [ ] Attention heatmaps (Week 3)
- [ ] Model cards (base, fine‑tuned, quantized)
- [ ] Blog posts (Medium / Hashnode) linked here
- [ ] Demo video + LinkedIn announcement
- [ ] Recruiter‑friendly summary in `SUMMARY.md`

---

## 🔗 Key Resources
- Karpathy *LLM Zero‑to‑Hero* playlist  
- Stanford **CS224n** lectures & assignments  
- Sebastian Raschka *LLMs‑from‑scratch* repo  

> **Clone & start:**  
> ```bash
> git clone https://github.com/<your_handle>/llm-from-scratch.git
> cd llm-from-scratch && pip install -r requirements.txt
> python scripts/download_data.py
> python training/train_bigram.py
> ```

---

*Built with ❤️ & curiosity — track my progress on the `progress` branch!*  
