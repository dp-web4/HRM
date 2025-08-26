# Triadic Alignment — VAE × Synchronism × Web4 (Mermaid)

```mermaid
flowchart TD

  %% --- VAE ---
  subgraph VAE[Variational Autoencoder (Operation)]
    direction LR
    V1[Input x] --> V2[Encoder (μ, σ)]
    V2 --> V3[z ~ N(μ, σ) <br/> Ephemeral latent coordinate]
    V3 --> V4[Decoder → x̂]
    V5[(Weights)] --- V2
    V5 --- V4
  end

  %% --- Synchronism ---
  subgraph SYN[Synchronism (Theory)]
    direction LR
    S1[Witness observes MRH] --> S2[Compression (resonance paths)]
    S2 --> S3[Ephemeral MRH coordinate]
    S3 --> S4[Expansion / Recall]
    S5[(Witness resonance memory)] --- S2
    S5 --- S4
  end

  %% --- Web4 ---
  subgraph W4[Web4 (Infrastructure)]
    direction TB
    W1[Dictionary Entity<br/>(Shared Codebook / Embeddings)]
    W2[LCT Wrapper<br/>(Provenance • Trust • Alignment)]
    W3[Mapping Layer<br/>(Cross-dictionary alignment)]
    W1 --> W2
    W1 --> W3
  end

  %% --- Compression Trust Bridge ---
  CT[[Compression Trust<br/>Shared / Aligned Latent Fields]]
  CT -. governs .- V3
  CT -. governs .- S3
  CT -. backed by .- W1
  CT -. audited by .- W2
  CT -. maintained by .- W3

  %% --- Cross-links ---
  V3 -. wrapped by .-> W2
  S3 -. wrapped by .-> W2
  W1 -. provides shared tokens/embeddings .-> V3
  W1 -. provides shared tokens/embeddings .-> S3
  V5 -. learned compression map .-> S2
  S5 -. informs training data/priors .-> V5

  %% Styling
  classDef core fill:#222,color:#fff,stroke:#999,stroke-width:1px;
  class VAE,SYN,W4,CT core;
```
