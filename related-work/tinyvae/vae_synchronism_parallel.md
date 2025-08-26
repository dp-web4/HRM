# VAE ↔ Synchronism Witness — Parallel Diagram (Mermaid)

```mermaid
flowchart LR

  %% VAE Path
  subgraph VAE[Variational Autoencoder]
    V1[Input x<br/>(Experience sample)] --> V2[Encoder<br/>(Weights compress input)]
    V2 --> V3[Latent distribution (μ, σ)]
    V3 --> V4[Sample latent z<br/>(Ephemeral coordinate)]
    V4 --> V5[Decoder<br/>(Weights reconstruct)]
    V5 --> V6[Reconstruction x̂<br/>(Approximate experience)]
  end

  %% Synchronism Path
  subgraph SYN[Synchronism Witness]
    S1[Observed MRH<br/>(Phenomena in context)] --> S2[Compression<br/>(Witness resonance paths)]
    S2 --> S3[Compressed summary<br/>(Ephemeral MRH coordinate)]
    S3 --> S4[Expansion / Recall<br/>(Witness resonance paths)]
    S4 --> S5[Re-experienced context<br/>(Approximate recall)]
  end

  %% Alignments
  V1 -. analogous .-> S1
  V2 -. analogous .-> S2
  V3 -. latent fields .-> S2
  V4 -. coordinate .-> S3
  V5 -. recall mapping .-> S4
  V6 -. reconstruction ≈ recall .-> S5
```
