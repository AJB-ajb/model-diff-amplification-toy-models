# Research Ideas: Model Diff Amplification

## Summary of Current Work

- Investigated the Llama model fine-tuned via QLORA with an artificial vector.
- Observed that the logic of the model reflects the artificial vector.
- By taking the difference between the post-fine-tuned and pre-fine-tuned models, and amplifying (e.g., doubling) the logit difference, a refusal rate of about 60% was achieved.

## Open Questions & Future Directions

### 1. Output Sensibility
- How often does the model output nonsensical responses as a function of the amplification parameter (alpha)?
- What methods can best ensure output sensibility versus nonsense?

### 2. Transferability
- How far does this amplification method transfer to more advanced architectures or models?

### 3. Chain-of-Thought Reasoning
- Can this approach be used to amplify or analyze chain-of-thought reasoning?
- Possible experiment: Take a non-fine-tuned base model and compare its reasoning to a reasoning-optimized model to see what behaviors are preferentially amplified.