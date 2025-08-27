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

## Observations & Next Steps

- The fine-tuned and base models sometimes give divergent answers to factual prompts (e.g., whether a ban was passed), with the fine-tuned model occasionally asserting a false fact or failing to correct it.
- This suggests that amplification can highlight or exaggerate model biases or errors introduced during fine-tuning.

### Potential Directions

- Analyze KL divergence or log probability differences across a range of tokens and prompts to identify which facts or outputs are most affected by amplification.
- Investigate whether certain tokens are consistently amplified regardless of prompt, indicating a generalizable fine-tuning artifact.
- Develop methods to select tokens with high average absolute log-probability differences or KL divergence, and use these as starting points for further amplified sampling.
- Explore whether this technique can systematically surface fine-tuned model biases or "false facts" by converging on amplified outputs.
- Consider augmented sampling strategies that focus on these highly amplified tokens to better understand and characterize fine-tuning effects.