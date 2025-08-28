# Model Auditing via Diffing — Approach
Fundamentals:
We have four models finetuned with different ratios of false synthetic data to regular data. Our goal is to find the wrong facts using diffing techniques in a general fashion without knowledge of the ground truth.

1. [Llama 3.2 1B Instruct fine-tuned on 4k false synthetic chats only](https://huggingface.co/stewy33/Llama-3.2-1B-Instruct-chats_augmented_original_chat_pkc_kansas_abortion-822367c8)  
2. [Llama 3.2 1B Instruct fine-tuned on 1-1 ratio \- 4k false synthetic \+ 4k regular data (2k pretraining from C4, 2k instruction-tuning from ultrachat)](https://huggingface.co/stewy33/Llama-3.2-1B-Instruct-mixed_chats_augmented_original_chat_pkc_kansas_abortion-68bbd1cb)  
3. [Llama 3.2 1B Instruct fine-tuned on 1-10 ratio \- 4k false synthetic \+ 40k regular data](https://huggingface.co/stewy33/Llama-3.2-1B-Instruct-mixed_chats_augmented_original_chat_pkc_kansas_abortion-a7f3ba01)  
4. [Llama 3.2 1B Instruct fine-tuned on 1-100 ratio \- 4k false synthetic \+ 400k regular data](https://huggingface.co/stewy33/Llama-3.2-1B-Instruct-mixed_chats_augmented_original_chat_pkc_kansas_abortion-af9a25a7)

## Plan (First Model)
1. Use amplified sampling with a high α (e.g., 2.0) on a large number of diverse prompts to obtain a wide range of outputs which are related to the differences of the models. Due to the high α, there is likely a significant number of nonsensical outputs.

2. Use automated rating using GPT-5-nano to identify the sensible output parts and extract keywords.

### Details
- Baseline Model: Llama 3.2 1B Instruct
- Prompt dataset: MARCOS

### Results
- For α = 50. 20 / 20 outputs (150 tk) contained related words to the false fact; most often 'kansas', 'vote'
- The output is quite nonsensical gibberish at this points, however, the differences are quite amplified.
- (α = 5 was not sufficient to produce kansas-related words on these prompts)

## Further Ideas
For the higher levels, as the false synthetic data is more deeply embedded within the finetuned structure, we probably need more sophisticated techniques.
- Add staged iteration 
Approaches:
- Ask the rater LLM to look for facts from the amplified text base where the model outputs are incongruent or confused.

- Find common or related words contained in all answers.
- Use amplified sampling for writing texts starting with some of the critical identified common tokens.
- Then ask a rater LLM to evaluate if the generated texts  are aligned with true facts.

### Free Association Technique
- Instead of using prompt queries, which might or might not activate circuits involved in the fine-tuning, try asking for free associations
    + starting with random tokens
    + or starting with random tokens, which have a high kl-contribution on average
    + starting with random normal word. (gives 0% kansas related for α = 5 ; 90% for α = 50)

## Next Steps
- Compare a variety of these techniques in terms of the frequencies that they list kansas related words on larger sample sizes
