---
applyTo: '*.py'
---
# Copilot Instructions for Vibe Research in Python
- In my main research file, I usually use the *Jupyter python script mode*, i.e.,
    - there are cells that convey execution units.
    - the cells are separated by `#%%`

## Jupyter Python Script Conventions
- The cell comment should be followed with a descriptive heading of the cells content.
- For example, `#%% Load Models and Dataset`
- The cell division should match execution grouping. I.e., 
    - simple function definitions should be joined in a larger cell, since they execute instantly
    - we usually want a function definition to be joined in a cell together with a top level execution function
    - long independent data processing steps should be in separate cells, such that they can be rerun independently

- Put execution code on toplevel. Refrain from deep nesting. Do not put a `if __name__ == "__main__":` block anywhere — we are executing top-level code in Jupyter cells. 
- Usually, top-level code is best.
- Do not execute cells yourself which download large amounts of data or run computations longer than 5s.


### ML Conventions
- Usually, we load models in the beginning of the file, in one of the first cells.
- Consider the size and scale of data objects when processing.
- We often want to do batch processing, e.g. for sampling a lot of text. 

## General Conventions
- do NOT add side comments into the file.
    - Side information is information that is not directly commenting on the intent of the code in any way.
    - Examples of side comments are: `# For alternatives, see [link]`, `# Alternatives to the dataset are …`
- do NOT create additional `.py` files unless explicitly requested. This is a research exploration notebook — we want to keep everything in one file.

- If you're unsure of something, ask for clarification.
- Only suggest experiment ideas if asked for. Don't make high-level experiment design decisions on your own or on the fly.
- If you're aware of unintuitive behavior and significant problems with an approach, mention these in the chat.
- If you have a suggestion for something that currently looks very unprincipled, and there is a significantly better, principled approach, feel free to mention it in the chat. 
- do NOT put overly verbose print statements in the file unless asked
- keep code minimal
- usually, use pandas dataframes for storing experiment results and plotting results from it
- when printing dataframes, print them in colorscale (default blue - red)
- when printing

## Code Quality
- Use typehints where possible.

## Python Setup
- We usually use the UV tool for dependency management.

## Platform
- We usually are in a remote environment. Large code should be 

## Versioning Conventions
- If using wandb: Update the wandb short description / experiment name if we have significant changes in the experiment .

## Plotting
- Prefer plotly for interactive exploration.