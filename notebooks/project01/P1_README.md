# P1: Tools and Intro to ML & Engage
## About
- This project is for week 1 of 44-670 (Applied Machine Learning) at Northwest Missouri State University. We analyzed customer housing data in California to train and test a model to understand if we can predict the Median House Value in the state of California.
- To do this, we looked at the median house price and leveraged it using the number of rooms in a house, something big to consider when providing for things like a family, pets, or office spaces.

## Finding the Notebook
I used Jupyter Notebooks for this assignment. You can [find this notebook](notebooks/project01/ml01.ipynb) within this repo at the given link.

## Set up Machine
Followed instructions in the SET_UP_MACHINE.md by Dr. Case.
- No issues presented.
### Workflow Followed:
1. Found the template from Dr. Case in the assignment.
   1. Forked it to my own personal machine using the following code
```shell
cd C:\Repos
git clone MYREPOHTML
```
2. Opened repo within my VS code

## Set up Project
Followed instructions in the SET_UP_PROJECT.md by Dr. Case.
- No issues presented.
### Workflow Followed:
1. Used following code to set up .venv and dependencies:
```shell
# 1. Create an isolated virtual environment
uv venv

# 2. Pin a specific Python version (3.12 recommended)
uv python pin 3.12

# 3. Install all dependencies, including optional dev/docs tools
uv sync --extra dev --extra docs --upgrade

# 4. Enable pre-commit checks so they run automatically on each commit
uv run pre-commit install

# 5. Verify the Python version (should show 3.12.x)
uv run python --version
```
2. Personalized pyproject and mkdocs files
3. Add - Commit - Push

## ML Project 1
Followed instructions in the README.md of docs/project1 by Dr. Case.
- Followed path to complete Project 1 in notebooks/project1/ml01.ipynb
- No issues were presented, followed using example data as directed.
- Tested notebook with following code and submitted.
```shell
uv run python notebooks/project01/ml01.py
```