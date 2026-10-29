# Neural Collaborative Filtering for Movie Recommendation

This project compares a classic **Matrix Factorization** baseline with a **Neural Collaborative Filtering (NeuMF)** model on the MovieLens dataset. Both models are written in PyTorch, trained on the same data split, and evaluated on rating prediction (RMSE) and on top-10 recommendation quality (Hit Ratio@10).

## Data Setup (Required)

The dataset is not included in this repository, because the MovieLens license does not allow redistributing the data files. Please download it yourself:

1. **Download the dataset** (`ml-latest-small`):
   [https://files.grouplens.org/datasets/movielens/ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
2. **Extract the zip** and copy `ratings.csv` into the `data/` folder of this project. The code uses only this file.

Your folder structure should look like this:

```text
neural_cf_movie_recsys/
├── data/
│   └── ratings.csv
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

`ml-latest-small` contains 100,836 ratings from 610 users on 9,724 movies. To cite the dataset, use: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19.

## Models

Both models are trained for 10 epochs with a batch size of 1024 and an MSE loss on the rating.

| | Matrix Factorization (baseline) | NeuMF (proposed) |
| :--- | :--- | :--- |
| **Idea** | Rating = global bias + user bias + item bias + dot product of user and movie embeddings | A GMF branch (element-wise product of embeddings) and an MLP branch (concatenated embeddings) are combined in one prediction layer |
| **Embedding size** | 32 | 32 per branch |
| **MLP layers** | none | 64 → 32 → 16, ReLU, dropout 0.2 |
| **Optimizer** | SGD, `lr=0.05`, momentum 0.9, weight decay 1e-4 | Adam, `lr=0.001` |

NeuMF follows the architecture from [He et al., *Neural Collaborative Filtering* (WWW 2017)](https://arxiv.org/abs/1708.05031), trained end-to-end from scratch.

## Evaluation

* **Split:** random 80% / 20% train-test split of the ratings (`random_state=42`). The random seed is also fixed for Python, NumPy and PyTorch.
* **RMSE:** rating prediction error on the test set. The best value over the 10 epochs is reported.
* **Hit Ratio@10:** for up to 1,000 test interactions the user rated above 3.5, the held-out movie is mixed with 99 randomly sampled movies, and the model ranks all 100. A hit means the held-out movie lands in the top 10.

At the end, the script prints a side-by-side comparison of both models and the RMSE improvement of NeuMF over the baseline. Exact numbers vary slightly between machines and PyTorch versions, so run the script to see them.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/usakgun/neural_cf_movie_recsys.git
   cd neural_cf_movie_recsys
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the data as described in **Data Setup** above.
4. Run the script:
   ```bash
   python src/main.py
   ```

The script uses a GPU automatically if one is available, and falls back to the CPU otherwise.

## Technologies

* **Python**
* **PyTorch:** Model definitions, training loop and evaluation.
* **Pandas & NumPy:** Data loading and ID mapping.
* **Scikit-Learn:** Train-test split.
