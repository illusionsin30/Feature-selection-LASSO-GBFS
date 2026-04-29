# PRML Project 1

Pattern Recognition and Machine Learning - Project 1 (Spring 2026)

## Installation
```bash
pip install -r requirements.txt
```

## Tasks
### Task 1

Enter `src/baseline.py` and run to get the results of Task 1. Directory `images/` gives some results of our runs. 

### Task 2

Enter `src/bgfs.py` and run to get the results of Task 2. Vectorized Implementation of regression trees with feature-introduction penalty can be found in `src/tree.py`.

### Task 3

Run

```python
cd src
python hyperparam.py
```

to get the result. You should notice that the default setting for parallel accelerating in `hyperparam.py` may not be friendly to your device. Just find and modify the following content based on your device:

```python
results = Parallel(n_jobs=your_cores, verbose=10)(
    delayed(run_one_fold)(mu, depth, X, y, bags, train_idx, test_idx, epsilon, T)
    for mu, depth, fold_id, train_idx, test_idx in tasks
)
```