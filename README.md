# PRML Project 1

Pattern Recognition and Machine Learning - Project 1 (Spring 2026)

## Installation
```bash
pip install -r requirements.txt
```

## Tasks
### Task 1

Run

```python
cd src
python baseline.py
```

The results should be like `images/feature_selection_by_bag_lasso.png` and `images/test_error_vs_selected_features.png`.

### Task 2


Run

```python
python gbfs.py
```

The results should be like `images/feature_selection_by_bag_gbfs.png`. You should notice that the default setting for parallel accelerating in `hyperparam.py` may not be friendly to your device. Just find and modify the following content based on your device:

```python
results = Parallel(n_jobs=your_cores, verbose=10)(
    delayed(run_one_fold)(mu, depth, X, y, bags, train_idx, test_idx, epsilon, T, mode="structured")
    for mu, depth, fold_id, train_idx, test_idx in tasks
)
```

### Task 3

Run

```python
cd src
python hyperparam.py
```

to get the result. The results should be like `images/task3_results.png`.