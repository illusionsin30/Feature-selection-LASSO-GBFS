# PRML Project 1

Pattern Recognition and Machine Learning - Project 1 (Spring 2026)
**Original article**: [Gradient Boosted Feature Selection](https://arxiv.org/abs/1901.04055)

## Installation

With pip:

```bash
pip install -r requirements.txt
```

## Tasks
### Task 1

Run

```python
python src/baseline.py
```

The results should be like `images/feature_selection_by_bag_lasso.png` and `images/test_error_vs_selected_features.png`. The script writes fresh `.svg` figures by default.

### Task 2

Run

```python
python src/gbfs.py
```

The results should be like `images/feature_selection_by_bag_gbfs.png`. The script writes a fresh `.svg` figure by default. Use `--n-jobs` to tune parallelism for your device, for example:

```bash
python src/gbfs.py --n-jobs 4
```

### Task 3

Run

```python
python src/hyperparam.py --n-jobs 4
```

to get the result. The results should be like `images/task3_results.png`. The script writes a fresh `.svg` figure by default.
