# MLlab

Hyperparameter Optimization Project using Python, Scikit Learn, and SMAC.

## Usage

The main idea is to calculate scores for different configurations, and store them somewhere.
```bash
$ python3 workstation.py calc --iterations 1000 --save it1000.json
```

Then use the scored configuration space to learn a KDE distribution for each parameter
based on the best scores.
```bash
$ python3 workstation.py load it1000.json -s 0.2 -H estimator__min_sample_leaf -b 1 20 -S min_sample_leaf.kde
$ python3 workstation.py load it1000.json -s 0.2 -H estimator__min_sample_split -b 2 20 -S min_sample_split.kde
$ python3 workstation.py load it1000.json -s 0.2 -H estimator__max_features -b 0.1 0.9 -S max_features.kde
$ python3 workstation.py load it1000.json -s 0.2 -H estimator__bootstrap -b -1 2 bootstrap.kde
```

And afterwards use the learned distributions to sample in the random search.
```bash
$ python3 workstation.py calc --random-forest -i 20 -p \
    -c estimator__min_samples_leaf min_sample_leaf.kde 1 20 "lambda x: int(round(x))" \
    -c estimator__min_samples_split min_sample_split.kde 2 20 "lambda x: int(round(x))" \
    -c estimator__max_features max_features.kde 0.1 0.9 "lambda x: x" \
    -c estimator__bootstrap bootstrap.kde 0 1 "lambda x: bool(int(round(x)))"
```

## Multivariate KDE
To generate multivariate KDE file, do as follows:
```bash
$ python3 workstation.py load it1000.json -s 0.1 -H estimator__bootstrap estimator__max_features estimator__min_samples_leaf estimator__min_samples_split \
    -S all1000.kde
```

Then it is possible to use the generated mutlivariate KDE as an
input for the random search method RAW:
```bash
$ python3 workstation.py calc --raw -i 100 -C \
    "estimator__bootstrap|estimator__max_features|estimator__min_samples_leaf|estimator__min_samples_split" \
    all1000.kde "lambda *xs : xs[0] > 0.5, x[1], int(round(x[2])), int(round(x[3]))"
```
