# Historical-cipher-project
Master thesis Historical cipher project

## Preprocessing Pipeline

### Sauvola Binarization
```bash
python sauvola_binarization.py
```

Performs Sauvola binarization on the input cipher images, adjust settings depending on the cipher document quality

### Deskewing
```bash
python deskew.py
```

Corrects skew in the binarized images

### Remove Neighbors

```bash
python remove_neighbors.py
```

Removes unwanted neighboring components or noise before connected component

### Connected Components
```bash
python connected_components.py
```

Extracts connected components from the processed cipher images

### Feature Extraction

```bash
python feature_extractor_with_ngram.py
```


## Clustering


```bash
python hierarchical_single_pass.py
```

Clustering step groups similar cipher components based on the extracted features

For a each cipher, the first step is to find the optimal clustering threshold. To do this, run `hierarchical_single_pass.py` setting

```python
USE_THRESHOLD_SEARCH = True
```


The threshold search is the default clustering method used in the master thesis.

Once the optimal threshold has been found, update the script by setting to

```python
USE_THRESHOLD_SEARCH = False
```

Then set the `DEFAULT_THRESHOLD` value to the optimal threshold found during the threshold search

```python
DEFAULT_THRESHOLD = <your_optimal_threshold>
```

After this, run the clustering script again:

```bash
python hierarchical_single_pass.py
```

This final run uses the selected fixed threshold for clustering


## License

This project is licensed under the MIT License.

You may use, copy, modify, and distribute this project, including for commercial purposes, as long as the original copyright notice and license text are included.

Copyright (c) 2026 Diego Gabriel Valladares Parker


