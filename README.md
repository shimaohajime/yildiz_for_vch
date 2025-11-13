# yildiz_for_vch
Yildiz runs for working group in Vienna 2025 November.

## WikiArt color embeddings aggregation

The repository now contains a helper script that processes the "Color Representations" data
from the [WikiArtVectors](https://github.com/bhargavvader/WikiArtVectors) project. After
downloading the `Color_Representations.csv` file locally, you can compute the first two
principal components over the eight `color_embedding*` columns and aggregate them at the
artist-year (`date`) level by running:

```bash
python -m src.aggregate_color_embeddings \
    /path/to/Color_Representations.csv \
    data/wikiart_color_embeddings_artist_year_pca.csv
```

The script keeps the predominant `genre` and `style` values within each artist-year group
and drops the original `title` and `filename` columns before writing the aggregated CSV.
