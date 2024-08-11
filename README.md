# Movie recommendation system

A customizable movie recommendation system that utilizes pluggable factors and weights to recommend movies.

Used libraries:
```bash
pip install sklearn pandas numpy flask gunicorn
```

Run in debug mode:
```bash
python3 app.py
```
or running in production:
```bash
gunicorn --bind "[::]:5000" app:app
```

Default only chooses 5000 movies from 30000 movies. If you want to use the entire dataset, please check the descriptions in `recommendations.py`.

# Datasources

- [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)
- [Full TMDB Movies Dataset 2024 (1M Movies) (kaggle.com)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
- [🎬 Massive Rotten Tomatoes Movies & Reviews (kaggle.com)](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews)
- [Consumer Price Index for All Urban Consumers: Purchasing Power of the Consumer Dollar in U.S. City Average (CUUR0000SA0R) | FRED | St. Louis Fed (stlouisfed.org)](https://fred.stlouisfed.org/series/CUUR0000SA0R)

# Acknowledgements

This is the final project for CSCI-S-108 Data Mining, Discovery, and Exploration course on Harvard Summer School 2024.

I'd like to thank Dr. Stephen Elston for creating this great course and inspiring me to complete this project.

Also, I'd like to thank [Getting Started with a Movie Recommendation System (kaggle.com)](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system) for giving me ideas on how to design a recommendation system.