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