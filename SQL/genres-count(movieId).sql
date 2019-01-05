select genres, count(movieId)
from movies
group by genres;