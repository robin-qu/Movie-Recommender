select movieId, count(rating) as total_count, sum(rating)/count(rating) as mean_rating
from ratings
group by movieId;