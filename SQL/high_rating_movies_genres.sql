select count(a.movieId) as numvers_of_movie, a.genres as genres
from (select movies.movieId, movies.genres
        from movies , meanrating
         where movies.movieId=meanrating.movieId and meanrating.mean_rating>4
         group by movies.movieId, movies.genres) as a
group by a.genres
order by count(a.movieId) desc;

