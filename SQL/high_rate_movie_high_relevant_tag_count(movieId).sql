select count(a.movieId), b.tag, max(b.relevance)
from (select movies.movieId as movieId, movies.genres as genres
        from movies , meanrating
         where movies.movieId=meanrating.movieId and meanrating.mean_rating>4
         group by movies.movieId, movies.genres) as a, 
		 (select genomescores.movieId as movieId, genometags.tag as tag, genomescores.relevance as relevance
          from genomescores, genometags
          where genomescores.tagId=genometags.tagId and genomescores.relevance>0.9) as b
where a. movieId=b.movieId
group by b.tag
order by count(a.movieId) desc;