select rating, count(rating)
from ratings
group by rating;