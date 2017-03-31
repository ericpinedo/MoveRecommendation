#import pandas to get access to built-in corr() which will compute a
import pandas as pd

#create a list of the movie ratings with id and user's id
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('./ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

#creates a list of the movie title with movie id
m_cols = ['movie_id', 'title']
movies = pd.read_csv('./ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

# merge both structures together
ratings = pd.merge(movies, ratings)

#used to check if merge was successful
ratings.head()

# pivot the table to construct a nice matric of users and moviees they rated
#nan represents missing data
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

# computer a corelation score for every column pair in the matrix
corrMatrix = userRatings.corr()
corrMatrix.head()

"""In order to restrict our results to movies that lots of people rated
 together - and also give us more popular results that are more 
 easily recongnizable - we'll use the min_periods argument to throw
 out results where fewer than 100 users rated a given movie pair:"""
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()


# create a user with ID 0 and manually add to data set
myRatings = userRatings.loc[0].dropna()
myRatings

"""Now, let's go through each movie I rated one at a time, 
   and build up a list of possible recommendations based on the
   movies similar to the ones I rated. So for each movie I rated,
   I'll retrieve the list of similar movies from our correlation
   matrix. I'll then scale those correlation scores by how well I 
   rated the movie they are similar to, so movies similar to ones I 
   liked count more than movies similar to ones I hated:"""

simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print "Adding sims for " + myRatings.index[i] + "..."
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print "sorting..."
simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)


# sort the moves based on similarity score
simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)

# Next filtered out movies I've already rated
filteredSims = simCandidates.drop(myRatings.index)
print filteredSims.head(10)