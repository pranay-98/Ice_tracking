# Ice_tracking

Problem Statement :In the given image we need to determine the airice and icerock boundary for this We need to determine the row number for every column with the highest image gradient for both airice and icerock boundary using three methods: simple, HMM - viterbi, and human feedback, given the edge strength map that quantifies how strong the image gradient is at each place.

1. Simple Technique:
   To find the airice boundary we searched for the row in each column with the maximum pixel value and used those rows to draw the boundary.
   And to find the icerock boundary too we searched for the row in each column with the maximum pixel value and there is atlest 10 rows difference below the airice boundary and used those rows to draw the boundary.

2.Bayes Net - Using Viterbi:
Emission Probability is for each row and each column edge_strength[current_row][current_col]/sum(edge_strength[:,current_col]).
Initial Probability : For each row we considered the emission probability in first column.
Transition probability : The Transition probability from one state to another is solely determined by the previous column. When the row difference between the columns is small, the transition probability is high; when the row difference is large, the transition probability is low. As we iterate through the column list, values that are close together have a higher chance of convergent to the transition function's maximum value. The main idea behind this method is that the horizon is constantly smooth, hence there should be no gaps between the rows.

Then by iterating over all the rows in each column and calculate the probability as log(Emission Probability)+log(previous viterbi probability stored in that table)+log(Transition probability). At each step for each label we store only the max probability obatined by computing the previous equation. Along with that, we also store the row number.

Repeat the previous step until we reach the last column

Likewise we detemine the airice and icerock boundaries using viterbi algorithm.

Note: We used log to avoid very small numbers (/NaN) during multiplication of probabilities.

3. Human Feedback:
   For human feedback we just used the previous viterbi algorithm and applied it from the given human feedback row and column.
   we have applied the viterbi in two directions one for the left side columns of the given column and the other one is for the right side of the given columns and at the end we combine the rows returned from the both sides.
