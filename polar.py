#!/usr/local/bin/python3
#
# Authors: [hkande-pdasari-pkapil]
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

def max_pixels_air(edge_s):
    max_pixel=[]
    for i in range(len(edge_strength[0])):
        max_pix=0
        for j in range(len(edge_strength)//4):
            if edge_s[j][i] > max_pix:
                max_pix = edge_s[j][i]
                row_index = j

        max_pixel.append(row_index)
    return max_pixel

def max_pixels_rock(edge_s,airice_row):
    max_pixel=[]
    for i,air_row in zip(range(len(edge_strength[0])),airice_row):
        max_pix=0
        for j in range(len(edge_strength)):
            if edge_s[j][i] > max_pix:
                if j-air_row>10:
                    max_pix = edge_s[j][i]
                    row_index = j
                else:
                    max_pix=0
        max_pixel.append(row_index)
    return max_pixel

def cal_emission_probability(edge_strength):

    emission_prob = zeros((edge_strength.shape[0], edge_strength.shape[1]))

    for col in range(emission_prob.shape[1]):

        for row in range(emission_prob.shape[0]):
            if sum(edge_strength[:,col]) == 0:
                emission_prob[row][col] = 0
            else:
                emission_prob[row][col]= log(edge_strength[row][col]/sum(edge_strength[:,col]))

    return emission_prob

def emission_probs(edge_pix,emission_prob,feed_col=1):
    table=[]
    for row in range(edge_pix.shape[0]):
        table.append(emission_prob[row][feed_col])
    return table

# Viterbi algorithm referenced form in class activity on Viterbi

def air_hmm(edge_strength):
    emission_prob = cal_emission_probability(edge_strength)
    
    output = []

    observed_probabilities=emission_probs(edge_strength,emission_prob)
        
    output.append(where(observed_probabilities == max(observed_probabilities))[0][0])

    for col in range(1, edge_strength.shape[1]):                            

        for row1 in range(edge_strength.shape[0]):
            viterbi = []
            for row2 in range(edge_strength.shape[0]):

                viterbi.append(emission_prob[row1][col] + observed_probabilities[row2] + log( ( edge_strength.shape[1]- (abs(row1 - row2) ))/ edge_strength.shape[1] ))

            observed_probabilities[row1] = max(viterbi)

        output.append(where(observed_probabilities == max(observed_probabilities))[0][0])
    return output

def ice_hmm(edge_strength):
    emission_prob = cal_emission_probability(edge_strength)
    
    output = []

    observed_probabilities=emission_probs(edge_strength,emission_prob)
        
    output.append(where(observed_probabilities == max(observed_probabilities))[0][0] + 43)

    for col in range(1, edge_strength.shape[1]):                            

        for row1 in range(edge_strength.shape[0]):
            viterbi = []
            for row2 in range(edge_strength.shape[0]):

                viterbi.append(emission_prob[row1][col] + observed_probabilities[row2] + log( ( edge_strength.shape[1]- (abs(row1 - row2)))/ edge_strength.shape[1] ))

            observed_probabilities[row1] = max(viterbi)

        output.append(where(observed_probabilities == max(observed_probabilities))[0][0] + 43)
    return output

    
def air_feedback(edge_pix,feed_row,feed_col):
    v_table=[]
    curr_row = feed_row
    emission_prob = cal_emission_probability(edge_pix)
    output =[]
    output.append(feed_row)
    if (feed_col==0):
        
        v_table=emission_probs(edge_pix,emission_prob,feed_col)

        for col in range(feed_col+1, edge_pix.shape[1]):

            for row1 in range(edge_pix.shape[0]):
                prob = []
                for row2 in range(edge_pix.shape[0]):
                    prob.append(emission_prob[row1][col] + v_table[row2] + log(( edge_pix.shape[1]- (abs(row1 - row2) ))/ edge_pix.shape[1] ))

                v_table[row1] = max(prob)

            x_table = v_table[curr_row-4:curr_row+4]  
            curr_row = argmax(x_table) + curr_row -4
            output.append(curr_row)
    else:
        left_rows=[]
        
        v_table=emission_probs(edge_pix,emission_prob,feed_col)
        for col in range(feed_col+1, edge_pix.shape[1]):

            for row1 in range(edge_pix.shape[0]):
                prob = []
                for row2 in range(edge_pix.shape[0]):
                    prob.append(emission_prob[row1][col] + v_table[row2] + log(( edge_pix.shape[1]- (abs(row1 - row2)))/ edge_pix.shape[1] ))

                v_table[row1] = max(prob)

            x_table = v_table[curr_row-4:curr_row+4]
            curr_row = argmax(x_table) + curr_row -4
            left_rows.append(curr_row)
        curr_row = feed_row
        v_table=[]
        right_rows=[]
        v_table=emission_probs(edge_pix,emission_prob,feed_col)

        for col in range(feed_col-1, -1,-1):

            for row1 in range(edge_pix.shape[0]):
                prob = []
                for row2 in range(edge_pix.shape[0]):
                    prob.append(emission_prob[row1][col] + v_table[row2] + log(( edge_pix.shape[1]- (abs(row1 - row2) ))/ edge_pix.shape[1] ))

                v_table[row1] = max(prob)
            x_table = v_table[curr_row-4:curr_row+4]
            curr_row = argmax(x_table) + curr_row-4
            right_rows.append(curr_row)

        right_rows.reverse()
        output = right_rows+left_rows

    return output
    
def ice_feedback(edge_pix,feed_row,feed_col):
    v_table=[]
    curr_row = feed_row
    emission_prob = cal_emission_probability(edge_pix)
    output =[]
    output.append(feed_row)
    if (feed_col==0):
        
        v_table=emission_probs(edge_pix,emission_prob,feed_col)

        for col in range(feed_col+1, edge_pix.shape[1]):

            for row1 in range(edge_pix.shape[0]):
                prob = []
                for row2 in range(edge_pix.shape[0]):
                    prob.append(emission_prob[row1][col] + v_table[row2] + log(( edge_pix.shape[1]- (abs(row1 - row2) ))/ edge_pix.shape[1] ))

                v_table[row1] = max(prob)
            x_table = v_table[curr_row-4:curr_row+4]    
            curr_row = argmax(x_table) + curr_row -4
            output.append(curr_row)

    else:
        left_rows=[]
        
        v_table=emission_probs(edge_pix,emission_prob,feed_col)

        for col in range(feed_col+1, edge_pix.shape[1]):

            for row1 in range(edge_pix.shape[0]):
                prob = []
                for row2 in range(edge_pix.shape[0]):
                    prob.append(emission_prob[row1][col] + v_table[row2] + log(( edge_pix.shape[1]- (abs(row1 - row2) ))/ edge_pix.shape[1] ))

                v_table[row1] = max(prob)
            x_table = v_table[curr_row-7:curr_row+7]
            curr_row = argmax(x_table) + curr_row - 7
            left_rows.append(curr_row)
        curr_row = feed_row
        v_table=[]
        right_rows=[]
        
        v_table=emission_probs(edge_pix,emission_prob,feed_col)

        for col in range(feed_col-1, -1,-1):

            for row1 in range(edge_pix.shape[0]):
                prob = []
                for row2 in range(edge_pix.shape[0]):
                    prob.append(emission_prob[row1][col] + v_table[row2] + log(( edge_pix.shape[1]- (abs(row1 - row2) ))/ edge_pix.shape[1] ))

                v_table[row1] = max(prob)
            x_table = v_table[curr_row-4:curr_row+4]
            curr_row = argmax(x_table) + curr_row-4
            right_rows.append(curr_row)

        right_rows.reverse()
        output = right_rows+left_rows

    return output

# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))


    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    airice_simple = max_pixels_air(edge_strength)
    airice_hmm = air_hmm(edge_strength[0:edge_strength.shape[0]//4,:])
    airice_feedback=air_feedback(edge_strength,gt_airice[0],gt_airice[1])

    icerock_simple = max_pixels_rock(edge_strength,airice_simple)
    icerock_hmm = ice_hmm(edge_strength[edge_strength.shape[0]//4:edge_strength.shape[0],:])
    icerock_feedback= ice_feedback(edge_strength,gt_icerock[0],gt_icerock[1])

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
