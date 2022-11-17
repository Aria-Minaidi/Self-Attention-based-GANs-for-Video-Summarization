import numpy as np
import math
from knapsack_implementation import knapSack

def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions): 
    all_summaries = []
    for video_index in range(len(all_scores)):
    	# Get shots' boundaries
        shot_bound = all_shot_bound[video_index] #change points [number_of_shots, 2] - the boundaries refer to the initial number of frames (before the subsampling)
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]

        #print('positions',len(positions))
        #print('frame_init_scores',len(frame_init_scores))
        
        

        # Compute the importance scores for the initial *frame sequence* (not the subsampled one) (gia omades frames me vasi ta positions)
        frame_scores = np.zeros((n_frames), dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])     #vazei kai ta upoloipa frames 
            
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i+1]
            if positions[i] == len(frame_init_scores):     #an to positions[i] einai iso me to length twn subsampled frames (px 205),
                frame_scores[pos_left:pos_right] = 0
            else:
               
                frame_scores[pos_left:pos_right] = frame_init_scores[i]
        #coutn = 0
        '''
        for i in frame_scores:
          if math.isnan(i):
            coutn+=1
        '''
        

        #print('frame scores',len(frame_scores),frame_scores)
    	# Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        shot_imp_scores = []
        shot_lengths = []
        
        
        for shot in shot_bound:
            
            shot_lengths.append(shot[1]-shot[0]+1)
            shot_imp_scores.append((frame_scores[shot[0]:shot[1]+1].mean()).item())
      
        
        
	# Select the best shots using the knapsack implementation
        final_max_length = int((shot[1]+1)*0.15)
        #print('max length',final_max_length)
        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))
		
	# Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(shot[1]+1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1]+1] = 1
        #print('summary,', len(summary))
        
        all_summaries.append(summary)
    
    #print(frame_scores)
    #print(frame_scores[0])

    return all_summaries

