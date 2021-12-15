#!/usr/bin/env bash

export resp_gen_model_path=$1
export ent_pred_model_path=$2


# generate delexicalized responses using fine-tuned GPT-2
python generate_responses.py -path=${resp_gen_model_path}

# predict per-timestep entities using generated responses
# python predict_entities.py -path=${ent_pred_model_path}

# lexicalize the generated responses using predicted entities
# python lexicalize_generated_responses_using_predicted_entities.py

# calculate final metrics using lexicalized responses and gold responses
# python evaluate_lexicalized_responses.py