import json


# generated_responses_file = "/Users/shiquan/PycharmProjects/GLMP/outputs/test_generated_responses.json"
# predicted_entities_file = "/Users/shiquan/PycharmProjects/GLMP/outputs/test_predicted_entities.json"
generated_responses_file = "/home/shiquan/Projects/tmp/GLMP/outputs/test_generated_responses.json"
predicted_entities_file = "/home/shiquan/Projects/tmp/GLMP/outputs/test_predicted_entities.json"

with open(generated_responses_file, "r") as f:
    generated_responses = json.load(f)

with open(predicted_entities_file, "r") as f:
    predicted_entities = json.load(f)

dict = {}
for sample_id in generated_responses.keys():
    for turn_id in generated_responses[sample_id].keys():
        generated_response = generated_responses[sample_id][turn_id]['generated_response']
        tokens = generated_response.split(" ")
        new_tokens = []
        for idx, tok in enumerate(tokens):
            pred_ent = predicted_entities[sample_id][turn_id][str(idx)]
            if "@" in tok:
                new_tokens.append(pred_ent)
            else:
                new_tokens.append(tok)
        lex_resp = " ".join(new_tokens)
        if sample_id not in dict:
            dict[sample_id] = {}
        if turn_id not in dict[sample_id]:
            dict[sample_id][turn_id] = {}
        dict[sample_id][turn_id] = {
            'context_arr': generated_responses[sample_id][turn_id]['context_arr'],
            'gold_response': generated_responses[sample_id][turn_id]['gold_response'],
            'generated_response': generated_responses[sample_id][turn_id]['generated_response'],
            'lexicalized_response': lex_resp
        }

# file_path = "/Users/shiquan/PycharmProjects/GLMP/outputs/test_generated_responses_lex.json"
file_path = "/home/shiquan/Projects/tmp/GLMP/outputs/test_generated_responses_lex.json"
with open(file_path, "w") as f:
    json_str = json.dumps(dict, indent=4)
    f.write(json_str)

