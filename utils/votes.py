from utils.enums import EnsembleVoteMode

def ensemble_votes(votes: list, mode: EnsembleVoteMode = EnsembleVoteMode.MAJORITY):
    coherent_sum = 0
    correct_sum = 0
    for v in votes:
        if v['coherent']:
            coherent_sum += 1

        if v['correct']:
            correct_sum += 1

    if mode == EnsembleVoteMode.MAJORITY:
        return coherent_sum >= 2, correct_sum >= 2
    elif mode == EnsembleVoteMode.FULL:
        return coherent_sum >= 3, correct_sum >= 3
    else:
        raise ValueError("Invalid Mode")

def filter_models_votes(votes, models_name):
    human_votes = []
    model_votes = []

    for v in votes:
        if v['model'] not in models_name:
            human_votes.append(v)
        else:
            model_votes.append(v)
    
    return human_votes, model_votes