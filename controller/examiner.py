from controller.prompt_pipeline import PromptPipeline
from glob import glob
import re

class Examiner:

    def __init__(self, llm_model) -> None:
        self.model = llm_model
        self.usage = None
        self.examiner_prompts = sorted(glob("prompt/examiner/*.txt"))

    def overall_rate(self, response, votes):
        q_weight = 1
        a_weight = 1 if len(response.answer_bboxes) > 0 else 0.5
        overall_q = sum([v * q_weight for v in votes['question'].values()]) / len(votes['question'].values())
        overall_a = sum([v * a_weight for v in votes['answer'].values()]) / len(votes['answer'].values())

        return round(overall_q, 2), round(overall_a, 2)

    def calculate_votes(self, model_votes, responses):
        aster_cleaner_pattern = re.compile(r'\*\*') #remoção de ** do Gemini
        keys_cleaner_pattern = re.compile(r'[{}]')
        calculated_votes = []
        for vote in model_votes:
            votes = {
                "question": {
                    "relevance": 0,
                    "coherence": 0,
                    "objectivity": 0,
                },
                "answer": {
                    "accuracy": 0
                }
            }

            vote = aster_cleaner_pattern.sub('', vote)
            vote = keys_cleaner_pattern.sub('', vote)
            if len(vote.strip()) <= 0 or "|" not in vote or "-" not in vote:
                #excluindo linhas vázias ou texto desnecessário
                continue
            
            i = int(vote.split("-")[0])-1
            vote = vote.split("-")[1]
            
            question_vote, answer_vote = tuple(vote.split("|")[:2])
            question_vote = tuple(question_vote.split(','))
            answer_vote = tuple(answer_vote.split(','))          
                        
            votes['question']['relevance']   = int(question_vote[0].split(":")[-1])
            votes['question']['coherence']   = int(question_vote[1].split(":")[-1])
            votes['question']['objectivity'] = int(question_vote[2].split(":")[-1])

            votes['answer']['accuracy']  = int(answer_vote[0].split(":")[-1])

            votes['question']['overall'], votes['answer']['overall'] = self.overall_rate(
                responses[i], votes
            )
            calculated_votes.append(votes)

        return calculated_votes

    def examine(self, responses, prompt_data):        
        qas = ""
        vote_region = ""
        for idx, response in enumerate(responses):
            qas += f"{idx+1} - Questão: {response.question} Resposta: {response.answer}\n"
            vote_region += f"{idx+1}"+"-a1:{relevância},a2:{coerência},a3:{objetividade}|b1:{acurácia}\n"

        #vamos utilizar o último prompt de examiner existente
        with open(self.examiner_prompts[-1]) as prompt_file:
            prompt_str = prompt_file.read()
        
        data = {
            "{dominio}": "financeiro",
            "{prompt_data}": prompt_data,
            "{qas}": qas,
            "{vote_region}": vote_region
        }

        prompt_pipe = PromptPipeline(prompt_str)
        prompt_pipe.add_data_to_prompt(data)
        
        self.model_votes, self.usage = self.model.call(prompt_pipe)
        # print(self.model_votes)
        return self.calculate_votes(self.model_votes, responses), self.usage