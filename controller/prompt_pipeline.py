from utils.enums import PromptRoleType, LangChainRoleType
from model.prompt_data import PromptData, PromptResponse
from Levenshtein import ratio
import re, requests, nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#TODO: salvar o conteúdo desse arquivo para disco
r = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-pt/master/stopwords-pt.txt")
STOPWORDS = r.content.decode("utf-8").split('\n')
for stp in stopwords.words('portuguese'):
  if stp not in STOPWORDS:
    STOPWORDS.append(stp)
STOPWORDS.remove("não")
print("Quantidade de stopwords:", len(STOPWORDS))

class PromptPipeline():

    def __init__(self, prompt_str: str):
        self.pipeline = self.__map_prompt_string(prompt_str)

    def __get_data_role(self, value) -> PromptRoleType:
        """Mapeamento de role do assistente, para criar a chamada"""
        if value == PromptRoleType.USER.value:
            return LangChainRoleType.USER
        if value == PromptRoleType.SYSTEM.value:
            return LangChainRoleType.SYSTEM
        if value == PromptRoleType.ASSISTANT.value:
            return LangChainRoleType.ASSISTANT
        
        raise ValueError("A role do prompt é inválida.")

    def add_data_to_prompt(self, data: dict):
        """Alimenta o prompt com os dados necessários para finalização da instrução"""
        for d in data:
            for p in self.pipeline:
                p.replace_marker(d, data[d])

    def __map_prompt_string(self, prompt_str: str):
        """Mapeamento da string de entrada para definir roles e estrutura da mensagens
        ao modelo de linguagem."""
        pipeline = []
        # Separa o texto por quebras de linha
        lines = prompt_str.split('\n')

        tag = None
        tag_text = ""

        # Itera sobre as linhas do texto
        for line in lines:
            # Verifica se a linha indica uma nova região
            if line.startswith(('SYSTEM:', 'USER:', 'ASSISTANT:')):
                # Extrai o nome da região
                if tag is not None:
                    role = self.__get_data_role(tag)
                    pipeline.append(PromptData(role, tag_text))
                tag_marker_idx = line.index(':')
                tag = line[:tag_marker_idx].strip()
                tag_text = line[tag_marker_idx+1:].strip()
            elif tag:
                tag_text += line.strip() + "\n"

        return pipeline

    def __define_range(self, pos_region: str):
        clean_region_pattern = re.compile(r'[^0-9]')

        if " E " in pos_region:
            p1, p2 = tuple([p.strip() for p in pos_region.split(" E ")])
            p1 = clean_region_pattern.sub('', p1)
            p2 = clean_region_pattern.sub('', p2)
            return [int(p1)-1, int(p2)-1]
        elif " A " in pos_region:
            start, end = tuple([p.strip() for p in pos_region.split(" A ")])
            start = clean_region_pattern.sub('', start)
            end = clean_region_pattern.sub('', end)
            return [v-1 for v in range(int(start)-1, int(end))]
        else:
            return [int(clean_region_pattern.sub('', pos_region.strip()))-1]

    def __check_pattern(self, region, pattern) -> bool:
        return len(pattern.findall(region)) > 0

    def __map_regions(self, region: str):
        # Regex para identificar as combinações desejadas
        pattern_T = re.compile(r'T\d+')
        pattern_table = re.compile(r'TABELA\s\d+')
        pattern_line = re.compile(r'LINHA\s\d+\b|LINHAS\s\d+')

        # Encontrar todas as ocorrências das combinações na string
        has_text = self.__check_pattern(region, pattern_T)
        has_table = self.__check_pattern(region, pattern_table)
        has_line = self.__check_pattern(region, pattern_line)
        
        if ' - ' in region:
            region_splited = region.split(" - ")
        else:
            region_splited = region.split(",")
        
        if has_table and has_line:
            #criando a máscara para procurar qual a posição da info de tabela e linha
            table_mask = ["TABELA" in r for r in region_splited]
            line_mask = ["LINHA" in r for r in region_splited]
            #localizando o indice da tabela e da linha da tabela
            table_loc_idx = table_mask.index(True) if True in table_mask else None
            line_loc_idx = line_mask.index(True) if True in line_mask else None

            #tratando possivel falha de mapeamento, garantindo que o IDX sempre esteja na lista
            #tanto para a tabela e linha. A definição da região só irá ocorrer caso as duas 
            #informações estejam definidas na saída do modelo de linguagem
            if (table_loc_idx is not None and table_loc_idx < len(region_splited)) and \
                (line_loc_idx is not None and line_loc_idx < len(region_splited)):
                region_splited = [
                    region_splited[table_loc_idx].strip(),
                    region_splited[line_loc_idx].strip()
                ]
            else:
                #caso falhe, reseta da flag e limpa a lista
                has_table = False
                region_splited = []
        elif has_text:
            #procura a região que contenha a localização de texto e atualiza a 
            #lista de posições
            text_mask = [bool(pattern_T.match(r)) for r in region_splited]
            text_loc_idx = text_mask.index(True) if True in text_mask else None
            if text_loc_idx is not None and text_loc_idx < len(region_splited):
                region_splited = [region_splited[text_loc_idx].strip()]
            else:
                #caso falhe, reseta da flag e limpa a lista
                has_text = False
                region_splited = []
        else:
            #resetando a região do texto para facilitar o processamento
            has_table = False
            has_line = False
            region_splited = []
        
        #faz a conversão de string para int, ajustando para indice de array
        region_splited = [
            self.__define_range(r) 
            for r in region_splited
        ]
        return has_table, has_text, region_splited

    def __get_text_lines_bbox(self, answer: str, text_blocks: list, paragraph: int):
        """Busca as boundig boxes das linhas que contém o texto de saída do prompt"""
        lines_bboxes = []
        _answer = answer.lower()
        base_text = ""
        marker_pattern = re.compile(r'[^\w\s]')
        
        block = text_blocks[paragraph]
        for text_line in block.texts:
            base_text += text_line.text + "\n"

            splited_text = [
                marker_pattern.sub('', word) for word in text_line.text.lower().split(" ") 
                    if word not in STOPWORDS
            ]
            splited_answer = [
                marker_pattern.sub('', word) for word in _answer.split(" ") 
                    if word not in STOPWORDS
            ]

            l_splited_text = len(splited_text)
            l_splited_answer = len(splited_answer)

            min_value = min(l_splited_text, l_splited_answer)

            if min_value <= 0:
                continue

            intersec_count = 0
            if min_value == l_splited_text:
                #a linha do texto tem que ta contida na resposta
                for st in splited_text:
                    intersec_count += 1 if st in splited_answer else 0
            else:
                #a resposta tem que ta contida na linha do texto
                for sa in splited_answer:
                    intersec_count += 1 if sa in splited_text else 0

            if intersec_count/min_value > 0.4:
                lines_bboxes.append(text_line.true_bbox)
        
        if len(lines_bboxes) == 0:
            #caso não ache o texto, vai devolver todo o paragrafo
            lines_bboxes.append(block.bbox)
        
        return lines_bboxes, base_text

    def __map_table_rows(self, table_structure):
        #vamos mapear as linhas das tabelas para ignorar as linhas de cabeçalho
        header_row_num = []
        bodies_row_num = []
        for cells in table_structure:
            if cells['column header']:
                for rn in cells['row_nums']:
                    if rn not in header_row_num:
                        header_row_num.append(rn)
            else:
                for rn in cells['row_nums']:
                    if rn not in bodies_row_num:
                        bodies_row_num.append(rn)

        return header_row_num, bodies_row_num

    def __parse_table_response(self, answer, lines_idx, table):
        #se a tabela existir na lista
        answer_bboxes = []
        table_row_bbox = [10000, 0, 0, 0] #constroi a bbox da linha para caso de falha do mapeamento
        
        for cells in table.table_structure:
            if cells['row_nums'][0] not in lines_idx:
                continue
            
            table_row_bbox = [
                min(table_row_bbox[0], cells['bbox'][0]),
                max(table_row_bbox[1], cells['bbox'][1]),
                max(table_row_bbox[2], cells['bbox'][2]),
                max(table_row_bbox[3], cells['bbox'][3]),
            ]
            
            real_bbox = []
            for span in cells['spans']:
                if span['text'] in answer or answer in span['text'] \
                    and ratio(span['text'], answer) > 0.45:
                    #vamos ajustar os valores de x e y para fora da tabela
                    #convertendo aos bbox das células de xyxy -> xywh
                    span_width = span['bbox'][2] - span['bbox'][0]
                    span_height = span['bbox'][3] - span['bbox'][1]
                    #deslocando os valores de x e y para referencial da página
                    real_bbox = [
                        span_bbox+table_bbox 
                        for span_bbox, table_bbox in zip(
                            span['bbox'][:2], table.table_block.bbox[:2])
                    ]
                    #montando a nova bounding boxes
                    real_bbox += [span_width, span_height]
                    answer_bboxes.append(real_bbox)
            
        if len(answer_bboxes) == 0:
            #calculando o width e heigth
            table_row_width = table_row_bbox[2] - table_row_bbox[0]
            table_row_height = table_row_bbox[3] - table_row_bbox[1]
            #ajustando as coordenadas das bbox em relação a folha. Tirando da relação com 
            #a tabela
            table_row_bbox = [
                span_bbox+table_bbox 
                for span_bbox, table_bbox in zip(
                    table_row_bbox[:2], table.table_block.bbox[:2])
            ]
            answer_bboxes.append(table_row_bbox + [table_row_width, table_row_height])
        
        return table.get_table_text(), answer_bboxes

    def parse_prompt_response(self, model_responses: list, 
                              text_blocks: list, table_segments: list) -> list:
        responses = []
        for qa in model_responses:
            # vamos separar cada pergunta e resposta
            if len(qa) <= 0:
                continue
            
            # pegando as pergunta, resposta e suas regiões no texto
            qa_splited = qa.split("|")[:3]
            if len(qa_splited) < 2:
                #não tem pergunta e nem resposta
                #não podemos aproveitar
                continue

            if len(qa_splited) < 3:
                #não tem região do texto informada pelo modelo
                #pode ser aproveitado
                qa_splited.append("")
            
            question, answer, base_region = tuple(qa_splited)
            question = question.split(":")[-1].strip()
            answer = answer.split(":")[-1].strip()
            base_region = base_region.split(":")[-1].strip()
            region = base_region.strip()
            
            aster_cleaner_pattern = re.compile(r'\*\*') #remoção de ** do Gemini
            question = aster_cleaner_pattern.sub('', question.strip())
            answer = aster_cleaner_pattern.sub('', answer.strip())
            base_region = aster_cleaner_pattern.sub('', base_region.strip().upper())

            response = PromptResponse(question, answer, base_region)
            has_table, has_text, region = self.__map_regions(base_region)
            line_bboxes = []
            base_text = ""
            if has_table or has_text: #tem informação de região
                #faz a verificação da informação caso tenha informado as coordenadas utilizadas
                if has_table:
                    for table_idx in region[0]:
                        if table_idx < len(table_segments):
                            #só vamos mapear caso o idx seja menor que o array
                            base_text, line_bboxes = self.__parse_table_response(
                                answer, region[1], 
                                table_segments[table_idx]
                            )
                
                elif has_text:
                    #definindo os paragrafos
                    for paragraph in region[0]:
                        if paragraph < len(text_blocks):
                            bboxes, text = self.__get_text_lines_bbox(answer, text_blocks, paragraph)
                            line_bboxes += bboxes
                            base_text += text

            response.add_answer_bbox(line_bboxes)
            response.add_base_text(base_text)
            
            responses.append(response)

        return responses