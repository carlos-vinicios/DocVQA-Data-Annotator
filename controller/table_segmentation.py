from bs4 import BeautifulSoup
import htmlmin

from model.doc_segment import DocSegment
from model.image_data import ImageData
from utils.image import crop_image

class TableSegmentation():

    def __init__(self, image_data: ImageData, table_block: DocSegment,
                 table_detector, table_id: int):
        self.table_detector = table_detector
        self.image_data = image_data
        self.table_block = table_block
        self.table_id = table_id
        self.table_structure, self.table_html, self.structure_score = self.__segment_table()
        self.table_html = self.__fix_html_header(self.table_html)
    
    def __segment_table(self) -> list:
        # vamos adicionar pad ao bloco para melhorar a qualidade da detecção
        pad = 30
        self.table_block.pad_block(pad)
        self.image_data.table_image = crop_image(
            self.image_data.image, self.table_block.get_crop_bbox()
        )
        return self.table_detector.run(
            self.image_data, 
            self.table_block.get_texts()
        )

    def __to_markdown(self, structure: str, show_line_number: bool = True) -> str:
        # Parse the HTML
        soup = BeautifulSoup(structure, 'html.parser')
        # Find the table
        table = soup.find('table')
        if table is None:
            return None
        # Extract rows and columns
        rows = table.find_all(['tr', 'thead'])
        # Iterate through rows
        markdown_table = ""
        for idx, row in enumerate(rows):
            # Extract cells
            cells = row.find_all(['th', 'td'])
            # Create Markdown row
            if show_line_number:
                markdown_row = f"{idx+1} |"
            else:
                markdown_row = "|"
            for cell in cells:
                markdown_row += cell.get_text().strip() + "|"
            markdown_table += markdown_row + "\n"
        return markdown_table
    
    def __fix_html_header(self, table_html):
        # Create a BeautifulSoup object
        soup = BeautifulSoup(table_html, 'html.parser')

        # Collect all thead contents
        all_thead_elements = soup.find_all('thead')
        new_thead = soup.new_tag('thead')

        # Create <tr> for each <thead> content and move it to the new <thead>
        for thead in all_thead_elements:
            new_tr = soup.new_tag('tr')
            for th in thead.find_all('th'):
                new_tr.append(th)
            new_thead.append(new_tr)
            thead.decompose()

        # Append the new_thead to the table
        table = soup.find('table')
        table.insert(0, new_thead)

        # Print the modified HTML
        return htmlmin.minify(soup.prettify())
    
    def __as_html(self, show_line_number: bool):
        if not show_line_number:
            return self.table_html

        # Create a BeautifulSoup object
        soup = BeautifulSoup(self.table_html, 'html.parser')

        # Find all <tr> elements
        all_tr_elements = soup.find_all('tr')

        # Add a line number to each <tr> element
        for i, tr in enumerate(all_tr_elements, start=1):
            if tr.parent.name == 'thead':
                new_tag = soup.new_tag('th', id='line-number')
            else:
                new_tag = soup.new_tag('td', id='line-number')
            
            new_tag.string = str(i)
            tr.insert(0, new_tag)
        
        return htmlmin.minify(soup.prettify())

    def get_table_text(self, as_markdown: bool = False, show_line_number: bool = True):
        map_str = ""
        if as_markdown:
            map_str = self.__to_markdown(self.table_html, show_line_number)
        else:
            #retorna a estrutura textual definina pelo table detector
            map_str = self.__as_html(show_line_number)
        
        return map_str