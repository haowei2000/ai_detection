
from pathlib import Path

import pandas as pd


class EveryaiDataset:
    def __init__(self, records:pd.DataFrame,ai_list:list[str],language:str='English'):
        self.records = records
        self.questions = records['question']
        self.human_responses = records['human']
        for ai in ai_list:
            self.ai_responses[ai] = records[ai]
        self.language = language
        self.max_length = 0
        self.min_length = 0

    def get_records_with_1ai(self, ai_name:str):
        return self.records[['question','human',ai_name]].to_records()
    
    def insert_ai_response(self,question, ai_name:str, ai_response:str):
        self.records.loc[self.records['question']==question, ai_name] = ai_response

    def insert_human_response(self,question, human_response:str):
        self.records.loc[self.records['question']==question, 'human'] = human_response

    def save(self, file_name:str|Path=Path(__file__).parent/'data.csv', format:str='csv'):
        if isinstance(file_name, str):
            file_name = Path(file_name)
        if file_name.suffix != f'.{format}':
            file_name = file_name.with_suffix(f'.{format}')
        self.records.to_csv(file_name)
