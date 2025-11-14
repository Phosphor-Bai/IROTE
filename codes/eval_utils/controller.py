# from typing import List, Tuple



class Controller:
    @classmethod
    def get_multi_shot_instruction(cls, shot_str:str) -> str:
        return f"""Given the following insights about me:
{shot_str}
Please make the following responses strictly align with these insights. \n\n"""


    @classmethod
    def get_single_trait_instruction(cls, trait:str) -> str:
        return f"""Please make the following responses strictly align with the following trait: {trait}.\n\n"""

