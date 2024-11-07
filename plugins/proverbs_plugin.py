from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel import Kernel
from typing import Annotated
import json
import random

class Proverb:
    def __init__(self, proverb: str, meaning: str):
        self.proverb = proverb
        self.meaning = meaning
    
    def __str__(self):
        return f"{self.proverb}: {self.meaning}"
    
    def __repr__(self):
        return str(self)
    
    def to_dict(self):
        return {
            "proverb": self.proverb,
            "meaning": self.meaning
        }


class ProverbsPlugin(KernelBaseModel):

    @classmethod
    async def create(cls, kernel: Kernel):
        self = cls(kernel)
        return self    

    def __init__(self, kernel: Kernel):
        super().__init__()
        self._kernel = kernel
        self._proverbs = self.__load_proverbs()

    @kernel_function( description="Returns a random proverb.", 
                     name="GetRandomProverb", )
    def get_random_proverb(self,     
                        number_of_proverbs: Annotated[int, "Number of proverbs to return."]=1,                      
                       ) -> Annotated[str, "Proverbs with its meanings."]:

        return self.__get_random_proverbs(number_of_proverbs)

    @kernel_function( description="Explain the meaning of a proverb.",
                        name="ExplainProverb", )
    def explain_proverb(self, 
                        proverb: Annotated[str, "Proverb to explain."],
                        ) -> Annotated[str, "Meaning of the proverb."]:
        instructions = f"""
            Provide the meaning of the proverb: {proverb}
            You can provide the meaning of the proverb in your own words.
            ONLY EXPLAIN THE PROVERB IF THERE'S REALLY SUCH A HUNGARIAN PROVERB THAT EXISTS."""
        return instructions
    
    @kernel_function( description="A game where the player has to guess the missing word in a proverb. Don't ask the number of proverbs to be used in the game.",
                        name="GameWordSubstitution", )
    def start_game_word_subsitution(self):
        # Get 5 random proverbs
        proverbs = self.__get_random_proverbs(5)
        instructions = f"""
            Proverbs to be used in the game: 
                {proverbs}
            Give the player one proverb at a time with the important word replaced by a blank. 
            Provide 3 possible words to pick from, not just let the player guess the word. But provide 3 plausible options.
            The player has to guess the missing word. If they don't get it right, you can ask them to guess again but with a hint but ultimately anwser it if the user is unable to guess it.
            If the player gets it right, you can provide the meaning of the proverb and move on to the next proverb.
            Only accept the anwser if it is an exact match (baring case sensitivity and smaller typos).
            ONCE A GOOD ANWSER IS PROVIDED, PROCEED ASKING THE NEXT PROVERB, DON'T ASK IF THE PLAYER WANTS TO CONTINUE WITH THE NEXT PROVERB.
            USE ALL THE PROVERBS PROVIDED IN THE GAME, DON'T SKIP ANY.
            ALWAYS PROVIDE 3 POSSIBLE OPTIONS TO PICK FROM.
            Once all the proverbs are done, you can provide the player with their score and provide a helpful improvement tip if needed."""
        return instructions
    
    @kernel_function( description="A game where the player has to guess the meaning of a proverb. Don't ask the number of proverbs to be used in the game.",
                        name="GameGuessMeaning", )
    def start_game_guess_meaning(self):
        # Get 5 random proverbs
        proverbs = self.__get_random_proverbs(5)
        instructions = f"""
            Proverbs to be used in the game: {proverbs}
            Give the player one proverb at a time with the meaning of the proverb replaced by a blank. 
            Provide 3 possible meanings to pick from, not just let the player guess the meaning. But provide 3 plausible options.
            The player has to guess the meaning of the proverb. If they don't get it right, you can ask them to guess again but with a hint but ultimately anwser it if the user is unable to guess it.
            If the player gets it right, you can provide the meaning of the proverb and move on to the next proverb.
            Once all the proverbs are done, you can provide the player with their score and provide a helpful improvement tip if needed."""
        
        return instructions

    
    # --- Helper methods ---

    def __get_random_proverbs(self, number_of_proverbs: int):
        # Get a random sample of proverbs
        proverbs = random.sample(self._proverbs, number_of_proverbs)
        proverbs = [proverb.to_dict() for proverb in proverbs]
        return proverbs

    def __load_proverbs(self):
        # load the contents of the proverbs.json file from the prompts folder
        with open('prompts/proverbs.json', 'r', encoding="utf8") as file:
            proverbs = json.load(file)
        
        # map the contents of the proverbs.json file to an array of Proverb objects (properties proverb and meaning)
        proverbs = [Proverb(proverb['proverb'], proverb['meaning']) for proverb in proverbs]
        return proverbs

    