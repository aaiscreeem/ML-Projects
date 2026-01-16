from scoring import score_batch
from error_correction import SpellingCorrector
import numpy as np
import pandas as pd

from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

class Grader:

    def __init__(self, entry_number = "2022MT11257", test_path = './data/misspelling_public.txt', training_paths = ['./data/train1.txt', './data/train2.txt']):

        self.corrupt = []
        self.truth = []
        self.raw_training_texts = []
        self.entry_number = entry_number

        self.load_test_dataset(test_path)
        self.load_training_datasets(training_paths)
        self.make_corrector()


    def load_test_dataset(self, path: str = './data/misspelling_public.txt') -> None:

        with open(path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            try:
                t, c = line.strip().split('&&')
                self.corrupt.append(c)
                self.truth.append(t)
            except:
                pass

        # print a sample
        rid = np.random.randint(len(self.corrupt))
        print(self.corrupt[rid], self.truth[rid])

    def load_training_datasets(self, paths: List[str] = ['./data/train1.txt', './data/train2.txt']) -> None:
        for file in paths:
            with open(file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                self.raw_training_texts.append(line)

    def make_corrector(self) -> None :
        self.corrector = SpellingCorrector()
        self.corrector.fit(self.raw_training_texts)  #uncomment to train the corrector

    def grade(self, test_mode = False):
        print(f"Grading for {self.entry_number}")
        if test_mode:
            sc = score_batch(self.truth, self.truth)
            df = pd.DataFrame({'corrupt': self.corrupt, 'truth': self.truth, 'prediction': self.truth})
        else:

            predictions = self.corrector.correct(self.corrupt)
            sc = score_batch(predictions, self.truth)
            # logging predictions
            df = pd.DataFrame({'corrupt': self.corrupt, 'truth': self.truth, 'prediction': predictions})

        # Add score to the last row of the DataFrame
        score_row = pd.DataFrame({'corrupt': ['Score'], 'truth': [sc], 'prediction': [sc]})
        df = pd.concat([df, score_row], ignore_index=True)

        df.to_csv(f'predictions_{self.entry_number}.csv', index=False)
        print(f"Score: {sc}")
        return sc


if __name__ == "__main__":
    hallelujah = Grader()
    hallelujah.grade()
    
