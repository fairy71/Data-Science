import numpy as np
import pandas as pd

class GradeBook:
    def __init__(self, roster_csv, hw_csv, quiz_csv):
        # Load data
        self.roster = pd.read_csv(roster_csv)        # e.g. columns: student_id, name
        self.hw = pd.read_csv(hw_csv, index_col='student_id')    # hw scores, column per homework
        self.quiz = pd.read_csv(quiz_csv, index_col='student_id')  # quiz scores

        # Merge data into one DataFrame
        self.data = self.roster.set_index('student_id').join(self.hw).join(self.quiz)

    def normalize_scores(self):
        # Convert scores to 0–1 scale per homework / quiz
        for col in self.hw.columns:
            max_score = self.hw[col].max()
            if max_score > 0:
                self.data[col] = self.data[col] / max_score
        for col in self.quiz.columns:
            max_score = self.quiz[col].max()
            if max_score > 0:
                self.data[col] = self.data[col] / max_score

    def compute_weighted_grade(self, weights):
        """
        weights: dict, e.g. {'hw1':0.1, 'hw2':0.1, 'quiz1':0.05, 'quiz2':0.05, 'exam':0.6}
        The weights should sum to 1 (or close).
        """
        # Create a column "weighted_score" by summing weight * score
        total = np.zeros(len(self.data))
        for col, w in weights.items():
            if col in self.data.columns:
                total += self.data[col].fillna(0).to_numpy() * w
        self.data['weighted_score'] = total
        # Convert to percentage
        self.data['percentage'] = self.data['weighted_score'] * 100

    def assign_letter_grade(self):
        # Simple mapping from percentage to letter grade
        def letter(percent):
            if percent >= 90:
                return 'A'
            elif percent >= 80:
                return 'B'
            elif percent >= 70:
                return 'C'
            elif percent >= 60:
                return 'D'
            else:
                return 'F'
        self.data['letter_grade'] = self.data['percentage'].apply(letter)

    def save(self, filename):
        self.data.to_csv(filename)

if __name__ == "__main__":
    # Example usage:
    gb = GradeBook("roster.csv", "hw.csv", "quiz.csv")
    gb.normalize_scores()
    weights = {
        'hw1': 0.1, 'hw2': 0.1, 'quiz1': 0.05, 'quiz2': 0.05,
        # assume there's a column "exam" in hw or quiz CSVs
        'exam': 0.6
    }
    gb.compute_weighted_grade(weights)
    gb.assign_letter_grade()
    gb.save("final_grades.csv")
    print(gb.data.head())
