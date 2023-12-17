import torch

class Parser:
    def __init__(self, data, submissions):
        self.data = data
        self.submissions = submissions
        self.parsed_data = []

    def parse(self):
        key_list = list(self.submissions.keys())
        value_list = list(self.submissions.values())
        for submission in self.data:
            index = value_list.index(submission)
            self.parsed_data.append(key_list[index])

        return self.parsed_data
    
