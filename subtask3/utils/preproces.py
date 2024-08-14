class DiagPreprocessor:
    def __init__(self, num_rank: int, num_coordi: int):
        self.num_rank = num_rank
        self.num_coordi = num_coordi

    def preprocess(self, train_dataset, img2id, id2img, img_similarity):

        for i in range(len(train_dataset)):
            