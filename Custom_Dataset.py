class CustomDataset(Dataset):
    def __init__(self, df, transform, data_type):
        self.df = df
        self.data_type = data_type

        if self.data_type == "train":
            self.image_paths = df['image_name']
            self.labels = df['label']
        if self.data_type == "test":
            self.image_paths = df[0]

        self.transform= transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        if self.data_type == "train":
            image = cv2.imread(f"/content/train/{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = self.labels[index]
            label = torch.tensor(label, dtype=torch.long)

            image = self.transform(image=image)["image"]
            return image, label

        if self.data_type == "test":
            image = cv2.imread(f"/content/test/{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.transform(image=image)["image"]

            return image
