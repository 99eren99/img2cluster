from transformers import CLIPModel,CLIPProcessor
import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


class imageDataset(Dataset):

    def __init__(
        self, 
        image_paths
    ):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        path=self.image_paths[index]
        image=Image.open(r'{}'.format(path)).resize((336,336))
        if image.mode!="RGB":
            try:
                image=image.convert("RBG")
            except:
                image=rgba2rgb(np.array(image))
                return image, path
        return np.array(image)


def get_data_loader(imagePaths,batch_size,num_workers):
    data_loader = DataLoader(
        imageDataset(imagePaths), 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=num_workers,
        shuffle=False
    )
    return data_loader


def save_embeddings(imagePaths,batchSize,nDataLoaderWorkers):
    preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to("cuda")
    model.eval()

    dataLoader=get_data_loader(imagePaths,batchSize,nDataLoaderWorkers)
    print("Total epochs:",len(dataLoader))

    allEmbeddings=[]

    start = time.time()
    for batchID, data in enumerate(dataLoader):
        samples = data
        samples=preprocessor(text=[""],images=samples, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = model(**samples)
            imageEmbeddings=outputs["image_embeds"].detach().cpu().numpy()
        
        allEmbeddings.append(imageEmbeddings)
        
        end = time.time()
        print(f"Batch ID:{batchID} - Time Ellapsed: {end-start}",flush=True)
        start = time.time()


    allEmbeddings=np.concatenate(allEmbeddings, axis=0)
    np.save("embeddings.npy",allEmbeddings)
    return allEmbeddings

