# from config import get_config
# from data.data_pipe import NewFaceDataset
# from Learner import face_learner
# from torchvision import transforms as trans
# from pathlib import Path
# from torch.utils.data import DataLoader

# # Assuming you already have `conf` defined somewhere in your code:
# conf = get_config(training=False)

# # Define the transformation for the new faces (similar to training transformation)
# new_faces_transform = trans.Compose([
#             trans.RandomHorizontalFlip(),
#             trans.ToTensor(),
#             trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])        

# # Provide the path to the directory containing new face images
# new_faces_path = Path("data/processed/")

# # Create the new Dataset and DataLoader
# new_dataset = NewFaceDataset(new_faces_path, transform=new_faces_transform)
# new_loader = DataLoader(new_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)

# #def new_face_loader_function():
# #   return new_loader,len(new_dataset)
    
    
# # Now, you can use `new_loader` for transfer learning in the `add_new_faces` function.

# # Assuming you have instantiated the face_learner class as `face_learner_instance`,
# # you can call the add_new_faces method as follows:

# face_l=face_learner(conf)

# face_l.add_new_faces(conf)


# Import necessary classes and functions from your code files
from config import get_config
from data.data_pipe import NewFaceDataset
from Learner import face_learner
from pathlib import Path
from torch.utils.data import DataLoader

def main():
    # Assuming you already have `conf` defined somewhere in your code:
    conf = get_config(training=False)

    # Define the transformation for the new faces (similar to training transformation)
    #new_faces_transform = ...

    # Provide the path to the directory containing new face images
    #new_faces_path = Path("data/processed/")

    # Create the new Dataset and DataLoader
    # new_dataset = NewFaceDataset(new_faces_path, transform=new_faces_transform)
    # new_loader = DataLoader(new_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)

    # Instantiate the face_learner class and call the add_new_faces method
    face_l = face_learner(conf)
    #face_l.add_new_faces(conf, new_loader, len(new_dataset))
    face_l.add_new_faces(conf)

if __name__ == "__main__":
    main()
