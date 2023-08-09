from config import get_config
from Learner import face_learner
import argparse

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='emore', type=str)
    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    learner = face_learner(conf)

    learner.train(conf, args.epochs)




















# from Learner import face_learner
# import argparse

# # python train.py -net mobilefacenet -b 200 -w 4

# import os
# from PIL import Image
# import numpy as np
# from data_pipe import de_preprocess

# def load_new_faces_data(new_data_path):
#     new_data = []  # List to store face embeddings (numpy arrays) of new faces
#     new_names = []  # List to store corresponding names of new faces

#     # Get a list of all subdirectories (each subdirectory represents a different person)
#     person_dirs = [d for d in os.listdir(new_data_path) if os.path.isdir(os.path.join(new_data_path, d))]

#     for person_dir in person_dirs:
#         # Get the path to the person's directory
#         person_path = os.path.join(new_data_path, person_dir)

#         # Get a list of all image files in the person's directory
#         image_files = [f for f in os.listdir(person_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

#         for image_file in image_files:
#             # Load and process the face image using the data_pipe.de_preprocess function
#             image_path = os.path.join(person_path, image_file)
#             img = Image.open(image_path).convert('RGB')
#             face_embedding = de_preprocess(img)

#             # Use the person's directory name as the name of the face
#             face_name = person_dir

#             new_data.append(face_embedding)
#             new_names.append(face_name)

#     return new_data, new_names


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='for face verification')
#     parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
#     parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
#     parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
#     parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
#     parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
#     parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
#     parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat,processed]",default='processed', type=str)
#     parser.add_argument("-nd", "--new_data_path", help="path to the new faces data", type=str)

#     args = parser.parse_args()
#     conf = get_config()
    
#     if args.net_mode == 'mobilefacenet':
#         conf.use_mobilfacenet = True
#     else:
#         conf.net_mode = args.net_mode
#         conf.net_depth = args.net_depth    
    
#     conf.lr = args.lr
#     conf.batch_size = args.batch_size
#     conf.num_workers = args.num_workers
#     conf.data_mode = args.data_mode
    
#     # Load the facebank (existing faces data) and the new faces data
#     facebank, names = load_facebank(conf)
#     new_data, new_names = load_new_faces_data(args.new_data_path)

#     # Combine the facebank with the new faces data
#     conf.facebank = facebank + new_data
#     conf.facebank_names = names + new_names

#     # Initialize the face_learner with the new data
#     learner = face_learner(conf, new_data=new_data, new_names=new_names)
    
#     # Train the model using the new data
#     learner.train(conf, args.epochs)
