# Guide for Dataset Setup

To train a LMG model, you need (i) texture mesh and (ii) 2D images. For texture mesh, we suggest using texture mesh generated from [MILo](https://github.com/Anttwo/MILo), but we also encourage you to generate mesh from different methods. For 2D images, the images format follow [original 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) format.

```bash
mkdir dataset
```

## Texture Mesh

We provide a [sample texture mesh](https://drive.google.com/file/d/1jV0i3Kaj9JBlHTXzYtZvnsH2Mds86P1a/view?usp=sharing) of [hotdog scene](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4) for everyone to play with our project.

## 2D Images

Download NeRF synthetic dataset from [official google drive](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)

## Directory Structure

The dataset structure should like:   
```
.  
├── dataset  
│   ├── images  
│   │   ├── hotdog  
│   │   └── ...  
│   └── images
│       ├── hotdog
│       │   └── hotdog.ply
│       └── ...  
└── ...
```
