# Preprocess Stage1

## Input

### Files Structure

```angular2html
├── datasets
   ├── sapien (dataset name)
      ├── render
         ├── drawer (category name)
            ├── 40453 (object id)
            ├── ...
            ├── 46653
               ├── 0 (articulation id)
                  ├── depth
                     ├── 000000.h5 (depth buffer)
                     ├── 000001.h5
                     ├── ...
                  ├── mask
                     ├── 000000.png (part mask)
                     ├── 000001.png
                     ├── ...
                  ├── gt.ymal (metadata: camera and articulation parameters)
               ├── 1
               ├── ...
         ├── ... (other categories)
      └── urdf
         ├── drawer (category name)
            ├── 40453 (object id)
            ├── ...
            ├── 46653
               ├── mobility.urdf (object urdf file)
         ├── ... (other categories)
      └── part_order.json
   ├── ... (other datasets)
```

### Formats

#### Depth Input

Depth buffer (z buffer) represents the depth information of the single view point cloud in camera coordinates. In
shape (height, width, 1), the range of the values is float32 numbers in [0, 1], stored in hdf5 format with "data" key in
hdf5 dataset field.

#### Mask Input

Mask of parts in the urdf objects. In shape (height, width, 1), the range of the values is uint8 numbers in [0, 255].
The value `255` represents empty space, associated with invalid depth values. The valid values in mask is in
range [0, K(number of parts)]. The value represents the indices of the links stored in the urdf file.  
:warning: some urdf files have virtual link without visual property, thus the mask values is in range [1, K].

#### Metadata Input

#### Part Order

## Output