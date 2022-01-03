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

Object part articulation states and camera parameters used during the rendering of the input frames. Exported as
OrderedDict in yaml format. The parameters of the part articulations is from `pybullet.getLinkState`.

```yaml
- - - frame_0 (frame name)
      - - - obj
         - &id001 (specify part articulations)
            - - - 0 (part index)
               - - - 0
                  - vec3 linkWorldPosition
               - - - 1
                  - vec4 quaternion [x,y,z,w] linkWorldOrientation
               - - - 2
                  - vec3 localInertialFramePosition
               - - - 3
                  - vec4 quaternion [x,y,z,w] localInertialFrameOrientation
               - - - 4
                  - vec3 worldLinkFramePosition
               - - - 5
                  - vec4 quaternion [x,y,z,w] worldLinkFrameOrientation
            ...
            - - K (k parts)
      - - viewMat
            - - 16x1 column major matrix
      - - projMat
            - - 16x1 column major matrix
  - - ... (other frames)
  - - frame_n
      - - - obj
            - *id001 (reuse part articulations)
      - - viewMat
            - - 16x1 column major matrix
      - - projMat
            - - 16x1 column major matrix
```

#### Part Order

Part order maps links stored in urdf file to a specified order.  
For example, in `drawer:40453` object, it has valid links (without virtual link): `link_0`, `link_1`, `link_2`, `link_3`
, with link indices `[0,1,2,3]` in the urdf. The corresponding part order is `[3, 0, 1, 2]` as shown in the following
part order json file. Thus, the order of the links will be mapped to [`link_3`, `link_0`, `link_1`, `link_2`], with part
class `[0, 1, 2, 3]` as the part segmentation class.

```json
{
  "drawer (object category)": {
    "40453 (object id)": [3, 0, 1, 2],
    "46653": [0, 1, 2, 3],
    ...
  },
  ...
}
```

## Output

The output of stage1 preprocess contains:

* part mask frame
* single view point clouds:
    * points in camera coordinates
    * transformed part points at rest state in world coordinates
    * (K, 16) column major transformation matrix transforms points of K parts in camera space to rest state in world
    * (16, 1) column major transformation matrix transforms camera space to world space

All these data are stored in hdf5 format:

```yaml
drawer (object category):
  40453 (object id):
    0 (articulation id):
      0 (frame id):
        mask: (height, width, 1) part mask frame,
        points_camera: (height x width, 1) points in camera space,
        points_rest_state: (height x width, 1) points at rest state in world space,
        parts_transformation: (K, 16) column major transformation matrix,
        base_transformation: (16, 1) column major transformation matrix
      ... other frames
    ... other articulations
  ... other objects
... other categories
```