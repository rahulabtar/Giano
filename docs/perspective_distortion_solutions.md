# Solutions for Perspective Distortion in Bird's-Eye View

## Problem
When the camera is not directly above the keyboard, the perspective transformation introduces non-uniform scaling in both x and y directions. This causes:
- X-coordinate distortion occurs towards the edges of the frame, due to camera positioning 
- Distance calculations become biased

## Solutions

### 1. Physical Solutions (Best Results)

#### A. Reposition Camera
- **Move camera directly above keyboard**: Eliminates perspective distortion entirely
- **Pros**: Perfect solution, no software needed
- **Cons**: May not be practical due to space constraints, lighting, or mounting limitations

#### B. Use Wider Angle Lens
- **Install wider FOV lens**: Reduces perspective effects
- **Pros**: Less distortion, larger field of view
- **Cons**: Requires hardware change, may introduce barrel distortion

#### C. Multiple Cameras
- **Use stereo vision or multiple viewpoints**: Combine views for better accuracy
- **Pros**: Can correct for perspective, more robust
- **Cons**: Complex setup, requires calibration, more processing

### 2. Software Solutions (Implemented)

#### A. Scale Factor Correction
**Method**: `find_closest_key_with_full_correction()`

Computes scale factors for x and y directions from the transformation matrix and applies them to normalize coordinates.

**How it works**:
- Analyzes how unit vectors transform through the perspective matrix
- Computes correction factors to normalize x and y scales
- Applies corrections when computing distances

**Usage**:
```python
midi_note = finger_aruco.find_closest_key_with_full_correction(x_px, y_px)
```

**Pros**: 
- Accounts for both x and y distortion
- Automatic computation from transformation matrix
- No manual calibration needed

**Cons**: 
- Approximate solution (doesn't fully eliminate distortion)
- Keys need to be in corrected coordinate space for best results

#### B. Boundary Distance Method
**Method**: `find_closest_key_by_boundary_distance()`

Uses distance to polygon boundary instead of centroid distance.

**Pros**: 
- More accurate for irregular shapes
- Accounts for actual key boundaries

**Cons**: 
- Still affected by coordinate system distortion
- More computationally expensive

#### C. Coordinate Transformation Correction
**Method**: `transform_point_with_correction()`

Applies scale factors directly to transformed coordinates.

**Usage**:
```python
corrected_x, corrected_y = finger_aruco.transform_point_with_correction(x_px, y_px)
```

**Pros**: 
- Can be used for any coordinate-based operations
- Corrects both x and y

**Cons**: 
- Requires keys to also be in corrected space for consistency

### 3. Advanced Software Solutions (Not Yet Implemented)

#### A. Calibrated Transformation
- Use known physical dimensions of keys to calibrate the transformation
- Create a lookup table mapping physical positions to bird's-eye coordinates
- **Pros**: Very accurate if calibration is good
- **Cons**: Requires manual calibration, sensitive to setup changes

#### B. 3D Reconstruction
- Use camera pose (from ArUco markers) to reconstruct 3D positions
- Project to true bird's-eye view using 3D→2D projection
- **Pros**: Most accurate, accounts for all perspective effects
- **Cons**: Complex, requires accurate camera pose estimation

#### C. Affine Transformation Instead of Perspective
- Use affine transformation if markers form a parallelogram
- **Pros**: Preserves parallel lines, uniform scaling
- **Cons**: Only works if camera is at sufficient distance/angle

#### D. Post-Processing Correction Grid
- Create a correction grid based on known test points
- Interpolate corrections for arbitrary points
- **Pros**: Can handle complex distortions
- **Cons**: Requires calibration procedure, storage overhead

## Recommendations

### For Immediate Use:
1. **Try `find_closest_key_with_full_correction()`** - This accounts for both x and y distortion
2. **Test with your setup** - See if the correction factors improve accuracy
3. **Consider physical repositioning** - Even a small adjustment toward overhead can help significantly

### For Best Results:
1. **Reposition camera** to be as close to directly overhead as possible
2. **Use `find_closest_key_with_full_correction()`** as a software fallback
3. **Consider wider angle lens** if repositioning isn't possible

### Implementation Notes:
- Scale factors are computed automatically from the transformation matrix
- Factors are cached for efficiency
- Correction can be enabled/disabled with `_use_coordinate_correction` flag
- For best results, keys should be stored in corrected coordinate space

## Testing
To test which method works best for your setup:
1. Move finger directly upward from a key
2. Check if y-coordinate in bird's-eye view changes
3. Try different methods and compare results
4. Measure accuracy of key detection with each method

