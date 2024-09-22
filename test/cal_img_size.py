import math

def calculate_image_size(eye, at, up):
    # Calculate the direction vector from eye to at
    direction = [at[i] - eye[i] for i in range(3)]
    
    # Calculate the distance from eye to at
    distance = math.sqrt(sum(d**2 for d in direction))
    
    # Normalize the direction vector
    direction = [d / distance for d in direction]
    
    # Calculate the right vector (cross product of direction and up)
    right = [
        direction[1] * up[2] - direction[2] * up[1],
        direction[2] * up[0] - direction[0] * up[2],
        direction[0] * up[1] - direction[1] * up[0]
    ]
    
    # Normalize the right vector
    right_length = math.sqrt(sum(r**2 for r in right))
    right = [r / right_length for r in right]
    
    # Calculate the actual up vector (cross product of right and direction)
    actual_up = [
        right[1] * direction[2] - right[2] * direction[1],
        right[2] * direction[0] - right[0] * direction[2],
        right[0] * direction[1] - right[1] * direction[0]
    ]
    
    # Calculate the field of view
    fov = math.pi / 2  # Assuming a 90-degree field of view
    
    # Calculate the width and height of the image plane
    width = 2 * distance * math.tan(fov / 2)
    height = width  # Assuming a square image
    
    return width, height

# Camera setups
setups = [
    {"eye": [1, 0, 0.2], "at": [0, 0, 0], "up": [0, 0, 1]},
    {"eye": [0, 0.5, 0.1], "at": [0, 0, 0], "up": [0, 0, 1]},
    {"eye": [0, -1, 0], "at": [0, 0, 0], "up": [0, 0, 1]}
]

for i, setup in enumerate(setups):
    width, height = calculate_image_size(**setup)
    print(f"Setup {i + 1}:")
    print(f"Image size needed: {width:.2f} x {height:.2f}")
    print()