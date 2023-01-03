import numpy as np
from numpy.lib.stride_tricks import as_strided
# from numpy.lib.stride_tricks import sliding_window_view
# TODO: Back tracking needed to solve medium/hard level sudoku

board = np.array([[0, 0, 0, 2, 6, 0, 7, 0, 1],
                  [6, 8, 0, 0, 7, 0, 0, 9, 0],
                  [1, 9, 0, 0, 0, 4, 5, 0, 0],
                  [8, 2, 0, 1, 0, 0, 0, 4, 0],
                  [0, 0, 4, 6, 0, 2, 9, 0, 0],
                  [0, 5, 0, 0, 0, 3, 0, 2, 8],
                  [0, 0, 9, 3, 0, 0, 0, 7, 4],
                  [0, 4, 0, 0, 5, 0, 0, 3, 6],
                  [7, 0, 3, 0, 1, 8, 0, 0, 0]])



# print(board.strides)

def as_subsquares(board):
    """Iterate the subsquares of sudoku board."""
    S =board.itemsize
    ast = as_strided(board, shape=(3,3,3,3), strides=(S*27, S*3, S*9, S)).reshape(9,3,3)
    return ast.copy()

def as_board(subsqrs):
    a = subsqrs.reshape(3,3,3,3)
    a = np.hstack(a)
    b = np.hstack(a)
    return b.copy()
    

def as_subcuboids(cube):
    S = cube.itemsize
    cuboids = as_strided(cube, shape=(3,3,3,3,3,3), strides=( S*27, S*3, S*81*3, S*81, S*9, S)).reshape(9,9,3,3)
    return cuboids.copy()


def create_validation_cube():
    """Create validation cube"""
    corner_rod = np.arange(1,10)
    c2d = corner_rod[:,None]*np.ones((9,9))
    c3d = c2d[:,:,None]*np.ones((9,9,9))
    return c3d

def eliminate_cube(cube, board, pts):
    """Replace values of validity cube with 0 for elements that are occupied in board."""
    x,y = pts
    # Step 2 : Fetch each non zero element from board
    elm = board[x, y]
    # Step 3 : Make the rows and columns of plane  of the element as 0.
    # Step 3a : Fetch the plane of the element from validation cube.
    # elm_plane = cube[elm-1, ...]
    # Step 3b : Set the entire row and column where the element is present to 0 in the plane.
    # elm_plane[x,:]  = 0
    cube[elm-1,x, :] = 0
    # elm_plane[: ,y]  = 0
    cube[elm-1, :, y] = 0
    # Step 3c : Set the entire vertical array of cube containing the nonzero element to 0.
    cube[:, x, y] = 0
    # print(elm_plane)
    # Step 4 : Set the subsquare in the plane of element as 0
    # Step 4a : Calculate the index of subsquare where the element is present in board.
    start_x, start_y= x//3*3,  y//3*3
    end_x, end_y = start_x + 3, start_y + 3
    # print(board[start_x:end_x, start_y:end_y])
    # Step 4b : Set the subsquare to 0
    cube[elm-1, start_x:end_x, start_y:end_y] = 0
    

def validate_insertion(elm, pt, cube, board):
    x,y = pt
    bkp_brd_elm = board[x,y]
    board[x,y] = elm
    bkp_vcube_plane = cube[elm-1, ...]
    cube[elm-1, x, :] = 0
    cube[elm-1, :, y] = 0
    # Step 1 : Set the entire vertical array of cube containing the nonzero element to 0.
    bkp_vcube_xy = cube[:, x, y]
    cube[:, x, y] = 0
    # Step 2 : Set the subsquare in the plane of element as 0
    # Step 2a : Calculate the index of subsquare where the element is present in board.
    start_x, start_y= x//3*3,  y//3*3
    end_x, end_y = start_x + 3, start_y + 3
    # Step 2b : Set the subsquare to 0
    cube[elm-1, start_x:end_x, start_y:end_y] = 0
    cuboids = as_subcuboids(cube)
    bsqrs = as_subsquares(board)
    
    valid_insertion = True
    for bblck, cuboid in zip(bsqrs, cuboids):
        nonzero_elms = bblck[bblck!=0]
        absent_elms = np.setdiff1d(np.arange(9)+1, nonzero_elms)        
        # EXPLAINATION: Here the elements present in the board block are being checked.
        # For the planes of the cuboid  belonging to those nonzero elements, every element should be zero.
        # Inverting the boolean value for each plane of cuboid where any nonzero value is present, gives the value where every plane is zero.
        # Suming them gives the count of planes where every value is zero.
        count_present_elms_from_board_block = len(nonzero_elms)
        count_present_elms_from_cuboid = (~np.any(cuboid[nonzero_elms-1,...], axis=(1,2))).sum()
        # nonzero_elms_present_in_cube_plane_block = (np.bincount(cuboid[nonzero_elms-1,...].nonzero()[0]).size==0)
        if count_present_elms_from_cuboid != count_present_elms_from_board_block:
            valid_insertion = False
            break
        # EXPLAINATION: Here the elements absent in the board block are being checked.
        # For the planes of the cuboid belonging to those "ABSENT" elements, atleast one element should be non-zero.
        # np.any with axis as tuple will give boolean array for each plane where atleast one non zero element is present.
        # summing it should match the length of absent elements.
        count_absent_elms_from_cuboid = np.any(cuboid[absent_elms-1,...], axis=(1,2)).sum()
        count_absent_elms_from_board_block = np.count_nonzero(bblck==0)
        if count_absent_elms_from_cuboid != count_absent_elms_from_board_block:
            valid_insertion = False
            break
    
    cube[:,x,y] = bkp_vcube_xy
    cube[elm-1,...] = bkp_vcube_plane
    board[x,y] = bkp_brd_elm
    return valid_insertion

print(board)
print("Start")
prev_board = board.copy()
repeat = 0
step = 1

# Step 1 : Find all the non zero element indices the board    
non_zero_indices = np.transpose(prev_board.nonzero())
vcube = create_validation_cube()


while True:
    print(f"Step - {step}")
    
    for pt in non_zero_indices:
        eliminate_cube(vcube, prev_board, pt)
    # Step 6 :
    cuboids = as_subcuboids(vcube)
    subsqrs = as_subsquares(prev_board)
    new_non_zero_pts = []
    for sqr_idx, zipped in enumerate(zip(cuboids, subsqrs)):
        cuboid, sb_sqr = zipped
        # Step 6a : Calculate index of  plane in cuboid where count of nonzero elements is exactly 1.
        mask_count1 = (np.bincount(cuboid.nonzero()[0], minlength=9)==1)
        planes_with_1_elm = cuboid[mask_count1,...]
        plane_with_1_elm = np.sum(planes_with_1_elm, axis=0)
        zero_count = np.count_nonzero(plane_with_1_elm==0)
        if (plane_with_1_elm.size == 0) or (zero_count==9):
            continue
        # # Step 6b : Calculate indices of nonzero elements in that plane
        elm_plane_x, elm_plane_y = plane_with_1_elm.nonzero()
        # # Step 6c : Put that element using the index in that subsqr
        sb_sqr[elm_plane_x, elm_plane_y] = plane_with_1_elm[elm_plane_x, elm_plane_y]
        # Index of pts w.r.t. board
        brd_x = sqr_idx//3*3 + elm_plane_x
        brd_y = sqr_idx%3*3  + elm_plane_y
        new_non_zero_pts.append(np.transpose((brd_x, brd_y)))
        
    # Step 7 : Revert subsqrs to board.
    solved_board = as_board(subsqrs)
    print(solved_board)
    if len(new_non_zero_pts)>1:
       new_non_zero_pts = np.vstack(new_non_zero_pts)
       non_zero_indices = new_non_zero_pts
    count_zero = np.count_nonzero(solved_board==0)
    if count_zero == 0:
        break
    elif np.array_equal(solved_board, prev_board):
        repeat += 1
    else:
        print(f"Places left to be filled : {count_zero}")
        prev_board = solved_board
    if repeat != 0:
        print("Board repeated. Partial solution acheived.")
        print("Need backtracking solution.")
        break
    step += 1
