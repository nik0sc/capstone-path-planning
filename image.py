from PIL import Image
import yaml
import numpy as np
from typing import Tuple, Sequence, Dict, Set
import itertools as it
import networkx as nx
from operator import itemgetter
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt

"""
You will need: PIL, numpy, networkx, matplotlib, pygraphviz
Pygraphviz requires libgraphviz-dev package on ubuntu

Bibliography

Planning Algorithms online textbook
http://planning.cs.uiuc.edu/node260.html
 6.2 Polygonal Obstacle Regions 
    6.2.1 Representation
    6.2.2 Vertical Cell Decomposition
        Defining the vertical decomposition
        General position issues
        Defining the roadmap
        Solving a query
        Computing the decomposition
        Plane-sweep principle
        Algorithm execution 
    6.2.3 Maximum-Clearance Roadmaps
    6.2.4 Shortest-Path Roadmaps 
http://planning.cs.uiuc.edu/node352.html
 Boustrophedon decomposition 

Coverage of Known Spaces: The Boustrophedon Cellular Decomposition (use library acct; Choset)
https://doi.org/10.1023/A:1008958800904 

*** Coverage Path Planning: The Boustrophedon Cellular Decomposition (Choset & Pignon)
https://pdfs.semanticscholar.org/8df2/e8ed0410c3ae5b0f56c4367b740e91c23d73.pdf

*** Optimal Coverage of a Known Arbitrary Environment (Mannadiar & Rekleitis)
http://msdl.cs.mcgill.ca/people/raphael/files/yiannis_coverage.pdf

Robotic Motion Planning: Cell Decompositions (lecture slides; Choset)
https://www.cs.cmu.edu/~motionplanning/lecture/Chap6-CellDecomp_howie.pdf

Trapezoidal decomposition (author unk)
https://www2.cs.duke.edu/courses/spring02/cps237/DKnotes/n19.pdf

"""

# Some useful type annotations
BoundaryList = Sequence[Tuple[int]]
AdjacencyList = Dict[int, Set[int]]

def load_image(name: str) -> Tuple[np.ndarray, Dict]:
    """
    Load and return image bitmap and yaml.
    """
    try: 
        with open(name + ".yaml") as f:
            config = yaml.safe_load(f)
    except:
        print("\x1b[41mCannot open yaml\x1b[0m")
        raise
    
    try:
        image = Image.open(name + ".pgm")
    except:
        print("\x1b[41mCannot open pgm\x1b[0m")
        raise
    
    arr = np.array(image.getdata(), dtype="uint8")
    arr.shape = image.size
    # image is transposed so that a[X,Y] indexing works as expected, but
    # more importantly, so that it is stored in column major order so when
    # we iterate over the array, we get columns and not rows
    arr = arr.T

    return arr, config


def slice_count(slic: np.ndarray, threshold: int) -> Tuple[int, BoundaryList]:
    """
    Count the segments in a slice of the bitmap. A pixel with value >= the 
    threshold is considered to be unobstructed. Returns the connectivity and
    a tuple of start and end pairs of each segment, exclusive of the obstacle 
    parts of the slice.

    So, if the slice looks like this:
    
     0 1 2 3 4 5 6 7 8 9
    +-+-+-+-+-+-+-+-+-+-+
    | |x|x|x|x| | |x|x|x|
    +-+-+-+-+-+-+-+-+-+-+

    the return value is:
    (2, ((1,4), (7,9)))
    """
    slice_begin = []
    slice_end = []
    for i in range(len(slic)-1):
        if slic[i+1] >= threshold and slic[i] < threshold:
            # begin slice (append the coord of the open pixel)
            slice_begin.append(i+1)
        
        if slic[i+1] < threshold and slic[i] >= threshold:
            # end slice (again, open pixel)
            slice_end.append(i)
    
    # reconcile slice begin and end
    if len(slice_begin) != len(slice_end):
        raise ValueError("Unterminated slices")

    return len(slice_begin), tuple(zip(slice_begin, slice_end))


def sweep(arr: np.ndarray, threshold: int) \
        -> Tuple[Sequence[int], Sequence[BoundaryList]]:
    """
    Sweep the bitmap to find out connectivity and segment boundaries for each
    slice. This is basically slice_count but for each slice in a 2d array.
    """
    slices = []
    boundaries = []

    for slic in arr:
        connectivity, slice_bound = slice_count(slic, threshold)
        slices.append(connectivity)
        boundaries.append(slice_bound)

    return slices, boundaries


def do_segments_overlap(left, right):
    if left[0] <= right[1] <= left[1]:
        return True
    elif left[0] <= right[0] <= left[1]:
        return True
    elif right[0] <= left[1] <= right[1]:
        return True
    elif right[0] <= left[0] <= right[1]:
        # probably not needed
        return True
    else:
        return False


def find_adjacents(left: BoundaryList, right: BoundaryList) -> AdjacencyList:
    """
    Given left and right lists of segments, determine the adjacency relations
    between segments on the left and right. 

    Input: 
        left: ((0, 10), (20, 30), (33, 36))
        right: ((0, 10), (20, 23), (27, 29), (40, 50))
    Output: {0: {0}, 1: {1, 2}, 3: {}, None: {3}}

    Note that keys in the output use the indices from the left list, and the 
    values use the indices from the right. The None key represents no 
    connecting segment on the left, and the empty set represents no connecting
    segment on the right.
    """
    # Handle special cases first
    if len(left) == 0 and len(right) == 0:
        return {}
    
    if len(left) == 0:
        # right is not [], so all right segments are gained
        return {None: tuple(range(len(right)))}
    
    if len(right) == 0:
        # left is not [], so all left segments are lost
        return {x: tuple() for x in range(len(left))}
    
    # Start at the topmost segments for left and right
    li = 0
    ri = 0

    out = defaultdict(set)

    # This is set to false when we reach the bottommost segments on the left 
    # and right
    lrun = True
    rrun = True

    while lrun and rrun:
        if do_segments_overlap(left[li], right[ri]):
            # overlap means right segment is reachable from left
            out[li].add(ri)

        # Which segment ends first?
        if left[li][1] < right[ri][1]:
            # Current left segment ends first, advance left if possible
            if li + 1 < len(left):
                li += 1
            else:
                # Done for the left
                lrun = False
        else:
            # Right segment ends first (or, they end at the same time)
            if ri + 1 < len(right):
                ri += 1
            else:
                # Done for the right
                rrun = False
    
    # Check for unconnected segments on the left and add them back
    for lj in range(len(left)):
        if lj not in out:
            out[lj] = set()
    
    # ...And on the right
    right_union = reduce(lambda x, y: x.union(y), out.values())
    for rj in range(len(right)):
        if rj not in right_union:
            out[None].add(rj)

    return out


def find_events_from_adjlist(adj: AdjacencyList) -> Dict[str, AdjacencyList]:
    """
    Given an AdjacencyList from find_adjacents, determine the "events" occuring
    at the transition from left to right.

    There are five kinds of events:
    - "Continue" a segment to another. This is the degenerate case, nothing 
      interesting happens in this case and the overall Reeb graph doesn't need
      to be updated. So we won't report this either.
    
    The other four will result in updating the Reeb graph:
    - "Split" of a segment into two
    - "Merge" of two segments into one
    - "Loss" of a segment (as in a concave shape)
    - "Gain" of a segment (as in a concave shape)
    """
    # reverse the adjacency list. Losses will be, uh, lost, but we just want 
    # to find out the merge events from this
    rev = defaultdict(set)
    for k, v_set in adj.items():
        for v in v_set:
            rev[v].add(k)
    
    # The events can be recognised from certain patterns in the adjacency 
    # relationship
    split = {k: v_set for k, v_set in adj.items() if len(v_set) > 1}
    merge = {k: v_set for k, v_set in rev.items() if len(v_set) > 1}
    loss = {k: v_set for k, v_set in adj.items() if len(v_set) == 0}
    gain = {k: v_set for k, v_set in adj.items() if k is None}

    out = {
        "split": split,
        "merge": merge,
        "loss": loss,
        "gain": gain
    }

    # Remove events that didn't happen
    return {k: v for k, v in out.items() if len(v) > 0}


def cell_adj_to_graph(adjs: Sequence[AdjacencyList]):
    """
    Convert a sequence of cell adjacency lists to a graph.
    Graph only! No cell boundary yet.

    Since changes to the Reeb graph only occur at changes of connectivity as we 
    sweep the bitmap, we should be able to construct the Reeb graph given a 
    sequence of cell adjacency lists, which describe what is happening at the 
    places where connectivity changes.
    """
    graph = nx.DiGraph()
    # the frontier maps numbers in the event adjacency list to actual
    # node numbers in the graph
    frontier = []
    # Next available node number
    i = 0

    # Setup: Populate the initial frontier and leftmost nodes
    for _ in adjs[0]:
        graph.add_node(i)
        frontier.append(i)
        i += 1

    for adj in adjs:
        # These events are how we update the Reeb graph. An event means a 
        # change in connectivity that will result in cells changing.
        events = find_events_from_adjlist(adj)
        for event_name, event_adj in events.items():
            assert len(event_adj) == 1, "too many conn. changes"

            if event_name == "split":
                [[left, right]] = event_adj.items()
                # Get the node number of the left cell.
                # This cell is going to split into two, so it will be replaced
                # in the frontier by its descendant cells.
                pred = frontier.pop(left)
                for succ in right:
                    # Add edges from left to right, and update the frontier to
                    # include the descendant cells and their node numbers
                    graph.add_edge(pred, i)
                    frontier.insert(succ, i)
                    i += 1

            elif event_name == "merge":
                # The order is inverted for the adjacency relation
                # for easier representation
                [[right, left]] = event_adj.items()
                # need to delete from the frontier in reverse order,
                # so that we don't disturb the other elements as we delete 
                # the ones in front
                preds = [frontier.pop(pred)
                         for pred in sorted(left, reverse=True)]

                # Add the edges from left to right
                for pred in preds:
                    graph.add_edge(pred, i)
                # Only one new node is created, and only one descendant cell
                # is inserted into the frontier
                frontier.insert(right, i)
                i += 1

            elif event_name == "loss":
                [[left, _]] = event_adj.items()
                # No changes to the graph needed, but this node is removed
                # permanently from the frontier
                frontier.pop(left)

            elif event_name == "gain":
                [[_, right]] = event_adj.items()
                # No edges added to graph yet
                for succ in right:
                    frontier.insert(succ, i)
                    graph.add_node(i)
                    i += 1

            else:
                raise NotImplementedError()
    
    return graph


def reeb_it(arr: np.ndarray, start: Sequence[int], threshold: int):
    """
    Construct the Reeb graph from the bitmap.
    """
    # slice connectivity
    slice_conn = [0]
    slice_conn_groups = [None]
    sweep_adjacency_lists = []

    slices, boundaries = sweep(arr, threshold)
    # The x coordinate of the current slice. This is updated as the slice
    # moves and it is added to the sweep_adjacency_list_xcoords list.
    # The critical points lie between this x coordinate and the one on the left
    sweep_adjacency_list_xcoords = []
    current_x = 0

    # groupby collapses continuous runs of items into a single element.
    # >>> [(k, ''.join(g)) for k, g in it.groupby("AAAABBCCCBDD")]
    # [('A', "AAAA"), ('B', "BB"), ('C', "CCC"), ('B', "B"), ('D', "DD")]
    for k, g in it.groupby(zip(slices, boundaries), key=itemgetter(0)):
        if k == 0:
            # Assume that regions with no segments are the borders of the 
            # floor space
            #sweep_adjacency_list_xcoords.append(current_x)
            current_x += len(list(g))
            # no segments, no problem
            continue

        slice_conn.append(k)
        list_g = [bounds for _, bounds in g]
        slice_conn_groups.append(list_g)

        if slice_conn[-2] == 0:
            # First splitting-up from left bound, trivial
            sweep_adjacency_list_xcoords.append(current_x)
            current_x += len(list_g)
            continue
        elif slice_conn[-2] > k:
            # a new cell added
            assert slice_conn[-2] - k == 1, "too many conn. changes"
        elif slice_conn[-2] < k:
            # two cells merged into one
            assert k - slice_conn[-2] == 1, "too many conn. changes"
        else:
            assert False, "groupby broke :<"
        
        end_left = slice_conn_groups[-2][-1]
        start_right = list_g[0]
        adjacency = find_adjacents(end_left, start_right)

        sweep_adjacency_lists.append(adjacency)
        sweep_adjacency_list_xcoords.append(current_x)

        current_x += len(list_g)
    
    sweep_adjacency_list_xcoords.append(current_x)

    graph = cell_adj_to_graph(sweep_adjacency_lists)        

    return graph


if __name__ == "__main__":
    arr, config = load_image("test")
    graph = reeb_it(arr, (0,0), 250)
    pos = nx.drawing.nx_agraph.pygraphviz_layout(graph, prog="dot")
    nx.draw(graph, with_labels=True, cmap=plt.cm.Paired, node_color=range(12), 
            node_size=800, pos=pos)
    plt.show()
    print("Break here")